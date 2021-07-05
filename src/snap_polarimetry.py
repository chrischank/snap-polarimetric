"""
This module is the main script for applying pre-processing steps based on
SNAP software on Sentinel 1 L1C GRD images.
"""
import copy
import os
import shutil
import sys
import xml.etree.ElementTree as Et
from pathlib import Path
from string import Template
from typing import List

import numpy as np
import rasterio
from geojson import Feature, FeatureCollection
from shapely.geometry import shape

from blockutils.common import update_extents
from blockutils.raster import is_empty
from blockutils.blocks import ProcessingBlock
from blockutils.datapath import set_data_path
from blockutils.exceptions import SupportedErrors, UP42Error
from blockutils.logging import get_logger
from blockutils.stac import STACQuery

# constants
LOGGER = get_logger(__name__)
PARAMS_FILE = os.environ.get("PARAMS_FILE")
GPT_CMD = "{gpt_path} {graph_xml_path} -e {source_file}"
TEMPLATE_XML = "template/snap_polarimetry_graph.xml"


# pylint: disable=unnecessary-pass
class WrongPolarizationError(ValueError):
    """
    This class passes to the next input file, if the current input file
    does not include the polarization.
    """

    pass


class SNAPPolarimetry(ProcessingBlock):
    """
    Polarimetric data preparation using SNAP
    """

    def __init__(self, params):
        # the SNAP xml graph template path

        params = STACQuery.from_dict(params, lambda x: True)
        params.set_param_if_not_exists("calibration_band", ["sigma"])
        params.set_param_if_not_exists("speckle_filter", True)
        params.set_param_if_not_exists("linear_to_db", True)
        params.set_param_if_not_exists("clip_to_aoi", False)
        params.set_param_if_not_exists("mask", None)
        params.set_param_if_not_exists("tcorrection", True)
        params.set_param_if_not_exists("polarisations", ["VV"])
        self.params = params

        self.path_to_template = Path(__file__).parent.joinpath(TEMPLATE_XML)

        # the temporary output path for the generated SNAP graphs
        self.path_to_tmp_out = Path("/tmp")

    @staticmethod
    def validate_polarisations(req_polarisations: list, avail_polarisations: list):
        """
        Check if requested polarisations are available
        """

        available = True
        for pol in req_polarisations:
            available = available and (pol in avail_polarisations)

        return available

    @staticmethod
    def safe_file_name(feature: Feature) -> str:
        """
        Returns the safe file name for the given feature (e.g. <safe_file_id>.SAFE)
        """

        safe_file_id = feature.properties.get("up42.data_path")
        safe_path = Path("/tmp/input").joinpath(safe_file_id)

        return list(safe_path.glob("*.SAFE"))[0].name

    @staticmethod
    def extract_polarisations(safe_file_path: Path):
        """
        This methods extract the existing polarisations from the input data.
        """

        tiff_file_list = list(safe_file_path.joinpath("measurement").glob("*.tiff"))

        pols = [
            (str(tiff_file_path.stem).split("-")[3]).upper()
            for tiff_file_path in tiff_file_list
        ]

        return pols

    def safe_file_path(self, feature: Feature) -> Path:
        """
        Returns the safe file path for the given feature
        (e.g. /tmp/input/<scene_id>/<safe_file_id>.SAFE)
        """

        safe_file_id = feature.properties.get("up42.data_path")

        return Path("/tmp/input/").joinpath(safe_file_id, self.safe_file_name(feature))

    def manifest_file_location(self, feature: Feature) -> Path:
        """
        Generates the manifest.safe file location for a given feature
        Looks up any *.SAFE files within the feature folder
        (expects one file to be present)
        """

        return self.safe_file_path(feature).joinpath("manifest.safe")

    def process_template(self, substitutes: dict) -> str:
        """
        Processes the snap default template and substitutes
        variables based on the given substitutions
        """
        src = self.path_to_template
        path_to_temp = Path(__file__).parent.joinpath("template/")

        shutil.copy(
            src, Path(path_to_temp).joinpath("snap_polarimetry_graph_%s.xml" % "copy")
        )
        dst = Path(__file__).parent.joinpath(
            "template/snap_polarimetry_graph_%s.xml" % "copy"
        )

        params: dict = {
            "Subset": self.params.clip_to_aoi,
            "Land-Sea-Mask": self.params.mask,
            "Speckle-Filter": self.params.speckle_filter,
            "Terrain-Correction": self.params.tcorrection,
            "LinearToFromdB": self.params.linear_to_db,
        }

        for key, val in params.items():
            if not val:
                LOGGER.info(f"{key} will be discarded.")
                self.revise_graph_xml(dst, key)

        file_pointer = open(dst)
        template = Template(file_pointer.read())

        return template.substitute(substitutes)

    def target_snap_graph_path(self, feature: Feature, polarisation: str) -> Path:
        """
        Returns the target path where the generated SNAP xml graph file should be stored
        """

        return Path(self.path_to_tmp_out).joinpath(
            f"{self.safe_file_name(feature)}_{polarisation}.xml"
        )

    def create_substitutions_dict(
        self, feature: Feature, polarisation: str, out_file_pol: str
    ):
        dict_default = {
            "read_file_manifest_path": self.manifest_file_location(feature),
            "downcase_polarisation": out_file_pol,
            "upcase_polarisation": polarisation.upper(),
            "sigma_band": "true",
            "gamma_band": "false",
            "beta_band": "false",
        }

        try:
            poly = self.params.geometry()
            geom = shape(poly)
            dict_default["polygon"] = geom.wkt
        except UP42Error:
            LOGGER.info("no ROI set, SNAP will process the whole scene.")

        if self.params.mask == ["sea"]:
            dict_default["mask_type"] = "false"
        if self.params.mask == ["land"]:
            dict_default["mask_type"] = "true"

        if self.params.calibration_band == ["sigma"]:
            dict_default["band_type"] = "Sigma0"
        elif self.params.calibration_band == ["gamma"]:
            dict_default["sigma_band"] = "false"
            dict_default["gamma_band"] = "true"
            dict_default["band_type"] = "Gamma0"
        elif self.params.calibration_band == ["beta"]:
            dict_default["sigma_band"] = "false"
            dict_default["beta_band"] = "true"
            dict_default["band_type"] = "Beta0"
        else:
            LOGGER.error("Wrong calibration band type.")

        return dict_default

    def generate_snap_graph(
        self, feature: Feature, polarisation: str, out_file_pol: str
    ):
        """
        Generates the snap graph xml file for the
        given feature, based on the snap graph xml template
        """
        dict_default = self.create_substitutions_dict(
            feature, polarisation, out_file_pol
        )
        result = self.process_template(dict_default)
        self.target_snap_graph_path(feature, polarisation).write_text(result)

    @staticmethod
    def replace_dem():
        """
        This methods checks if the latitude of input data is not covered by of SRTM, the default
        Digital Elevation Model (DEM), inside .xml template file. If that would be the case,
        it uses ASTER 1sec GDEM as DEM for applying terrain correction.
        """
        dst = Path(__file__).parent.joinpath("template/snap_polarimetry_graph.xml")
        tree = Et.parse(dst)
        root = tree.getroot()
        all_nodes = root.findall("node")
        for index, _ in enumerate(all_nodes):
            if all_nodes[index].attrib["id"] == "Terrain-Correction":
                all_nodes[index].find("parameters")[1].text = "ASTER 1sec GDEM"
            tree.write(dst)

    @staticmethod
    def extract_relevant_coordinate(coor):
        """
        This method checks for the maximum (minimum) latitude
        for the Northern (Southern) Hemisphere.Then this latitude
        will be used to check whether area of interest, containing this latitude,
        is covered by default Digital Elevation Model (SRTM) or not.
        """
        if coor[1] and coor[3] < 0:
            relevant_coor = min(coor[1], coor[3])
        if coor[1] and coor[3] > 0:
            relevant_coor = max(coor[1], coor[3])
        return relevant_coor

    def assert_dem(self, coor):
        """
        This method makes sure the correct DEM is been used at .xml file
        """
        r_c = self.extract_relevant_coordinate(coor)
        if not -56.0 < r_c < 60.0:
            self.replace_dem()
            LOGGER.info("SRTM is been replace by ASTER GDEM.")

    def assert_input_params(self):
        if not self.params.clip_to_aoi:
            if self.params.bbox or self.params.contains or self.params.intersects:
                raise UP42Error(
                    SupportedErrors.WRONG_INPUT_ERROR,
                    "When clip_to_aoi is set to False, bbox, contains "
                    "and intersects must be set to null.",
                )
        else:
            if (
                self.params.bbox is None
                and self.params.contains is None
                and self.params.intersects is None
            ):
                raise UP42Error(
                    SupportedErrors.WRONG_INPUT_ERROR,
                    "When clip_to_aoi set to True, you MUST define the same "
                    "coordinates in bbox, contains or intersect for both "
                    "the S1 and SNAP blocks.",
                )

    def process_snap(self, feature: Feature, requested_pols) -> list:
        """
        Wrapper method to facilitate the setup and the actual execution of the SNAP processing
        command for the given feature
        """
        out_files = []

        input_file_path = self.safe_file_path(feature)
        available_pols = self.extract_polarisations(input_file_path)

        if not self.validate_polarisations(requested_pols, available_pols):
            raise WrongPolarizationError(
                "Polarization missing; proceeding to next file"
            )

        for polarisation in requested_pols:

            # Construct output snap processing file path with SAFE id plus polarization
            # i.e. S1A_IW_GRDH_1SDV_20190928T051659_20190928T051724_029217_035192_D2A2_vv
            out_file_pol = (
                f"/tmp/input/{str(input_file_path.stem)}_{polarisation.lower()}"
            )
            self.generate_snap_graph(feature, polarisation, out_file_pol)

            cmd = GPT_CMD.format(
                gpt_path="gpt",
                graph_xml_path=self.target_snap_graph_path(feature, polarisation),
                source_file=input_file_path,
            )

            LOGGER.info(f"Running SNAP command: {cmd}")
            # Need to use os.system; subprocess does not work
            return_value = os.system(cmd)

            if return_value:
                ## Note to future self:
                ## return_value = 35072 means docker container ran out of memory!!
                ## Increase it to be higher than 8gb + 2gb swap
                LOGGER.error(
                    f"SNAP did not finish successfully with error code {return_value}"
                )
                sys.exit(return_value)

            # There are cases where in the end the clipped output image is empty
            if self.params.clip_to_aoi and is_empty(Path(out_file_pol + ".tif")):
                LOGGER.info(
                    f"Output file {out_file_pol} empty, removing it from list of returned images"
                )
            else:
                out_files.append(out_file_pol)

        return out_files

    def process(self, input_fc: FeatureCollection):
        """
        Main wrapper method to facilitate snap processing per feature
        """
        polarisations: List = self.params.polarisations or ["VV"]

        self.assert_input_params()

        results: List[Feature] = []
        out_dict: dict = {}
        for in_feature in input_fc.get("features"):
            coordinate = in_feature["bbox"]
            self.assert_dem(coordinate)
            try:
                processed_graphs = self.process_snap(in_feature, polarisations)
                LOGGER.info("SNAP processing is finished!")
                if not processed_graphs:
                    LOGGER.debug("No processed images returned, will continue")
                    continue
                out_feature = copy.deepcopy(in_feature)
                processed_tif_uuid = out_feature.properties["up42.data_path"]
                out_path = f"/tmp/output/{processed_tif_uuid}/"
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                for out_polarisation in processed_graphs:
                    # Besides the path we only need to change the capabilities
                    shutil.move(
                        ("%s.tif" % out_polarisation),
                        ("%s%s.tif" % (out_path, out_polarisation.split("_")[-1])),
                    )
                del out_feature["properties"]["up42.data_path"]
                set_data_path(out_feature, processed_tif_uuid + ".tif")
                results.append(out_feature)
                out_dict[processed_tif_uuid] = {
                    "id": processed_tif_uuid,
                    "z": [i.split("_")[-1] for i in processed_graphs],
                    "out_path": out_path,
                }
                Path(__file__).parent.joinpath(
                    "template/" "snap_polarimetry_graph_%s.xml" % "copy"
                ).unlink()
            except WrongPolarizationError:
                LOGGER.error(
                    f"WrongPolarizationError: some or all of the polarisations "
                    f"({polarisations}) don't exist in this product "
                    f"({self.safe_file_name(in_feature),}), skipping.",
                )
                continue

        if not results:
            raise UP42Error(
                SupportedErrors.NO_OUTPUT_ERROR,
                "The used input parameters don't result in any output "
                "when applied to the provided input images.",
            )

        for out_id in out_dict:  # pylint: disable=consider-using-dict-items
            my_out_path = out_dict[out_id]["out_path"]
            out_id_z = out_dict[out_id]["z"]
            if self.params.mask is not None:
                self.post_process(my_out_path, out_id_z)
            self.rename_final_stack(my_out_path, out_id_z)

        result_fc = FeatureCollection(results)

        if self.params.clip_to_aoi:
            result_fc = update_extents(result_fc)

        return result_fc

    @staticmethod
    def post_process(output_filepath, list_pol):
        """
        This method updates the novalue data to be 0 so it
        can be recognized by qgis.
        """
        for pol in list_pol:
            init_output = f"{output_filepath}{pol}.tif"
            with rasterio.open(init_output) as src:
                p_r = src.profile
                p_r.update(nodata=0)
                update_name = f"{output_filepath}updated_{pol}.tif"
                image_read = src.read()
                with rasterio.open(update_name, "w", **p_r) as dst:
                    for b_i in range(src.count):
                        dst.write(image_read[b_i, :, :], indexes=b_i + 1)

            Path(f"{output_filepath}/{pol}.tif").unlink()
            Path(update_name).rename(Path(f"{output_filepath}{pol}.tif"))

    @staticmethod
    def revise_graph_xml(xml_file, key: str):
        """
        This method checks whether, land-sea-mask or terrain-correction
        pre-processing step is needed or not. If not, it removes the
        corresponding node from the .xml file.
        """
        tree = Et.parse(xml_file)
        root = tree.getroot()
        all_nodes = root.findall("node")

        for index, _ in enumerate(all_nodes):
            if all_nodes[index].attrib["id"] == key:
                root.remove(all_nodes[index])
                params = all_nodes[index + 1].find("sources")
                params[0].attrib["refid"] = all_nodes[index - 1].attrib["id"]  # type: ignore
        tree.write(xml_file)

    @staticmethod
    def read_write_bigtiff(out_path, pol):
        """
        This method is a proper way to read big GeoTIFF raster data.
        """
        with rasterio.Env():
            with rasterio.open(f"{out_path}{pol[0]}.tif") as src0:
                kwargs = src0.profile
                kwargs.update(
                    count=len(pol),
                    bigtiff="YES",
                    compress="lzw",  # Output will be larger than 4GB
                )

                with rasterio.open(f"{out_path}stack.tif", "w", **kwargs) as dst:
                    for b_id, layer in enumerate(pol):
                        src = rasterio.open(f"{out_path}{layer}.tif")
                        windows = src.block_windows(1)
                        for _, window in windows:
                            src_data = src.read(1, window=window)
                            np.nan_to_num(src_data, copy=False)
                            dst.write(src_data, window=window, indexes=b_id + 1)
                        dst.set_band_description(b_id + 1, layer)

    def rename_final_stack(self, output_filepath, list_pol):
        """
        This method combines all the .tiff files with different polarization into one .tiff file.
        Then it renames and relocated the final output in the right directory.
        """
        LOGGER.info("Writing started.")
        self.read_write_bigtiff(output_filepath, list_pol)
        LOGGER.info("Writing is finished.")

        for pol in list_pol:
            pol_tif = f"{pol}.tif"
            Path(output_filepath).joinpath(pol_tif).unlink()

        # rename the files
        stack_tif = f"{output_filepath}stack.tif"
        Path(stack_tif).rename(
            Path("%s%s.tif" % (output_filepath, Path("%s" % output_filepath).stem))
        )

        my_file = f"{str(Path(output_filepath))}.tif"
        if os.path.exists(my_file):
            Path(my_file).unlink()

        # Move the renamed file to parent directory
        shutil.move(
            "%s%s.tif" % (output_filepath, Path("%s" % output_filepath).stem),
            "%s" % Path("%s" % output_filepath).parent,
        )

        # Remove the child directory
        try:
            shutil.rmtree(Path(output_filepath))
        # Deleting subfolder sometimes does not work in temp, then remove all subfiles.
        except (PermissionError, OSError):
            files_to_delete = Path(output_filepath).rglob("*.*")
            for file_path in files_to_delete:
                file_path.unlink()

    @classmethod
    def from_dict(cls, kwargs):
        """
        Instantiate a class with a dictionary of parameters
        """
        return cls(kwargs)
