"""
This module include multiple test cases to check the performance of the snap_polarimetry script.
"""
import os
import sys

from unittest.mock import patch
import shutil
from pathlib import Path, PosixPath
from xml.etree import ElementTree as ET

import attr
import geojson
import rasterio as rio
import numpy as np
import pytest

# pylint: disable=wrong-import-position
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from context import (
    SNAPPolarimetry,
    ensure_data_directories_exist,
    read_write_bigtiff,
)

TEST_POLARISATIONS = [
    (["VV"], ["VV"], True),
    (["HH"], ["HH"], True),
    (["VV"], ["HH"], False),
    (["VV", "VH"], ["VV", "VH"], True),
    (["VV", "VH"], ["HH", "HV"], False),
    (["VV", "VH"], ["VV"], False),
    (["HH"], ["HH", "HV"], True),
]


def make_dummy_raster_file(path):
    """
    Makes a dummy raster file in a given path.
    """
    with rio.open(
        path, "w", driver="GTiff", width=5, height=5, count=1, dtype="int16"
    ) as dst:
        dst.write(np.ones((1, 5, 5), dtype="int16"))
    return path


@pytest.fixture(scope="session", autouse=True)
# pylint: disable=redefined-outer-name
def fixture_mainclass():
    """
    This method initiates the SNAPPolarimetry( class from snap_polarimetry to be
    used to testing.
    """
    params = {
        "mask": ["sea"],
        "tcorrection": "false",
    }
    return SNAPPolarimetry(params)


@pytest.mark.parametrize("requested,available,expected", TEST_POLARISATIONS)
# pylint: disable=redefined-outer-name
def test_validate_polarisations(fixture_mainclass, requested, available, expected):
    """
    This methods checks whether the available polarization of the input
    data is equal to the expected polarizations.
    """
    assert fixture_mainclass.validate_polarisations(requested, available) == expected


# pylint: disable=redefined-outer-name
# pylint: disable=too-few-public-methods
@attr.s
class DummySafeFile:
    """
    This class initiate a dummy .SAFE file.
    """

    # pylint: disable-msg=R0913, R0902
    location = attr.ib()
    file_path = attr.ib()
    manifest_path = attr.ib()
    measurement_path = attr.ib()
    vh_file = attr.ib()
    vv_file = attr.ib()
    feature_collection = attr.ib()
    feature = attr.ib()


@pytest.fixture()
def safe_file():
    """
    This method creats a dummy .SAFE file and also dummy output
    after applying pre-processing steps with snap.
    :return:
    """
    # pylint: disable=too-many-locals
    ensure_data_directories_exist()

    # Set up the whole dummy input
    input_path = Path("/tmp/input")
    safe_path = input_path / "0a99c5a1-75c0-4a0d-a7dc-c2a551936be4"
    if safe_path.exists():
        shutil.rmtree(str(safe_path))
    safe_path.mkdir()

    safe_file_path = (
        safe_path / "S1B_IW_GRDH_1SDV_"
        "20190220T050359_20190220T050424_015025_01C12F_4EA4.SAFE"
    )
    safe_file_path.mkdir()

    _location_ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(_location_, "mock_data/data.json"), "rb") as f_p:
        test_featurecollection = geojson.load(f_p)
    test_feature = test_featurecollection.features[0]

    manifest_path = safe_file_path / "manifest.safe"
    manifest_path.write_text("")

    measurement_file_path = safe_file_path / "measurement"
    measurement_file_path.mkdir()

    vh_file = (
        measurement_file_path / "s1b-iw-grd-vh-"
        "20190220t050359-20190220t050424-015025-01c12f-002.tiff"
    )
    vv_file = (
        measurement_file_path / "s1b-iw-grd-vv-"
        "20190220t050359-20190220t050424-015025-01c12f-001.tiff"
    )

    make_dummy_raster_file(vh_file)
    make_dummy_raster_file(vv_file)

    test_safe_file = DummySafeFile(
        safe_path,
        safe_file_path,
        manifest_path,
        measurement_file_path,
        vh_file,
        vv_file,
        test_featurecollection,
        test_feature,
    )

    output_file_vv_before_move = Path(
        "/tmp/input/%s_%s.tif" % (safe_file_path.stem, "vv")
    )
    make_dummy_raster_file(output_file_vv_before_move)

    output_file_vh_before_move = Path(
        "/tmp/input/%s_%s.tif" % (safe_file_path.stem, "vh")
    )
    make_dummy_raster_file(output_file_vh_before_move)

    out_path = Path("/tmp/output/0a99c5a1-75c0-4a0d-a7dc-c2a551936be4")
    if out_path.exists():
        shutil.rmtree(str(out_path))
    out_path.mkdir()
    output_file_vv = out_path / "vv.tif"
    make_dummy_raster_file(output_file_vv)

    output_file_vh = out_path / "vh.tif"
    make_dummy_raster_file(output_file_vh)

    return test_safe_file


@pytest.fixture()
def safe_files():
    """
    This method creats two dummy .SAFE files and also dummy outputs
    after applying pre-processing steps with snap.
    :return:
    """
    # pylint: disable=too-many-locals
    ensure_data_directories_exist()

    _location_ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(_location_, "mock_data/two_data.json"), "rb") as f_p:
        test_featurecollection = geojson.load(f_p)

    # Set up the whole dummy input
    input_path = Path("/tmp/input")

    for feature in test_featurecollection.features:
        uid = feature.id
        s1_id = feature.properties["identification"]["externalId"] + ".SAFE"

        safe_path = input_path / uid
        if safe_path.exists():
            shutil.rmtree(str(safe_path))
        safe_path.mkdir()

        safe_file_path = safe_path / s1_id
        safe_file_path.mkdir()

        manifest_path = safe_file_path / "manifest.safe"
        manifest_path.write_text("")

        measurement_file_path = safe_file_path / "measurement"
        measurement_file_path.mkdir()

        vh_file = measurement_file_path / Path(
            "s1b-iw-grd-vh-" "%s-002.tiff" % s1_id.lower().replace("_", "-")[17:]
        )
        vv_file = measurement_file_path / Path(
            "s1b-iw-grd-vv-" "%s-001.tiff" % s1_id.lower().replace("_", "-")[17:]
        )

        make_dummy_raster_file(vh_file)
        make_dummy_raster_file(vv_file)

        test_fc = DummySafeFile(
            safe_path,
            safe_file_path,
            manifest_path,
            measurement_file_path,
            vh_file,
            vv_file,
            test_featurecollection,
            feature,
        )

        output_file_vv_before_move = Path(
            "/tmp/input/%s_%s.tif" % (safe_file_path.stem, "vv")
        )
        make_dummy_raster_file(output_file_vv_before_move)

        output_file_vh_before_move = Path(
            "/tmp/input/%s_%s.tif" % (safe_file_path.stem, "vh")
        )
        make_dummy_raster_file(output_file_vh_before_move)

        out_path = Path("/tmp/output/%s" % uid)
        if out_path.exists():
            shutil.rmtree(str(out_path))
        out_path.mkdir()
        output_file_vv = out_path / "vv.tif"
        make_dummy_raster_file(output_file_vv)

        output_file_vh = out_path / "vh.tif"
        make_dummy_raster_file(output_file_vh)

    return test_fc


# pylint: disable=redefined-outer-name
def test_extract_polarisations(fixture_mainclass, safe_file):
    """
    This methods checks whether extract_polarisation method
    returns the expected polarisations.
    """
    assert set(fixture_mainclass.extract_polarisations(safe_file.file_path)) == set(
        ["VH", "VV"]
    )


# pylint: disable=redefined-outer-name
def test_safe_file_name(fixture_mainclass, safe_file):
    """
    This methods checks whether safe_file_name methods creates the
    expected .SAFE file name.
    """

    expected_safe_file_name = (
        "S1B_IW_GRDH_1SDV_20190220T050359_20190220T050424_015025_01C12F_4EA4.SAFE"
    )

    assert (
        fixture_mainclass.safe_file_name(safe_file.feature) == expected_safe_file_name
    )


# pylint: disable=redefined-outer-name
def test_manifest_file_location(fixture_mainclass, safe_file):
    """
    This method checks whether the manifest file is located inside
    the expected directory.
    """
    assert (
        fixture_mainclass.manifest_file_location(safe_file.feature)
        == safe_file.manifest_path
    )


def test_create_substitutions_dict(safe_file):
    params = {
        "intersects": {
            "type": "Polygon",
            "coordinates": [
                [
                    [13.365898, 52.491561],
                    [13.385296, 52.491561],
                    [13.385296, 52.506191],
                    [13.365898, 52.506191],
                    [13.365898, 52.491561],
                ]
            ],
        },
        "mask": ["sea"],
        "tcorrection": "false",
    }

    test_feature = safe_file.feature
    dict_default = SNAPPolarimetry(params).create_substitutions_dict(
        test_feature, "VV", "vv"
    )
    assert (
        dict_default["polygon"] == "POLYGON ((13.365898 52.491561, 13.385296 52.491561,"
        " 13.385296 52.506191, 13.365898 52.506191, 13.365898 52.491561))"
    )


def test_create_substitutions_dict_no_subseting(safe_file):
    params = {
        "mask": ["sea"],
        "tcorrection": "false",
    }

    test_feature = safe_file.feature
    dict_default = SNAPPolarimetry(params).create_substitutions_dict(
        test_feature, "VV", "vv"
    )
    assert "polygon" not in dict_default
    assert dict_default["sigma_band"]
    assert dict_default["gamma_band"] == "false"


def test_create_substitutions_dict_add_beta_band(safe_file):
    params = {"mask": ["sea"], "tcorrection": "false", "calibration_band": "beta"}

    test_feature = safe_file.feature
    dict_default = SNAPPolarimetry(params).create_substitutions_dict(
        test_feature, "VV", "vv"
    )
    assert dict_default["beta_band"]


# pylint: disable=redefined-outer-name
def test_generate_snap_graph(fixture_mainclass, safe_file):
    """
    This method apply generate_snap_graph method to create a .xml file
    for snap graph template and checks whether this file reads the
    manifest.safe file from the expected directory.
    """
    fixture_mainclass.generate_snap_graph(
        safe_file.feature,
        "VV",
        "/tmp/input/S1B_IW_GRDH_1SDV_"
        "20190220T050359_20190220T050424_015025_01C12F_4EA4.SAFE_vv",
    )

    graph_xml_file = PosixPath(
        "/tmp/S1B_IW_GRDH_1SDV_"
        "20190220T050359_20190220T050424_015025_01C12F_4EA4.SAFE_VV.xml"
    )
    tree = ET.parse(str(graph_xml_file))
    all_nodes = tree.findall("node")

    for graph_node in all_nodes:
        if graph_node.attrib["id"] == "Read":

            params = graph_node.find("parameters")
            subnodes = list(params)
            path_to_manifest = subnodes[0].text

    expected_substring = (
        "0a99c5a1-75c0-4a0d-a7dc-c2a551936be4/"
        + "S1B_IW_GRDH_1SDV_20190220T050359_20190220T050424_015025_01C12F_4EA4.SAFE/manifest.safe"
    )

    assert path_to_manifest.endswith(expected_substring)


def test_generate_snap_graph_no_speckle_filter(safe_file):
    params = {"mask": ["sea"], "tcorrection": False, "speckle_filter": False}

    SNAPPolarimetry(params).generate_snap_graph(
        safe_file.feature,
        "VV",
        "/tmp/input/S1B_IW_GRDH_1SDV_"
        "20190220T050359_20190220T050424_015025_01C12F_4EA4.SAFE_vv",
    )
    graph_xml_file = PosixPath(
        "/tmp/S1B_IW_GRDH_1SDV_"
        "20190220T050359_20190220T050424_015025_01C12F_4EA4.SAFE_VV.xml"
    )
    tree = ET.parse(str(graph_xml_file))
    all_nodes = tree.findall("node")

    node_id_list = [graph_node.attrib["id"] for graph_node in all_nodes]

    assert "Speckle-Filter" not in node_id_list
    assert "LinearToFromdB" in node_id_list


def test_extract_relevant_coordinate(fixture_mainclass):
    """
    This method checks whether the correct latitude will be chosen. It is then used for selecting
    relevant Digital Elevation Model inside the .xml file.
    """
    bbox_1 = [-110.568535, 67.500465, -96.790337, 72.47541]
    bbox_2 = [9.94, -55.13, 9.97, -55.15]

    assert fixture_mainclass.extract_relevant_coordinate(bbox_1) == 72.47541
    assert fixture_mainclass.extract_relevant_coordinate(bbox_2) == -55.15


def test_assert_input_params():
    params = {"mask": ["sea"], "tcorrection": "false", "clip_to_aoi": "true"}

    # assert "polygon" not in dict_default
    with pytest.raises(ValueError) as e:
        SNAPPolarimetry(params).assert_input_params()
    assert (
        str(e.value)
        == "When clip_to_aoi set to True, you MUST define the same coordinates in bbox, contains"
        " or intersect for both the S1 and SNAP blocks."
    )


def test_assert_input_params_full():
    params = {
        "mask": ["sea"],
        "tcorrection": "false",
        "bbox": [
            14.558086395263674,
            53.4138293218823,
            14.584178924560549,
            53.433673900512616,
        ],
        "contains": None,
        "intersects": None,
    }

    # assert "polygon" not in dict_default
    with pytest.raises(ValueError) as e:
        SNAPPolarimetry(params).assert_input_params()
    assert (
        str(e.value)
        == "When clip_to_aoi is set to False, bbox, contains and intersects must be set to null."
    )


@patch("os.system", lambda x: 0)
# pylint: disable=redefined-outer-name
def test_process_snap(fixture_mainclass, safe_file):
    """
    This method tests the functionality of process_snap method. And checks
    whether the expected polarization is created.
    """
    test_feature = safe_file.feature

    output_file = fixture_mainclass.process_snap(test_feature, ["VV"])
    assert output_file == [
        "/tmp/input/"
        "S1B_IW_GRDH_1SDV_20190220T050359_20190220T050424_015025_01C12F_4EA4_vv"
    ]


@patch("os.system", lambda x: 0)
# pylint: disable=redefined-outer-name
def test_process_snap_multiple_polarisations(fixture_mainclass, safe_file):
    """
    This method tests the functionality of process_snap method for
    multiple polarizations.
    """
    test_feature = safe_file.feature

    output_file = fixture_mainclass.process_snap(test_feature, ["VV", "VH"])
    assert output_file == [
        "/tmp/input/"
        "S1B_IW_GRDH_1SDV_20190220T050359_20190220T050424_015025_01C12F_4EA4_vv",
        "/tmp/input/"
        "S1B_IW_GRDH_1SDV_20190220T050359_20190220T050424_015025_01C12F_4EA4_vh",
    ]


# pylint: disable=unused-variable
@patch("os.system", lambda x: 0)
def test_process_multiple_polarisations(fixture_mainclass, safe_file):
    """
    This method test the functionality of precess method. It checks
    whether the expected bbox and properties is included in the
    output.
    """
    test_fc = safe_file.feature_collection

    params = {"polarisations": ["VV", "VH"]}

    output_fc, out_dict = fixture_mainclass.process(test_fc, params)

    expected_bbox = [
        13.319549560546875,
        38.20473446610163,
        13.3209228515625,
        38.205813598134746,
    ]
    assert len(output_fc.features) == 1
    assert output_fc.features[0]["bbox"] == expected_bbox
    assert output_fc.features[0]["properties"]["up42.data_path"] != ""
    assert not Path(
        "/tmp/output/" + output_fc.features[0]["properties"]["up42.data_path"]
    ).is_file()


@patch("os.system", lambda x: 0)
def test_process_multiple_images_polarisations(fixture_mainclass, safe_files):
    """
    This method test the functionality of precess method. It checks
    whether the expected bbox and properties is included in the
    output.
    """
    test_fc = safe_files.feature_collection

    params = {"polarisations": ["VV", "VH"]}

    output_fc, out_dict = fixture_mainclass.process(test_fc, params)

    expected_bbox = [138.196686, 34.809418, 141.303055, 36.713043]
    assert len(output_fc.features) == 2
    assert output_fc.features[0]["bbox"] == expected_bbox
    assert output_fc.features[0]["properties"]["up42.data_path"] != ""
    assert not Path(
        "/tmp/output/" + output_fc.features[0]["properties"]["up42.data_path"]
    ).is_file()


@patch("os.system", lambda x: 0)
def test_run_multiple_scenes(safe_files):
    """
    This method test the functionality of the run method with multiple scenes.
    """

    # Copy two_data.json to tmp/input/data.json
    _location_ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    shutil.copyfile(
        os.path.join(_location_, "mock_data/two_data.json"),
        Path("/tmp/input/data.json"),
    )

    _ = safe_files

    os.environ["UP42_TASK_PARAMETERS"] = '{"mask": null, "tcorrection": false}'
    # params = load_params()
    SNAPPolarimetry.run()

    with open(Path("/tmp/output/data.json"), "rb") as f_p:
        test_featurecollection = geojson.load(f_p)

    assert Path(
        "/tmp/output/"
        + test_featurecollection.features[0]["properties"]["up42.data_path"]
    ).is_file()
    assert Path(
        "/tmp/output/"
        + test_featurecollection.features[1]["properties"]["up42.data_path"]
    ).is_file()

    # Clean up if exists
    if os.path.exists("/tmp/output/"):
        shutil.rmtree("/tmp/output/")
    if os.path.exists("/tmp/input/data.json"):
        os.remove("/tmp/input/data.json")


@patch("os.system", lambda x: 0)
def test_run_scene(safe_file):
    """
    This method test the functionality of the run method with one scene.
    """

    # Copy two_data.json to tmp/input/data.json
    _location_ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    shutil.copyfile(
        os.path.join(_location_, "mock_data/data.json"), Path("/tmp/input/data.json")
    )

    _ = safe_file

    os.environ["UP42_TASK_PARAMETERS"] = '{"mask": null, "tcorrection": false}'
    # params = load_params()
    SNAPPolarimetry.run()

    with open(Path("/tmp/output/data.json"), "rb") as f_p:
        test_featurecollection = geojson.load(f_p)

    assert Path(
        "/tmp/output/"
        + test_featurecollection.features[0]["properties"]["up42.data_path"]
    ).is_file()

    # Clean up if exists
    if os.path.exists("/tmp/output/"):
        shutil.rmtree("/tmp/output/")
    if os.path.exists("/tmp/input/data.json"):
        os.remove("/tmp/input/data.json")


def test_read_write_bigtiff():
    pol = ["vv", "vh"]
    output_file_vv = Path("/tmp/input/vv.tif")
    output_file_vh = Path("/tmp/input/vh.tif")
    make_dummy_raster_file(output_file_vv)
    make_dummy_raster_file(output_file_vh)

    read_write_bigtiff("/tmp/input/", pol)

    r = rio.open("/tmp/input/stack.tif")

    assert r.count == 2
