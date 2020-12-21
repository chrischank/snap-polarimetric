"""
End-to-end test: Fetches data, creates output, stores it in /tmp and checks if output
is valid.
"""
from pathlib import Path

import numpy as np
import geojson

from blockutils.e2e import E2ETest

# Disable unused params for assert
# pylint: disable=unused-argument
def asserts(input_dir: Path, output_dir: Path, quicklook_dir: Path, logger):
    # Print out bbox of one tile
    geojson_path = output_dir / "data.json"

    with open(str(geojson_path)) as f:
        feature_collection = geojson.load(f)

    result_bbox = feature_collection.features[0].bbox
    logger.info(result_bbox)

    # BBox might seem slightly off from the extent requested, but this is to be expected if no
    # terrain correction is applied
    assert np.allclose(
        np.array([14.589980, 53.414966, 14.626898, 53.433054]),
        np.array(result_bbox),
        atol=1e-04,
    )

    output_snap = (
        output_dir / feature_collection.features[0].properties["up42.data_path"]
    )

    logger.info(output_snap)

    assert output_snap.exists()


if __name__ == "__main__":
    e2e = E2ETest("snap-polarimetric")
    e2e.add_parameters(
        {
            "bbox": [14.558086, 53.413829, 14.584178, 53.433673],
            "mask": None,
            "tcorrection": False,
            "polarisations": ["VV"],
            "clip_to_aoi": True,
        }
    )
    e2e.add_gs_bucket("gs://floss-blocks-e2e-testing/e2e_snap_polarimetric/*")
    e2e.asserts = asserts
    e2e.run()
