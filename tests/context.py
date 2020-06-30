"""
This module is used in test_snap_polarimetry script.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../libs"))
)

# pylint: disable=unused-import,wrong-import-position
from snap_polarimetry import SNAPPolarimetry, is_empty, update_extents
