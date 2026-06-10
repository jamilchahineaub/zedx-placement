# conftest.py (repo root)
#
# Makes the pure-python modules under isaac/ and the repo root importable from
# pytest without installing the package. analysis/ tests import e.g.
#   import camera_rig
#   import geo_prescreener
# These modules have no omni/pyzed dependencies, so they import fine under
# plain python3. (Isaac-runtime modules like scene_builder are NOT imported by
# the test suite.)

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

for _p in (_ROOT, os.path.join(_ROOT, "isaac"), os.path.join(_ROOT, "analysis")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
