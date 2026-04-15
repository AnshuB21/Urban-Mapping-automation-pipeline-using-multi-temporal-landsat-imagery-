"""
Auto-computed configuration — do not edit.
Edit user_config.py instead.
"""

from pathlib import Path
import numpy as np

from user_config import (
    AOI_NAME, AOI_COORDS,
    YEAR_HISTORICAL, YEAR_RECENT,
)

# Automatically detect the best UTM zone for the AOI
def _utm_epsg(coords):
    cx   = np.mean([c[0] for c in coords])
    cy   = np.mean([c[1] for c in coords])
    zone = int((cx + 180) / 6) + 1
    return f"EPSG:326{zone:02d}" if cy >= 0 else f"EPSG:327{zone:02d}"

TARGET_CRS = _utm_epsg(AOI_COORDS)
TARGET_RES = 30  # meters

# Choose classification source based on year
# Dynamic World is available from 2015 (Sentinel-2 era)
# GHSL is available from 1975 onward (Landsat era)
DW_START_YEAR = 2015

METHOD_HISTORICAL = "GHSL"         if YEAR_HISTORICAL < DW_START_YEAR \
                    else "DynamicWorld"
METHOD_RECENT     = "DynamicWorld" if YEAR_RECENT >= DW_START_YEAR \
                    else "GHSL"

# Build Landsat compositing windows automatically
# We use a 4-year window around the target year, limited to
# the dry season (Oct-Apr) to reduce clouds in South Asia.
# Older years can need wider windows because scenes are sparse.

def _date_window(year):
    """Return (start, end) date strings for a ~4-year composite window."""
    half = 2
    return f"{year - half}-01-01", f"{year + half}-12-31"

# Prefer dates from user_config when provided; otherwise compute them
try:
    from user_config import DATE_HISTORICAL_START, DATE_HISTORICAL_END
    from user_config import DATE_RECENT_START, DATE_RECENT_END
except ImportError:
    DATE_HISTORICAL_START, DATE_HISTORICAL_END = _date_window(YEAR_HISTORICAL)
    DATE_RECENT_START,     DATE_RECENT_END     = _date_window(YEAR_RECENT)

# Output paths
COLAB_ROOT   = Path("/content")
PIPELINE_DIR = COLAB_ROOT / "pipeline_outputs" / AOI_NAME
RAW_DIR      = PIPELINE_DIR / "raw"
MAPS_DIR     = PIPELINE_DIR / "maps"
OUTPUTS_DIR  = PIPELINE_DIR / "outputs"

for _d in [RAW_DIR, MAPS_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Input raster file paths
GHSL_HISTORICAL_PATH    = RAW_DIR / f"{AOI_NAME}_GHSL_{YEAR_HISTORICAL}.tif"
GHSL_RECENT_PATH        = RAW_DIR / f"{AOI_NAME}_GHSL_{YEAR_RECENT}.tif"
DW_HISTORICAL_PATH      = RAW_DIR / f"{AOI_NAME}_DW_{YEAR_HISTORICAL}.tif"
DW_RECENT_PATH          = RAW_DIR / f"{AOI_NAME}_DW_{YEAR_RECENT}.tif"
LANDSAT_HISTORICAL_PATH = RAW_DIR / f"{AOI_NAME}_landsat_{YEAR_HISTORICAL}.tif"
LANDSAT_RECENT_PATH     = RAW_DIR / f"{AOI_NAME}_landsat_{YEAR_RECENT}.tif"

# GHSL settings
GHSL_THRESHOLD = 250.0   # m² threshold for built-up surface

# Final land cover classes
CLASSES = {
    1: ("Urban / Built-up",  "#cc0000"),
    2: ("Trees / Forest",    "#1a6600"),
    3: ("Grass / Shrub",     "#78c679"),
    4: ("Cropland",          "#ffffb2"),
    5: ("Water",             "#1f78b4"),
    6: ("Flooded / Wetland", "#80cdc1"),
    7: ("Bare / Sand",       "#d2b48c"),
}

# Dynamic World to project-class mapping
# DW: 0=water 1=trees 2=grass 3=flooded 4=crops
#     5=shrub 6=built 7=bare 8=snow
DW_CLASS_MAP = {
    0: 5,   # water       -> Water
    1: 2,   # trees       -> Trees / Forest
    2: 3,   # grass       -> Grass / Shrub
    3: 6,   # flooded veg -> Flooded / Wetland
    4: 4,   # crops       -> Cropland
    5: 3,   # shrub       -> Grass / Shrub (intentionally merged)
    6: 1,   # built       -> Urban (main focus class)
    7: 7,   # bare        -> Bare / Sand
    8: 7,   # snow/ice    -> Bare / Sand
}

DPI = 150


def print_config():
    print(f"\nPipeline Configuration")
    print(f"  AOI:        {AOI_NAME}")
    print(f"  UTM zone:   {TARGET_CRS}  (auto-detected)")
    print(f"  Historical: {YEAR_HISTORICAL}  → {METHOD_HISTORICAL}")
    print(f"              composite window {DATE_HISTORICAL_START} – {DATE_HISTORICAL_END}")
    print(f"  Recent:     {YEAR_RECENT}   → {METHOD_RECENT}")
    print(f"              composite window {DATE_RECENT_START} – {DATE_RECENT_END}")
    print(f"  Output:     {PIPELINE_DIR}")
    print()
