"""
USER CONFIGURATION — Edit this file only
=========================================
This is the ONLY file you need to edit.

Workflow:
  1. Edit AOI_NAME, AOI_COORDS, YEAR_HISTORICAL, YEAR_RECENT below
  2. Run: python run_pipeline.py
     → GEE authentication prompt appears once (first run only)
     → All data downloaded automatically
     → Classification and change maps produced automatically

No Google Drive setup, no manual downloads, and no GEE scripting needed.
"""

# Define your study area
AOI_NAME = "Banke"

# Paste polygon coordinates from GEE, Google Maps, or any GIS tool.
# Format: [[lon, lat], [lon, lat], ...]
AOI_COORDS =[
    [81.5804462723946,  28.242744340900337],
    [81.5639667802071,  28.189499791620374],
    [81.49530222942585, 28.141072631100005],
    [81.48431590130085, 28.08777745763861 ],
    [81.55710032512897, 28.051424661121995],
    [81.62713816692585, 28.010209962841863],
    [81.70541575481647, 27.9883840287223  ],
    [81.73150828411335, 28.008997526928592],
    [81.71090891887897, 28.075661225512068],
    [81.7012958817696,  28.095046540669294],
    [81.7122822098946,  28.177395056237327],
    [81.76584055950397, 28.220965688839495],
    [81.68344309856647, 28.271775627550884],
    [81.64499095012897, 28.281450965813857],
]

# Choose the two years you want to compare
YEAR_HISTORICAL = 1985   # Use ≥ 1975 for GHSL, or ≥ 2015 for Dynamic World
YEAR_RECENT     = 2023   # Use ≥ 2015 for Dynamic World
DATE_HISTORICAL_START = f"{YEAR_HISTORICAL}-01-01" 
DATE_HISTORICAL_END = f"{YEAR_HISTORICAL}-12-31"

DATE_RECENT_START = f"{YEAR_RECENT}-01-01" 
DATE_RECENT_END = f"{YEAR_RECENT}-12-31"
# You're done editing here
#
# Everything below is calculated automatically:
#
#   YEAR_HISTORICAL < 2015  -> GHSL (JRC, European Commission)
#   YEAR_RECENT    >= 2015  -> Dynamic World (Google)
#
#   Landsat date windows, UTM zone, output paths,
#   classification method, and file names are all auto-generated.
