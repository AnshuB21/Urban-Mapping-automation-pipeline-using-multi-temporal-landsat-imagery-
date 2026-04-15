# Urban Mapping Automation Pipeline (Multi-Temporal Landsat Imagery)

This project automates urban land-cover change detection between two years for a custom AOI.

# mount your google drive

# link your project in stage one, where gee.initialize (project="Your project id from GEE")

It runs in 3 stages:

- `Stage 0`: Download required rasters from Google Earth Engine (GEE)
- `Stage 1`: Classify land cover for historical and recent years
- `Stage 2`: Detect change, compute statistics, and generate maps/charts

## Project Structure

```text
final_clean/
  run_pipeline.py
  user_config.py
  config.py
  scripts/
    00_gee_export.py
    01_classify.py
    02_change_detection.py
```

## Requirements

- Python 3.10+
- GEE account with access enabled
- Python packages:
  - `earthengine-api`
  - `numpy`
  - `rasterio`
  - `matplotlib`
  - `scipy`
  - `requests`

Install dependencies:

```bash
pip install earthengine-api numpy rasterio matplotlib scipy requests
```

## Quick Start

1. Open `user_config.py` and edit:

- `AOI_NAME`
- `AOI_COORDS`
- `YEAR_HISTORICAL`
- `YEAR_RECENT`

2. Run full pipeline:

```bash
python run_pipeline.py
```

On first run, GEE authentication will prompt once.

## Useful Commands

Run full pipeline:

```bash
python run_pipeline.py
```

Start from a stage:

```bash
python run_pipeline.py --from-stage 1
```

Run only one stage:

```bash
python run_pipeline.py --stages 2
```

Show current config:

```bash
python run_pipeline.py --info
```

## Outputs

The pipeline writes outputs under:

```text
/content/pipeline_outputs/<AOI_NAME>/
```

Main output folders:

- `raw/` downloaded rasters
- `outputs/` classified rasters and CSV statistics
- `maps/` PNG maps/charts

Key files:

- `outputs/landcover_<year>.tif`
- `outputs/change_map.tif`
- `outputs/change_statistics.csv`
- `maps/urban_expansion_map.png`
- `maps/growth_direction_rose.png`
- `maps/urban_heatmaps.png`
- `maps/class_transitions.png`

## Notes

- Method is auto-selected by year:
  - `< 2015`: GHSL
  - `>= 2015`: Dynamic World
- Landsat composites are generated automatically.
- No manual GEE export scripts are needed.
