"""
Stage 0 — Fully Automated GEE Data Download
=============================================
This stage downloads every required raster directly from GEE to local disk.
No Google Drive setup and no manual export steps.

Important GHSL note:
  The GHSL built_surface band is in m² per 100m pixel.
  Exporting at 30m with averaging resampling dilutes values across
  ~11 sub-pixels, causing most to fall below the 250 m² threshold
  and severely undercounting urban area.

  What we do here instead:
    1. Apply the 250 m² threshold SERVER-SIDE in GEE at native 100m
    2. Export the resulting binary raster (0/1) at 30m using
       nearest-neighbour — resampling 0s and 1s is lossless

  So the binary GeoTIFF on disk is already classified:
    1 = urban (built_surface >= 250 m² at 100m)
    0 = non-urban
  Stage 1 reads this directly with no further thresholding.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import math
import logging
import tempfile
import requests
import zipfile
import rasterio
from rasterio.merge import merge

import config as cfg
import user_config as UC

log = logging.getLogger(__name__)

MAX_TILE_DIM = 1000
GEE_TIMEOUT  = 300


# Earth Engine setup

def init_gee():
    try:
        import ee
    except ImportError:
        log.error("earthengine-api not installed. Run: pip install earthengine-api")
        sys.exit(1)
    try:
        ee.Initialize(project='masterproject-481618')
        log.info("  ✅ GEE initialised (existing credentials)")
    except Exception:
        log.info("  🔑 No credentials — launching GEE authentication ...")
        ee.Authenticate()
        ee.Initialize()
        log.info("  ✅ GEE initialised after authentication")
    return ee


# Tile utilities

def aoi_bounds():
    lons = [c[0] for c in UC.AOI_COORDS]
    lats = [c[1] for c in UC.AOI_COORDS]
    return min(lons), min(lats), max(lons), max(lats)


def make_tiles(lon_min, lat_min, lon_max, lat_max, scale_m):
    lat_mid = (lat_min + lat_max) / 2
    m_per_deg_lon = 111320 * math.cos(math.radians(lat_mid))
    m_per_deg_lat = 111320.0
    tile_deg_lon  = (MAX_TILE_DIM * scale_m) / m_per_deg_lon
    tile_deg_lat  = (MAX_TILE_DIM * scale_m) / m_per_deg_lat
    tiles = []
    y = lat_min
    while y < lat_max:
        y2 = min(y + tile_deg_lat, lat_max)
        x = lon_min
        while x < lon_max:
            x2 = min(x + tile_deg_lon, lon_max)
            tiles.append((x, y, x2, y2))
            x = x2
        y = y2
    return tiles


# Download helpers

def download_image(ee, image, bounds, scale, dst_path, label,
                   resampling="bilinear"):
    tiles = make_tiles(*bounds, scale)
    n = len(tiles)
    if n == 1:
        log.info(f"  ↓ {label}  (single tile, {scale}m, {resampling})")
        _fetch_tile(ee, image, bounds, scale, dst_path, resampling)
    else:
        log.info(f"  ↓ {label}  ({n} tiles, {scale}m, {resampling})")
        with tempfile.TemporaryDirectory() as tmp:
            tile_paths = []
            for i, tb in enumerate(tiles):
                tp = Path(tmp) / f"tile_{i:04d}.tif"
                log.info(f"    tile {i+1}/{n} ...")
                _fetch_tile(ee, image, tb, scale, tp, resampling)
                tile_paths.append(tp)
            log.info(f"  🔗 Merging {n} tiles ...")
            _merge_tiles(tile_paths, dst_path)
    log.info(f"  ✅ {dst_path.name}  ({dst_path.stat().st_size/1e6:.1f} MB)")


def _fetch_tile(ee, image, bounds, scale, out_path, resampling="bilinear"):
    lon_min, lat_min, lon_max, lat_max = bounds
    region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    img_export = image if resampling == "near" else image.resample("bilinear")
    url = img_export.getDownloadURL({
        "region":    region,
        "scale":     scale,
        "crs":       "EPSG:4326",
        "format":    "GEO_TIFF",
        "maxPixels": 1e9,
    })
    resp = requests.get(url, timeout=GEE_TIMEOUT, stream=True)
    resp.raise_for_status()
    raw = b"".join(resp.iter_content(chunk_size=1 << 20))
    ctype = resp.headers.get("Content-Type", "")
    if "zip" in ctype or raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tif_name = next(n for n in zf.namelist() if n.endswith(".tif"))
            out_path.write_bytes(zf.read(tif_name))
    else:
        out_path.write_bytes(raw)


def _merge_tiles(tile_paths, out_path):
    datasets = [rasterio.open(p) for p in tile_paths]
    mosaic, transform = merge(datasets, nodata=float("nan"))
    # Clean up non-finite edge pixels from GEE before writing output
    import numpy as np
    mosaic = np.where(np.isfinite(mosaic), mosaic, 0.0)
    meta = datasets[0].meta.copy()
    meta.update(driver="GTiff", height=mosaic.shape[1],
                width=mosaic.shape[2], transform=transform,
                compress="lzw", nodata=255)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mosaic)
    for ds in datasets:
        ds.close()


# Image builders

def build_ghsl_binary(ee, year, aoi):
    epochs  = [1975, 1980, 1985, 1990, 2000, 2010, 2015, 2020]
    closest = min(epochs, key=lambda e: abs(e - year))
    if closest != year:
        log.warning(f"  ⚠️  No GHSL epoch for {year} — using {closest}")
    built = (ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S")
               .filter(ee.Filter.calendarRange(closest, closest, "year"))
               .first()
               .select("built_surface")
               .clip(aoi))
    binary = built.gte(cfg.GHSL_THRESHOLD).rename("urban_binary")
    log.info(f"  GHSL epoch: {closest}  threshold: >={cfg.GHSL_THRESHOLD:.0f} m²  (server-side)")
    return binary.toFloat(), closest


def build_dynamic_world(ee, date_start, date_end, aoi):
    img = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
             .filterBounds(aoi)
             .filterDate(date_start, date_end)
             .select("label")
             .mode()
             .clip(aoi))
    return img.toFloat()


def _qa_mask(ee, img):
    qa   = img.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return (img.updateMask(mask)
               .select("SR_B.")
               .multiply(0.0000275).add(-0.2)
               .copyProperties(img, img.propertyNames()))


def build_landsat5(ee, date_start, date_end, aoi):
    """
    Landsat 5 composite for visualisation only.
    Uses explicit Or filter for dry season (Oct-Dec + Jan-Apr) instead
    of calendarRange(10,4) which wraps unreliably in GEE.
    Relaxed cloud threshold (50%) for sparse historical coverage.
    Falls back to all-season if dry-season yields 0 scenes.
    """
    dry_filter = ee.Filter.Or(
        ee.Filter.calendarRange(10, 12, "month"),
        ee.Filter.calendarRange(1,   4, "month"),
    )
    base = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
              .filterBounds(aoi)
              .filterDate(date_start, date_end)
              .filter(ee.Filter.lt("CLOUD_COVER_LAND", 50))
              .map(lambda img: _qa_mask(ee, img)))
    dry  = base.filter(dry_filter)
    n    = dry.size().getInfo()
    log.info(f"    Landsat 5 dry-season scenes ")
    if n == 0:
        n_all = base.size().getInfo()
        log.info(f"    Falling back to all-season: {n_all} scenes")
        if n_all == 0:
            return None
        col = base
    else:
        col = dry
    return (col.median()
               .select(["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"],
                       ["B1","B2","B3","B4","B5","B7"])
               .clip(aoi).toFloat())


def build_landsat89(ee, date_start, date_end, aoi):
    """
    Landsat 8/9 composite for visualisation only.
    Same dry-season fix as build_landsat5.
    """
    dry_filter = ee.Filter.Or(
        ee.Filter.calendarRange(10, 12, "month"),
        ee.Filter.calendarRange(1,   4, "month"),
    )
    def _col(name):
        return (ee.ImageCollection(name)
                  .filterBounds(aoi)
                  .filterDate(date_start, date_end)
                  .filter(dry_filter)
                  .filter(ee.Filter.lt("CLOUD_COVER_LAND", 20))
                  .map(lambda img: _qa_mask(ee, img))
                  .select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"],
                          ["B2","B3","B4","B5","B6","B7"]))
    merged = _col("LANDSAT/LC08/C02/T1_L2").merge(
             _col("LANDSAT/LC09/C02/T1_L2"))
    n = merged.size().getInfo()
    log.info(f"    Landsat 8/9 scenes: {n}")
    if n == 0:
        return None
    return merged.median().clip(aoi).toFloat()


# Main pipeline entry

def run():
    log.info("══════════════════════════════════════════════")
    log.info("  STAGE 0 — AUTOMATED GEE DATA DOWNLOAD")
    log.info(f"  AOI:        {cfg.AOI_NAME}")
    log.info(f"  Historical: {cfg.YEAR_HISTORICAL} → {cfg.METHOD_HISTORICAL}")
    log.info(f"  Recent:     {cfg.YEAR_RECENT} → {cfg.METHOD_RECENT}")
    log.info("  Mode:       direct download from GEE")
    log.info("══════════════════════════════════════════════")

    ee     = init_gee()
    aoi    = ee.Geometry.Polygon([UC.AOI_COORDS])
    bounds = aoi_bounds()
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Process the historical epoch
    log.info(f"\n── Historical: {cfg.YEAR_HISTORICAL} ({cfg.METHOD_HISTORICAL}) ──")

    if cfg.METHOD_HISTORICAL == "GHSL":
        img, epoch = build_ghsl_binary(ee, cfg.YEAR_HISTORICAL, aoi)
        dst = cfg.GHSL_HISTORICAL_PATH
        if not dst.exists():
            download_image(ee, img, bounds, cfg.TARGET_RES, dst,
                           f"GHSL {epoch} binary", resampling="near")
        else:
            log.info(f"  ⏭  {dst.name} already exists — skipping")

    elif cfg.METHOD_HISTORICAL == "DynamicWorld":
        img = build_dynamic_world(ee, cfg.DATE_HISTORICAL_START,
                                      cfg.DATE_HISTORICAL_END, aoi)
        dst = cfg.DW_HISTORICAL_PATH
        if not dst.exists():
            download_image(ee, img, bounds, cfg.TARGET_RES, dst,
                           f"Dynamic World {cfg.YEAR_RECENT}", resampling="near")
        else:
            log.info(f"  ⏭  {dst.name} already exists — skipping")

    l5 = build_landsat5(ee, cfg.DATE_HISTORICAL_START, cfg.DATE_HISTORICAL_END, aoi)
    dst_l5 = cfg.LANDSAT_HISTORICAL_PATH
    if l5 is None:
        log.warning("  ")
    elif not dst_l5.exists():
        download_image(ee, l5, bounds, cfg.TARGET_RES, dst_l5,
                       "Landsat 5 composite", resampling="bilinear")
    else:
        log.info(f"  ⏭  {dst_l5.name} already exists — skipping")

    # Process the recent epoch
    log.info(f"\n── Recent: {cfg.YEAR_RECENT} ({cfg.METHOD_RECENT}) ──")

    if cfg.METHOD_RECENT == "DynamicWorld":
        img = build_dynamic_world(ee, cfg.DATE_RECENT_START, 
                                 cfg.DATE_RECENT_END, aoi)
        dst = cfg.DW_RECENT_PATH
        if not dst.exists():
            download_image(ee, img, bounds, 10, dst, 
                           f"Dynamic World {cfg.YEAR_RECENT}", resampling="near")
        else:
            log.info(f"  ⏭  {dst.name} already exists — skipping")
    elif cfg.METHOD_RECENT == "GHSL":
        img, epoch = build_ghsl_binary(ee, cfg.YEAR_RECENT, aoi)
        dst = cfg.GHSL_RECENT_PATH
        if not dst.exists():
            download_image(ee, img, bounds, cfg.TARGET_RES, dst,
                           f"GHSL {epoch} binary", resampling="near")
        else:
            log.info(f"  ⏭  {dst.name} already exists — skipping")

    l89 = build_landsat89(ee, cfg.DATE_RECENT_START, cfg.DATE_RECENT_END, aoi)
    dst_l89 = cfg.LANDSAT_RECENT_PATH
    if l89 is None:
        log.warning(" ")
    elif not dst_l89.exists():
        download_image(ee, l89, bounds, cfg.TARGET_RES, dst_l89,
                       "Landsat 8/9 composite", resampling="bilinear")
    else:
        log.info(f"  ⏭  {dst_l89.name} already exists — skipping")

    # Print a quick summary of downloaded rasters
    log.info("\n── Downloaded files ──────────────────────────")
    for f in sorted(cfg.RAW_DIR.glob("*.tif")):
        log.info(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
    log.info("\n Stage 0 complete")
    log.info("══════════════════════════════════════════════")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
