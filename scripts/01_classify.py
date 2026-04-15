"""
Stage 1 — Automatic Land Cover Classification
===============================================
This stage picks the best classification source
based on the years set in user_config.py.

Historical epoch (before 2015):
  Method: GHSL (JRC Global Human Settlement Layer)
  Product: GHS_BUILT_S — built-up surface area per pixel (m²)
  Resolution: 100m → resampled to 30m
  Coverage: 1975, 1980, 1985, 1990, 2000, 2010, 2015, 2020
  Reference: Pesaresi et al. (2023) JRC Technical Report

Recent epoch (2015 onwards):
  Method: Dynamic World (Google)
  Product: 9-class land cover from Sentinel-2
  Resolution: 10m → resampled to 30m
  Coverage: 2015 onwards, near real-time
  Reference: Brown et al. (2022) Nature Scientific Data

Why this pairing works well:
  Both are globally peer-reviewed published products
  No training data needed
  No threshold tuning needed
  Works for any AOI worldwide
  This is standard practice in urban change detection studies

7 output classes (consistent between both methods):
  1  Urban / Built-up
  2  Trees / Forest
  3  Grass / Shrub
  4  Cropland
  5  Water
  6  Flooded / Wetland
  7  Bare / Sand
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
import logging
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import config as cfg

log = logging.getLogger(__name__)


# Raster helpers
def load_raster(path):
    """Load raster, reproject to TARGET_CRS if needed."""
    with rasterio.open(path) as src:
        src_epsg = src.crs.to_epsg() if src.crs else None
        tgt_epsg = int(cfg.TARGET_CRS.split(":")[1])

        if src_epsg != tgt_epsg:
            transform, W, H = calculate_default_transform(
                src.crs, cfg.TARGET_CRS,
                src.width, src.height,
                *src.bounds, resolution=cfg.TARGET_RES)
            data = np.zeros((src.count, H, W), dtype=np.float32)
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=data[b-1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=cfg.TARGET_CRS,
                    resampling=Resampling.average)
            profile = src.profile.copy()
            profile.update(
                crs=cfg.TARGET_CRS, transform=transform,
                width=W, height=H, compress="lzw")
        else:
            data      = src.read().astype(np.float32)
            transform = src.transform
            profile   = src.profile.copy()
            H, W      = src.height, src.width

    valid = np.all(np.isfinite(data), axis=0)
    log.info(f"  {Path(path).name}: "
             f"{data.shape[2] if data.ndim==3 else W}×"
             f"{data.shape[1] if data.ndim==3 else H}  "
             f"valid={valid.sum():,} pixels")
    return data, valid, profile


def align_to_ref(src_data, src_profile, ref_profile):
    """Align src raster to ref grid using reprojection."""
    H = ref_profile["height"]
    W = ref_profile["width"]
    dst_t = ref_profile["transform"]
    dst_c = ref_profile["crs"]

    aligned = np.zeros((src_data.shape[0], H, W), dtype=np.float32)
    for b in range(src_data.shape[0]):
        reproject(
            source=src_data[b],
            destination=aligned[b],
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=dst_t,
            dst_crs=dst_c,
            resampling=Resampling.nearest)
    return aligned


# Classification methods
def classify_ghsl(ghsl_path, threshold=None):
    """
    Read the pre-classified GHSL binary raster produced by Stage 0.

    Stage 0 applies the 250 m² threshold SERVER-SIDE in GEE at native
    100m resolution, then exports the binary result (1=urban, 0=non-urban)
    at 30m using nearest-neighbour resampling.
    No threshold is applied here — the file already contains 0 and 1.
    """
    data, valid, profile = load_raster(ghsl_path)
    binary = data[0]

    # Replace non-finite nodata values with 0 before classification
    binary = np.where(np.isfinite(binary), binary, 0.0)

    H, W = binary.shape
    lc   = np.full((H, W), 3, dtype=np.uint8)  # start as grass/shrub everywhere

    # Stage 0 already encodes 1=urban and 0=non-urban.
    # Keep the condition strict so nodata (255) is never counted as urban.
    lc[(binary > 0.5) & (binary < 200)] = 1
    # lc[binary == 1.0] = 1
    lc[~valid]        = 255

    urban_km2 = float((lc == 1).sum() * (cfg.TARGET_RES / 1000) ** 2)
    log.info(f"  GHSL binary raster (threshold applied server-side at 100m)")
    log.info(f"  Urban area:     {urban_km2:.1f} km²")
    log.info(f"  Source: JRC Global Human Settlement Layer")
    log.info(f"  Reference: Pesaresi et al. (2023)")

    return lc, valid, profile


def classify_dynamic_world(dw_path):
    """
    Classify using Dynamic World pre-trained model.

    Remaps 9 DW classes to 7 standard land cover classes.
    DW classes: 0=water 1=trees 2=grass 3=flooded 4=crops
                5=shrub 6=built 7=bare 8=snow
    """
    data, valid, profile = load_raster(dw_path)

    # Clean NaN values before converting to integer class IDs
    raw = data[0].copy()
    raw[~np.isfinite(raw)] = 255
    dw = np.round(raw).astype(np.uint8)

    H, W = dw.shape
    lc   = np.full((H, W), 7, dtype=np.uint8)  # fallback class is bare

    for dw_cls, your_cls in cfg.DW_CLASS_MAP.items():
        lc[dw == dw_cls] = your_cls

    lc[~valid] = 255

    dw_names = {0:"water", 1:"trees", 2:"grass", 3:"flooded",
                4:"crops", 5:"shrub", 6:"built", 7:"bare", 8:"snow"}

    log.info(f"  Dynamic World class mapping:")
    for dw_cls, your_cls in cfg.DW_CLASS_MAP.items():
        n   = int((dw[valid] == dw_cls).sum())
        km2 = float(n * (cfg.TARGET_RES / 1000) ** 2)
        if km2 > 0.5:
            log.info(f"    DW {dw_cls} ({dw_names.get(dw_cls,'?'):<8}) "
                     f"→ {cfg.CLASSES[your_cls][0]}: {km2:.1f} km²")

    log.info(f"  Source: Google Dynamic World")
    log.info(f"  Reference: Brown et al. (2022) Nature Sci Data")

    return lc, valid, profile


# Area summary and saving helpers
def class_areas(lc, valid):
    return {
        cls: {
            "label": label,
            "km2": float((lc[valid] == cls).sum()
                         * (cfg.TARGET_RES / 1000) ** 2)
        }
        for cls, (label, _) in cfg.CLASSES.items()
    }


def save_landcover(lc, valid, profile, out_path):
    p = profile.copy()
    p.update(count=1, dtype="uint8", nodata=255, compress="lzw")
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(np.where(valid, lc, 255)[np.newaxis])
    log.info(f"  Saved: {out_path.name}")


# Plotting helpers
def get_lc_cmap():
    cmap = mcolors.ListedColormap(
        [c for _, (_, c) in cfg.CLASSES.items()])
    norm = mcolors.BoundaryNorm(
        [0.5] + [i + 0.5 for i in range(1, len(cfg.CLASSES) + 1)],
        len(cfg.CLASSES))
    return cmap, norm


def plot_landcover(lc, valid, yr, method, areas, out_path):
    """Single land cover map with legend and area statistics."""
    cmap_lc, norm_lc = get_lc_cmap()

    fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                             gridspec_kw={"width_ratios": [3, 1]})

    # Main land-cover map panel
    vis = np.where(valid, lc.astype(float), np.nan)
    axes[0].imshow(vis, cmap=cmap_lc, norm=norm_lc,
                   origin="upper", interpolation="none")
    axes[0].set_title(f"Land Cover — {cfg.AOI_NAME}  {yr}\n"
                      f"Method: {method}",
                      fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Per-class area bar chart
    labels = [cfg.CLASSES[c][0] for c in sorted(cfg.CLASSES)]
    values = [areas[c]["km2"] for c in sorted(cfg.CLASSES)]
    colors = [cfg.CLASSES[c][1] for c in sorted(cfg.CLASSES)]

    axes[1].barh(labels, values, color=colors,
                 edgecolor="white", height=0.6)
    for i, (lbl, val) in enumerate(zip(labels, values)):
        if val > 1:
            axes[1].text(val + max(values)*0.01, i,
                         f"{val:.0f} km²",
                         va="center", fontsize=8)
    axes[1].set_xlabel("Area (km²)")
    axes[1].set_title("Class Areas", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight",
                facecolor="white")
    plt.close()
    log.info(f"  Saved → {out_path.name}")


def plot_comparison(lc1, lc2, valid1, valid2,
                    areas1, areas2, method1, method2, out_path):
    """Side-by-side comparison map."""
    cmap_lc, norm_lc = get_lc_cmap()
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))

    for ax, lc, valid, yr, areas, method in [
        (axes[0], lc1, valid1,
         cfg.YEAR_HISTORICAL, areas1, method1),
        (axes[1], lc2, valid2,
         cfg.YEAR_RECENT, areas2, method2),
    ]:
        vis = np.where(valid, lc.astype(float), np.nan)
        ax.imshow(vis, cmap=cmap_lc, norm=norm_lc,
                  origin="upper", interpolation="none")
        urban = areas[1]["km2"]
        ax.set_title(f"{yr}  |  Urban: {urban:.1f} km²\n"
                     f"({method})",
                     fontsize=11, fontweight="bold")
        ax.axis("off")
        ax.legend(handles=[
            mpatches.Patch(
                color=c,
                label=f"{l}  ({areas[cls]['km2']:.0f} km²)")
            for cls, (l, c) in cfg.CLASSES.items()
            if areas[cls]["km2"] > 0.5
        ], loc="lower right", fontsize=8, framealpha=0.9)

    u1   = areas1[1]["km2"]
    u2   = areas2[1]["km2"]
    g    = u2 - u1
    p    = 100 * g / u1 if u1 > 0 else 0
    yrs  = cfg.YEAR_RECENT - cfg.YEAR_HISTORICAL
    rate = p / yrs if yrs > 0 else 0

    fig.suptitle(
        f"Land Cover Change — {cfg.AOI_NAME}  "
        f"{cfg.YEAR_HISTORICAL} → {cfg.YEAR_RECENT}\n"
        f"Urban: {u1:.1f} km² → {u2:.1f} km²  "
        f"(+{g:.1f} km²  +{p:.0f}%  {rate:.2f}%/year)",
        fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.DPI, bbox_inches="tight",
                facecolor="white")
    plt.close()
    log.info(f"  Saved → {out_path.name}")


# Main pipeline entry
def run():
    cfg.MAPS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("══════════════════════════════════════════════")
    log.info(f"STAGE 1 — LAND COVER CLASSIFICATION")
    log.info(f"  AOI: {cfg.AOI_NAME}")
    log.info(f"  {cfg.YEAR_HISTORICAL}: {cfg.METHOD_HISTORICAL}")
    log.info(f"  {cfg.YEAR_RECENT}:  {cfg.METHOD_RECENT}")
    log.info(f"  No training data. No manual thresholds.")
    log.info("══════════════════════════════════════════════")

    # Classify the historical year
    log.info(f"\n── {cfg.YEAR_HISTORICAL} "
             f"({cfg.METHOD_HISTORICAL}) ─────────────────────")

    if cfg.METHOD_HISTORICAL == "GHSL":
        if not cfg.GHSL_HISTORICAL_PATH.exists():
            log.error(f"Missing: {cfg.GHSL_HISTORICAL_PATH.name}")
            log.error("Run Stage 0 first")
            sys.exit(1)
        lc1, valid1, prof1 = classify_ghsl(cfg.GHSL_HISTORICAL_PATH)
        method1 = f"GHSL {cfg.YEAR_HISTORICAL} (JRC)"

    elif cfg.METHOD_HISTORICAL == "DynamicWorld":
        if not cfg.DW_HISTORICAL_PATH.exists():
            log.error(f"Missing: {cfg.DW_HISTORICAL_PATH.name}")
            sys.exit(1)
        lc1, valid1, prof1 = classify_dynamic_world(
            cfg.DW_HISTORICAL_PATH)
        method1 = f"Dynamic World {cfg.YEAR_HISTORICAL} (Google)"

    areas1 = class_areas(lc1, valid1)
    log.info(f"\n  Land cover {cfg.YEAR_HISTORICAL}:")
    for cls, d in areas1.items():
        if d["km2"] > 0.1:
            log.info(f"    {d['label']:<22} {d['km2']:>8.1f} km²")

    save_landcover(lc1, valid1, prof1,
                   cfg.OUTPUTS_DIR /
                   f"landcover_{cfg.YEAR_HISTORICAL}.tif")

    # Classify the recent year
    log.info(f"\n── {cfg.YEAR_RECENT} "
             f"({cfg.METHOD_RECENT}) ──────────────────────")

    if cfg.METHOD_RECENT == "DynamicWorld":
        if not cfg.DW_RECENT_PATH.exists():
            log.error(f"Missing: {cfg.DW_RECENT_PATH.name}")
            log.error("Run Stage 0 first")
            sys.exit(1)
        lc2, valid2, prof2 = classify_dynamic_world(
            cfg.DW_RECENT_PATH)
        method2 = f"Dynamic World {cfg.YEAR_RECENT} (Google)"

    elif cfg.METHOD_RECENT == "GHSL":
        if not cfg.GHSL_RECENT_PATH.exists():
            log.error(f"Missing: {cfg.GHSL_RECENT_PATH.name}")
            sys.exit(1)
        lc2, valid2, prof2 = classify_ghsl(cfg.GHSL_RECENT_PATH)
        method2 = f"GHSL {cfg.YEAR_RECENT} (JRC)"

    areas2 = class_areas(lc2, valid2)
    log.info(f"\n  Land cover {cfg.YEAR_RECENT}:")
    for cls, d in areas2.items():
        if d["km2"] > 0.1:
            log.info(f"    {d['label']:<22} {d['km2']:>8.1f} km²")

    save_landcover(lc2, valid2, prof2,
                   cfg.OUTPUTS_DIR /
                   f"landcover_{cfg.YEAR_RECENT}.tif")

    # Summarize results and store shared params for Stage 2
    u1   = areas1[1]["km2"]
    u2   = areas2[1]["km2"]
    g    = u2 - u1
    p    = 100 * g / u1 if u1 > 0 else 0
    yrs  = cfg.YEAR_RECENT - cfg.YEAR_HISTORICAL
    rate = (np.power(u2/u1, 1/yrs) - 1) * 100 \
           if u1 > 0 and u2 > 0 else 0

    params = {
        "aoi":              cfg.AOI_NAME,
        "method_historical": method1,
        "method_recent":     method2,
        "urban_historical_km2": u1,
        "urban_recent_km2":  u2,
        "growth_km2":        g,
        "growth_pct":        p,
        "annual_rate_pct":   rate,
        "areas_historical":  {k: v["km2"] for k, v in areas1.items()},
        "areas_recent":      {k: v["km2"] for k, v in areas2.items()},
    }
    with open(cfg.PIPELINE_DIR / "pipeline_params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Create and save all visualization figures
    log.info("\n── Generating visualisations ─────────────────")

    plot_landcover(lc1, valid1, cfg.YEAR_HISTORICAL,
                   method1, areas1,
                   cfg.MAPS_DIR /
                   f"landcover_{cfg.YEAR_HISTORICAL}.png")

    plot_landcover(lc2, valid2, cfg.YEAR_RECENT,
                   method2, areas2,
                   cfg.MAPS_DIR /
                   f"landcover_{cfg.YEAR_RECENT}.png")

    plot_comparison(lc1, lc2, valid1, valid2,
                    areas1, areas2, method1, method2,
                    cfg.MAPS_DIR / "landcover_comparison.png")

    log.info("\n══════════════════════════════════════════════")
    log.info("CLASSIFICATION COMPLETE")
    log.info("══════════════════════════════════════════════")
    log.info(f"  {cfg.YEAR_HISTORICAL} urban: {u1:.1f} km²  "
             f"({cfg.METHOD_HISTORICAL})")
    log.info(f"  {cfg.YEAR_RECENT} urban:  {u2:.1f} km²  "
             f"({cfg.METHOD_RECENT})")
    log.info(f"  Growth:      +{g:.1f} km²  (+{p:.1f}%)")
    log.info(f"  Annual rate: {rate:.2f}%/year")
    log.info("══════════════════════════════════════════════")
    log.info("✓ Stage 1 complete")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
