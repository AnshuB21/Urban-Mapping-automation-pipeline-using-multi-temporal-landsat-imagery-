"""
Stage 2 — Land Cover Change Detection
=======================================
This stage compares historical and recent land-cover maps.
It produces change stats, direction analysis, and final map outputs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import logging
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter

import config as cfg

log = logging.getLogger(__name__)


def load_aligned(path1, path2):
    """Load two land cover maps aligned to same grid."""
    with rasterio.open(path2) as ref:
        H, W      = ref.height, ref.width
        dst_t     = ref.transform
        dst_c     = ref.crs
        profile   = ref.profile.copy()
        lc2       = ref.read(1)

    with rasterio.open(path1) as src:
        lc1_raw = np.zeros((H, W), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=lc1_raw,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_t,
            dst_crs=dst_c,
            resampling=Resampling.nearest)

    valid = (lc1_raw != 255) & (lc2 != 255)
    return lc1_raw, lc2, valid, profile


def km2(mask, res=30):
    return float(mask.sum() * (res / 1000) ** 2)


def get_lc_cmap():
    cmap = mcolors.ListedColormap(
        [c for _, (_, c) in cfg.CLASSES.items()])
    norm = mcolors.BoundaryNorm(
        [0.5] + [i + 0.5 for i in range(1, len(cfg.CLASSES) + 1)],
        len(cfg.CLASSES))
    return cmap, norm


def run():
    cfg.MAPS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    p1 = cfg.OUTPUTS_DIR / f"landcover_{cfg.YEAR_HISTORICAL}.tif"
    p2 = cfg.OUTPUTS_DIR / f"landcover_{cfg.YEAR_RECENT}.tif"

    for p in [p1, p2]:
        if not p.exists():
            log.error(f"Missing: {p.name} — run Stage 1 first")
            sys.exit(1)

    # Load methods and metrics saved by Stage 1
    params_path = cfg.PIPELINE_DIR / "pipeline_params.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        method1 = params.get("method_historical", cfg.METHOD_HISTORICAL)
        method2 = params.get("method_recent",     cfg.METHOD_RECENT)
    else:
        method1 = cfg.METHOD_HISTORICAL
        method2 = cfg.METHOD_RECENT

    log.info("══════════════════════════════════════════════")
    log.info(f"STAGE 2 — CHANGE DETECTION  [{cfg.AOI_NAME}]")
    log.info(f"  {cfg.YEAR_HISTORICAL} ({method1})")
    log.info(f"  → {cfg.YEAR_RECENT} ({method2})")
    log.info("══════════════════════════════════════════════")

    lc1, lc2, valid, profile = load_aligned(p1, p2)
    H, W = lc1.shape

    # Build urban/non-urban masks for both years
    was_urban  = (lc1 == 1) & valid
    now_urban  = (lc2 == 1) & valid
    new_urban  = (~was_urban) & now_urban
    lost_urban = was_urban & (~now_urban)
    stable_u   = was_urban & now_urban

    # Change map legend: 0=stable non-urban, 1=stable urban,
    #                    2=new growth, 3=urban loss
    change = np.full((H, W), 0, dtype=np.uint8)
    change[stable_u]   = 1
    change[new_urban]  = 2
    change[lost_urban] = 3
    change[~valid]     = 255

    profile_bin = profile.copy()
    profile_bin.update(count=1, dtype="uint8",
                       nodata=255, compress="lzw")
    with rasterio.open(cfg.OUTPUTS_DIR/"change_map.tif",
                       "w", **profile_bin) as dst:
        dst.write(change[np.newaxis])

    # Analyze class transitions into and out of urban
    log.info("\n── What became urban ─────────────────────────")
    source_of_urban = {}
    for cls, (label, _) in cfg.CLASSES.items():
        if cls == 1:
            continue
        n   = int(((lc1 == cls) & (lc2 == 1) & valid).sum())
        km2_val = km2((lc1 == cls) & (lc2 == 1) & valid)
        source_of_urban[label] = km2_val
        if km2_val > 0.5:
            log.info(f"  {label:<22} → Urban: {km2_val:.1f} km²")

    log.info("\n── What urban became ─────────────────────────")
    urban_to = {}
    for cls, (label, _) in cfg.CLASSES.items():
        if cls == 1:
            continue
        km2_val = km2((lc1 == 1) & (lc2 == cls) & valid)
        urban_to[label] = km2_val
        if km2_val > 0.5:
            log.info(f"  Urban → {label:<22}: {km2_val:.1f} km²")

    # Estimate growth direction by AOI quadrants
    cy, cx = H // 2, W // 2
    dir_masks = {
        "N":  new_urban[:cy,     cx:cx+1],
        "NE": new_urban[:cy,     cx:    ],
        "E":  new_urban[cy:cy+1, cx:    ],
        "SE": new_urban[cy:,     cx:    ],
        "S":  new_urban[cy:,     cx:cx+1],
        "SW": new_urban[cy:,     :cx    ],
        "W":  new_urban[cy:cy+1, :cx    ],
        "NW": new_urban[:cy,     :cx    ],
    }
    dir_km2  = {d: km2(a) for d, a in dir_masks.items()}
    dominant = max(dir_km2, key=dir_km2.get)

    log.info("\n── Growth direction ──────────────────────────")
    for d, v in sorted(dir_km2.items(), key=lambda x: -x[1]):
        if v > 0:
            log.info(f"  {d:<4} {v:.1f} km²")

    # Compute summary statistics
    u1   = km2(was_urban)
    u2   = km2(now_urban)
    g    = km2(new_urban)
    loss = km2(lost_urban)
    net  = g - loss
    pct  = 100 * net / u1 if u1 > 0 else 0
    yrs  = cfg.YEAR_RECENT - cfg.YEAR_HISTORICAL
    rate = (np.power(u2/u1, 1/yrs) - 1) * 100 \
           if u1 > 0 and u2 > 0 else 0

    log.info(f"\n  Urban {cfg.YEAR_HISTORICAL}: {u1:.1f} km²")
    log.info(f"  Urban {cfg.YEAR_RECENT}: {u2:.1f} km²")
    log.info(f"  New growth:   +{g:.1f} km²")
    log.info(f"  Urban loss:   -{loss:.1f} km²")
    log.info(f"  Net change:   +{net:.1f} km²  (+{pct:.1f}%)")
    log.info(f"  Annual rate:  {rate:.2f}%/year")
    log.info(f"  Direction:    {dominant}")

    # Save summary table as CSV
    with open(cfg.OUTPUTS_DIR/"change_statistics.csv",
              "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value", "Unit"])
        w.writerow([f"Urban_{cfg.YEAR_HISTORICAL}", f"{u1:.2f}", "km²"])
        w.writerow([f"Urban_{cfg.YEAR_RECENT}",     f"{u2:.2f}", "km²"])
        w.writerow(["New_growth",    f"{g:.2f}",    "km²"])
        w.writerow(["Urban_loss",    f"{loss:.2f}", "km²"])
        w.writerow(["Net_change",    f"{net:.2f}",  "km²"])
        w.writerow(["Growth_pct",    f"{pct:.2f}",  "%"])
        w.writerow(["Annual_rate",   f"{rate:.2f}", "%/year"])
        w.writerow(["Dominant_dir",  dominant,      ""])
        w.writerow(["Method_historical", method1,   ""])
        w.writerow(["Method_recent",     method2,   ""])
        w.writerow([])
        w.writerow(["Source_class", "Converted_to_urban_km2"])
        for lbl, v in source_of_urban.items():
            if v > 0:
                w.writerow([lbl, f"{v:.2f}"])

    # Create map and chart outputs
    log.info("\n── Generating maps ───────────────────────────")
    cmap_lc, norm_lc = get_lc_cmap()
    cmap_chg = mcolors.ListedColormap(
        ["#d9ead3", "#f6b26b", "#cc0000", "#6fa8dc"])
    norm_chg = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)

    # 1) Three-panel land-cover and change map
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, lc, yr in [
        (axes[0], np.where(lc1==255, np.nan, lc1.astype(float)),
         cfg.YEAR_HISTORICAL),
        (axes[1], np.where(lc2==255, np.nan, lc2.astype(float)),
         cfg.YEAR_RECENT),
    ]:
        ax.imshow(lc, cmap=cmap_lc, norm=norm_lc,
                  origin="upper", interpolation="none")
        ax.set_title(f"Land Cover {yr}",
                     fontsize=12, fontweight="bold")
        ax.axis("off")
        ax.legend(handles=[
            mpatches.Patch(color=c, label=l)
            for _, (l, c) in cfg.CLASSES.items()
        ], loc="lower right", fontsize=7)

    chg_vis = np.where(change == 255, np.nan, change.astype(float))
    axes[2].imshow(chg_vis, cmap=cmap_chg, norm=norm_chg,
                   origin="upper", interpolation="none")
    axes[2].set_title(
        f"Urban Change\n{cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}",
        fontsize=12, fontweight="bold")
    axes[2].axis("off")
    axes[2].legend(handles=[
        mpatches.Patch(color="#d9ead3",
                       label="Stable non-urban"),
        mpatches.Patch(color="#f6b26b",
                       label=f"Stable urban ({km2(stable_u):.0f} km²)"),
        mpatches.Patch(color="#cc0000",
                       label=f"New urban (+{g:.0f} km²)"),
        mpatches.Patch(color="#6fa8dc",
                       label=f"Urban loss (-{loss:.0f} km²)"),
    ], loc="lower right", fontsize=8)

    fig.text(0.01, 0.02,
             f"{cfg.AOI_NAME}  {cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}\n"
             f"Urban: {u1:.1f}→{u2:.1f} km²  "
             f"+{net:.1f} km² (+{pct:.0f}%)\n"
             f"Rate: {rate:.2f}%/year  |  "
             f"Direction: {dominant}\n"
             f"Historical: {method1}\n"
             f"Recent: {method2}",
             fontsize=8,
             bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9))

    plt.suptitle(
        f"Urban Expansion — {cfg.AOI_NAME} "
        f"{cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}",
        fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR/"urban_expansion_map.png",
                dpi=cfg.DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  Saved → urban_expansion_map.png")

    # 2) Polar chart of growth direction
    order  = ["N","NE","E","SE","S","SW","W","NW"]
    vals   = [dir_km2[d] for d in order]
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"},
                           figsize=(7, 7))
    ax.bar(angles, vals, width=2*np.pi/8*0.8, bottom=0,
           color=["#cc0000" if v > 0 else "#e0e0e0" for v in vals],
           alpha=0.85, edgecolor="white")
    ax.set_xticks(angles)
    ax.set_xticklabels(order, fontsize=11)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(
        f"Growth Direction — {cfg.AOI_NAME}\n"
        f"{cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}",
        fontsize=13, fontweight="bold", pad=20)
    for ang, val in zip(angles, vals):
        if val > 0.5:
            ax.text(ang, val + max(vals) * 0.05,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR/"growth_direction_rose.png",
                dpi=cfg.DPI, bbox_inches="tight")
    plt.close()
    log.info("  Saved → growth_direction_rose.png")

    # 3) Urban density and hotspot heatmaps
    log.info("  Generating heatmaps ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()

    def heatmap(binary, sigma=15):
        d = gaussian_filter(binary.astype(np.float32), sigma=sigma)
        d[~valid] = np.nan
        return d

    panels = [
        (heatmap(was_urban),
         f"Urban Density {cfg.YEAR_HISTORICAL}"),
        (heatmap(now_urban),
         f"Urban Density {cfg.YEAR_RECENT}"),
        (heatmap(new_urban),
         f"Growth Hotspots {cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}"),
        (heatmap(now_urban, sigma=20),
         f"Urban Concentration {cfg.YEAR_RECENT}"),
    ]

    for ax, (data, title) in zip(axes, panels):
        pos  = data[np.isfinite(data) & (data > 0)]
        vmin = np.percentile(pos, 5)  if pos.size > 0 else 0
        vmax = np.percentile(pos, 98) if pos.size > 0 else 1
        msk  = np.ma.masked_where(
            (data <= 0) | ~np.isfinite(data), data)
        im = ax.imshow(msk, cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.046,
                     pad=0.04, label="Density")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.suptitle(f"Urban Density Heatmaps — {cfg.AOI_NAME}",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cfg.MAPS_DIR/"urban_heatmaps.png",
                dpi=cfg.DPI, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info("  Saved → urban_heatmaps.png")

    # 4) Chart showing which classes converted to urban
    labels_t = [l for l, v in source_of_urban.items() if v > 0.1]
    values_t = [v for v in source_of_urban.values() if v > 0.1]

    if labels_t:
        fig, ax = plt.subplots(figsize=(10, 5))
        colours = [c for cls, (l, c) in cfg.CLASSES.items()
                   if l in labels_t]
        bars = ax.barh(labels_t, values_t,
                       color=colours[:len(labels_t)],
                       edgecolor="white", height=0.6)
        for bar, val in zip(bars, values_t):
            ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f} km²", va="center", fontsize=10)
        ax.set_xlabel("Area converted to Urban (km²)", fontsize=11)
        ax.set_title(
            f"Source of Urban Growth — {cfg.AOI_NAME}\n"
            f"{cfg.YEAR_HISTORICAL}→{cfg.YEAR_RECENT}",
            fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(cfg.MAPS_DIR/"class_transitions.png",
                    dpi=cfg.DPI, bbox_inches="tight")
        plt.close()
        log.info("  Saved → class_transitions.png")

    log.info("\n══════════════════════════════════════════════")
    log.info("CHANGE DETECTION COMPLETE")
    log.info("══════════════════════════════════════════════")
    log.info(f"  Urban {cfg.YEAR_HISTORICAL}: {u1:.1f} km²")
    log.info(f"  Urban {cfg.YEAR_RECENT}: {u2:.1f} km²")
    log.info(f"  Net growth:  +{net:.1f} km²  (+{pct:.1f}%)")
    log.info(f"  Annual rate: {rate:.2f}%/year")
    log.info(f"  Direction:   {dominant}")
    log.info("══════════════════════════════════════════════")
    log.info("✓ Stage 2 complete")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s")
    run()
