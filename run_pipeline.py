"""
Land Cover Change Detection Pipeline
======================================
Edit user_config.py then run this file.

Stages:
  0  Authenticate GEE and download all data automatically
  1  Classify land cover (method auto-selected by year)
  2  Change detection + all maps

Usage:
  python run_pipeline.py              # run everything (GEE auth appears only once)
  python run_pipeline.py --from-stage 1   # start from Stage 1 if data is already downloaded
  python run_pipeline.py --stages 2       # run just one stage
  python run_pipeline.py --info           # print current configuration
"""

import sys
import time
import argparse
import logging
import traceback
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    PIPELINE_DIR, AOI_NAME, TARGET_CRS,
    METHOD_HISTORICAL, METHOD_RECENT,
    YEAR_HISTORICAL, YEAR_RECENT,
    DATE_HISTORICAL_START, DATE_HISTORICAL_END,
    DATE_RECENT_START, DATE_RECENT_END,
)

PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PIPELINE_DIR / "pipeline.log", mode="a"),
    ]
)
log = logging.getLogger(__name__)

STAGES = {
    0: ("Download from GEE (automated)",  "scripts.00_gee_export"),
    1: ("Land Cover Classification",       "scripts.01_classify"),
    2: ("Change Detection",                "scripts.02_change_detection"),
}


def run_stage(num):
    name, module_path = STAGES[num]
    log.info(f"\n{'═'*55}")
    log.info(f"  STAGE {num} — {name}")
    log.info(f"{'═'*55}\n")
    t0 = time.time()
    try:
        mod = importlib.import_module(module_path)
        importlib.reload(mod)
        mod.run()
        log.info(f"\n✅ Stage {num} complete  ({time.time()-t0:.1f}s)")
        return True
    except SystemExit as e:
        return e.code == 0
    except Exception:
        log.error(f"\n❌ Stage {num} FAILED")
        log.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description=f"Land Cover Pipeline — {AOI_NAME}")
    parser.add_argument("--stages",     nargs="+", type=int)
    parser.add_argument("--from-stage", type=int)
    parser.add_argument("--info",       action="store_true")
    args = parser.parse_args()

    if args.info:
        import numpy as np
        lons = [c[0] for c in __import__("user_config").AOI_COORDS]
        lats = [c[1] for c in __import__("user_config").AOI_COORDS]
        print(f"\n{'═'*52}")
        print(f"  PIPELINE CONFIGURATION")
        print(f"{'═'*52}")
        print(f"  AOI:         {AOI_NAME}")
        print(f"  Centre:      {np.mean(lons):.4f}°E  {np.mean(lats):.4f}°N")
        print(f"  UTM zone:    {TARGET_CRS}")
        print(f"  Historical:  {YEAR_HISTORICAL} → {METHOD_HISTORICAL}")
        print(f"               {DATE_HISTORICAL_START} – {DATE_HISTORICAL_END}")
        print(f"  Recent:      {YEAR_RECENT} → {METHOD_RECENT}")
        print(f"               {DATE_RECENT_START} – {DATE_RECENT_END}")
        print(f"  Output:      {PIPELINE_DIR}")
        print(f"  Download:    direct from GEE (no Google Drive)")
        print(f"{'═'*52}\n")
        return

    if args.stages:
        to_run = args.stages
    elif args.from_stage is not None:
        to_run = list(range(args.from_stage, max(STAGES) + 1))
    else:
        to_run = list(STAGES.keys())

    invalid = [s for s in to_run if s not in STAGES]
    if invalid:
        log.error(f"Invalid stages: {invalid}. Valid: {list(STAGES)}")
        sys.exit(1)

    log.info("══════════════════════════════════════════════")
    log.info("  LAND COVER CHANGE DETECTION PIPELINE")
    log.info(f"  AOI:        {AOI_NAME}")
    log.info(f"  Historical: {YEAR_HISTORICAL} → {METHOD_HISTORICAL}")
    log.info(f"  Recent:     {YEAR_RECENT} → {METHOD_RECENT}")
    log.info(f"  Stages:     {to_run}")
    log.info("══════════════════════════════════════════════")

    t_total = time.time()

    for num in to_run:
        ok = run_stage(num)
        if not ok:
            log.error(f"\nStopped at Stage {num}")
            log.error(f"Resume: python run_pipeline.py --from-stage {num}")
            sys.exit(1)

    elapsed = time.time() - t_total
    log.info(f"\n{'═'*55}")
    log.info(f"  DONE  ({elapsed:.0f}s)")
    log.info(f"  Maps:  {PIPELINE_DIR}/maps/")
    log.info(f"  Stats: {PIPELINE_DIR}/outputs/change_statistics.csv")
    log.info("═" * 55)


if __name__ == "__main__":
    main()
