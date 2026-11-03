"""Input paths for the panel build (scripts/tdlinx) and the region layer.

The pipeline's own paths and parameters live in run_pipeline.Config. Everything
else that used to sit here (column groups, colours, Land lookups, figure dirs)
is in archive/config_legacy.py. Importing this module has no side effects.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

DATA_PATH = PROJECT_ROOT.parent / "TD 2014-2025_LEH + GAM + Drogerie.xlsx"   # TDLinx workbook   # TDLinx workbook
CLEAN_2023_PATH = PROJECT_ROOT.parent / "2023_clean.xlsx"                    # 2023 sheet, re-exported
SHAPEFILE_PATH = DATA_DIR / "admin_shapefiles/VG5000_GEM.shp"         # Gemeinden, DHDN
CSV_PATH = DATA_DIR / "coordinates/adressen_mit_koordinaten_3.csv"    # geocoding cache
