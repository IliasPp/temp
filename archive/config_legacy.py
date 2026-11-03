import os
from pathlib import Path

# Project root = one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
NOTEBOOKS_DIR   = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR     = PROJECT_ROOT / "scripts"
FIGURES_DIR     = SCRIPTS_DIR  / "figures"
ISOCHRONES_DIR  = SCRIPTS_DIR  / "isochrones"
TDLINX_DIR      = SCRIPTS_DIR  / "tdlinx"
TIME_DIM_DIR    = SCRIPTS_DIR  / "time_dimension"
VIS_DIR         = SCRIPTS_DIR  / "visualizations"

# Create dirs that should always exist
for _d in (DATA_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Any project-wide constants
RANDOM_SEED = 42
DEFAULT_CRS = "EPSG:4326"

# Optional: environment-specific values (API keys, DB paths) via .env
# from dotenv import load_dotenv
# import os
# load_dotenv(PROJECT_ROOT / ".env")
# SOME_API_KEY = os.getenv("SOME_API_KEY")


TYPEN = [
    "Angeschlossener GAM",
    "Discount Geschäft",    #
    "Drogeriemarkt",
    "Getränkemarkt",
    "Großer Supermarkt",    #
    "Kleiner Supermarkt",   #
    "SB-Warenhaus",         #
    "Supermarkt",           #
    "Verbrauchermarkt",     #
]



BUNDESLAND_NAMES = {
    "SH": "Schleswig-Holstein",
    "HH": "Hamburg",
    "NI": "Niedersachsen",
    "HB": "Bremen",
    "NW": "Nordrhein-Westfalen",
    "HE": "Hessen",
    "RP": "Rheinland-Pfalz",
    "BW": "Baden-Württemberg",
    "BY": "Bayern",
    "SL": "Saarland",
    "BE": "Berlin",
    "BB": "Brandenburg",
    "MV": "Mecklenburg-Vorpommern",
    "SN": "Sachsen",
    "ST": "Sachsen-Anhalt",
    "TH": "Thüringen",
}

SCHLUESSEL_TO_ABR = {
    "01": "SH", "02": "HH", "03": "NI", "04": "HB",
    "05": "NW", "06": "HE", "07": "RP", "08": "BW",
    "09": "BY", "10": "SL", "11": "BE", "12": "BB",
    "13": "MV", "14": "SN", "15": "ST", "16": "TH",
}

SCHLUESSEL_TO_NAME = {
    k: BUNDESLAND_NAMES[v] for k, v in SCHLUESSEL_TO_ABR.items()
}

GROUP_COLORS = {
    # Major grocery groups
    "Edeka": "#005CA9",
    "Rewe": "#CC071E",
    "Schwarz": "#FFF000",
    "Aldi": "#003087",
    "Norma": "#E2001A",
    # Drugstores
    "Rossmann": "#E2001A",
    "dm": "#D40046",
    "Müller": "#F4A700",
    # Smaller / regional groups
    "Tegut": "#E30613",
    "Fristo": "#FF6600",
    "Hoffmann": "#006633",
    "Dennree": "#8BC34A",
    "Bünting": "#004A97",
    "Bartels-Langness": "#6A0DAD",
    # Fallback
    "Other": "#AAAAAA",
    "None": "#CCCCCC",
}

DIMENSIONS = {
    "identity"          : ["TDLinx",
                            "Name", 
                            "Name-juristisch", 
                            "Telefon", 
                            "Handelszentrale", 
                            "Banner", 
                            "Vertriebslinie", 
                            "Gruppe"],
    "type"              : ["Typ", 
                           "Nielsentyp", 
                           "Shop Type"],
    "location"          : ["Plz", 
                           "Ort", 
                           "Ortsteil", 
                           "Straße", 
                           "Ballungsraum", 
                           "Nielsengebiet", 
                           "Bundesland", 
                           "Regierungsbezirk", 
                           "Kreis", 
                           "Gemeindekennziffer", 
                           "Vertriebsregion Name",
                           "Vertriebslinie TDLinx",
                           "Handelszentrale TDLinx", 
                           "AGS5", 
                           "lat", 
                           "lon", 
                           "full_address"],
    "time"              : ["Eröffnung / Übernahme", 
                           "Ersteröffnungsdatum", 
                           "Wiedereröffnungsdatum"],
    "competition"       : ["Umsatzpotential Brutto (Mio EUR)", 
                           "Umsatzpotential Brutto Food (Mio EUR)",	
                           "Umsatzpotential Brutto Nonfood (Mio EUR)", 
                           "VKFL",	
                           "Verkaufsfläche Food", 
                           "Verkaufsfläche Nonfood"],
    "characteristics"   : ["Betriebsform zur Handelszentrale", 
                           "Parkplätze", 
                           "Kassen"],
    "panel"             : ["year", ] 
}

MARKET_ACTOR = "Gruppe"

MARKET_DEFINING_COLS = ["AGS5", "Shop Type"]

COMPETING_IN = "VKFL"  # change to "Umsatzpotential Brutto (Mio EUR)" for revenue-based competition

DATA_PATH       = PROJECT_ROOT.parent / "TD 2014-2025_LEH + GAM + Drogerie.xlsx"
SHAPEFILE_PATH  = DATA_DIR      / "admin_shapefiles/VG5000_GEM.shp"
CSV_PATH        = DATA_DIR      / "coordinates/adressen_mit_koordinaten_3.csv"
CLEAN_2023_PATH = PROJECT_ROOT.parent / "2023_clean.xlsx"
TIF_PATH        = DATA_DIR      / "pop/ghs_pop_germany_clipped.tif"


SAVE_FIGURE = True
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)
