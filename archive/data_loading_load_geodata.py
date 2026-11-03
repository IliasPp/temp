"""load_geodata() from scripts/tdlinx/data_loading.py -- superseded by
run_pipeline.region_layers(). Needed the Land lookups now in config_legacy.py."""
import geopandas as gpd

from archive.config_legacy import BUNDESLAND_NAMES, SCHLUESSEL_TO_ABR, SHAPEFILE_PATH


def load_geodata(shapefile_path=None):
    """Load shapefile and derive Kreis- and Bundesland-level GeoDataFrames."""
    shapefile_path = shapefile_path or SHAPEFILE_PATH
    gdf = gpd.read_file(shapefile_path)

    kreis_gdf = (
        gdf
        .assign(kreis_key=gdf["ARS"].str[:5])
        .dissolve(by="kreis_key", aggfunc="first")
        .reset_index()
    )

    bundesland_gdf = (
        gdf
        .assign(bl_key=gdf["ARS"].str[:2])
        .dissolve(by="bl_key", aggfunc="first")
        .reset_index()
    )
    bundesland_gdf["bl_key"] = (
        bundesland_gdf["bl_key"].map(SCHLUESSEL_TO_ABR).map(BUNDESLAND_NAMES)
    )

    return gdf, kreis_gdf, bundesland_gdf
