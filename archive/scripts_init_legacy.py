"""edeka_project.scripts — core package for the EDEKA analysis pipeline."""

__version__ = "0.1.0"

from . import config

__all__ = ["config"]

# from .config import *
# from trade_dimension.data_loading import load_data, load_geodata
# from trade_dimension.cleaning import clean, assign_gruppe, geocode_sheets
# from trade_dimension.metrics import (
#     compute_hhi, combine_hhi, filter_by_typ,
#     market_share, market_share_by_kreis_and_group,
#     fe_basic, fe_chain_dummies, fe_chain_and_format,
#     fe_vkfl_format_interaction, fe_chain_vkfl_interaction,
#     fe_single_chain, fe_compare_chains,
# )
# from visualisations.plots import (
#     plot_top_chains,
#     plot_universal,
#     plot_bundesland,
#     plot_germany,
#     plot_hhi_scatter,
#     plot_hhi_vs_firms,
#     plot_hhi_distributions,
#     plot_hhi_by_typ,
#     plot_hhi_map_kreis,
#     plot_hhi_map_bundesland,
#     plot_hhi_map_universal,
#     plot_market_share_map,
# )

# from time_dimension.time_dimension import (
#     store_lifespan, entrants, exits, chain_switches, shop_type_switches,
#     annual_transitions, annual_by_gruppe, survival_table, chain_switch_matrix,
# )


# from yellow_pages.yellow_pages import match_by_address_and_gruppe
# # from .standardization_match import add_standardized_cols

# from isochrones.isochrones import build_isochrone_catchments, route_nearest_competitors, attach_road_distances
# from isochrones.voronoi import build_voronoi, plot_voronoi_dominance, plot_voronoi_chain, find_food_deserts
# from isochrones.distances import nearest_competitors, summarise_by_gruppe
