---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.6
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Setup


Requirements:
- polars
- fastexcel
- geopandas
- matplotlib
- scipy
- scikit-learn
- pyarrow
- mapclassify

```python
import polars as pl, geopandas as gpd

plz = (gpd.read_file(r"C:\Users\f-papaio\Downloads\temp\AHS2023-toolkit\APPLICATIONS\SHAPES\POSTCODES\postcode_clean_final.shp")[["ZIP_CODE", "geometry"]]
          .rename(columns={"ZIP_CODE": "postcode_id"}))

df = pl.read_csv(r"C:\Users\f-papaio\Downloads\temp\AHS2023-toolkit\APPLICATIONS\DATA\OUTPUT\2026\AHS-Index-res-RENT-PLZ-2026\AHS-Index-res-RENT-PLZ-2026-wide.csv", schema_overrides={"postcode_id": pl.Utf8})
long = (
    df.with_columns(pl.col("postcode_id").str.zfill(5))
      .unpivot(index="postcode_id")
      .with_columns(
          year=pl.col("variable").str.extract(r"(\d{4})").cast(pl.Int16),
          kind=pl.when(pl.col("variable").str.contains("_se"))
                 .then(pl.lit("price_qm_se")).otherwise(pl.lit("price_qm")),
      )
      .pivot(on="kind", index=["postcode_id", "year"], values="value")
      .sort("postcode_id", "year")
)

plz["postcode_id"] = plz["postcode_id"].astype("Int64").astype(str).str.zfill(5)

csv_keys, shp_keys = set(long["postcode_id"].unique()), set(plz["postcode_id"])
print("in CSV, not in shp:", len(csv_keys - shp_keys))
print("in shp, not in CSV:", len(shp_keys - csv_keys))\

merged = plz.merge(long.to_pandas(), left_on="postcode_id", right_on="postcode_id", how="left")
```

```python
merged.describe()
```

```python
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

def plot_year(year, col="price_qm", out=None, vmin=None, vmax=None):
    d = long.filter(pl.col("year") == year).select("postcode_id", col).to_pandas()
    g = plz.merge(d, on="postcode_id", how="left")
    vmin = vmin or np.nanpercentile(g[col], 0.5)
    vmax = vmax or np.nanpercentile(g[col], 99.5)

    fig, ax = plt.subplots(figsize=(6, 8))
    g.plot(column=col, ax=ax, cmap="viridis", norm=LogNorm(vmin, vmax),
           legend=True,
           legend_kwds={"label": "€/m²", "shrink": 0.5, "extend": "both",
                        "ticks": [4, 5, 6, 8, 10, 12, 15, 20]},
           missing_kwds={"color": "lightgrey"}, linewidth=0, rasterized=True)
    ax.get_figure().axes[-1].yaxis.set_major_formatter("{x:.0f}")  # plain tick labels, not 10^x
    ax.set_axis_off(); ax.set_title(f"Residential rent per m², {year}")
    if out: fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig, ax

plot_year(2023) 
```

# Reading, Cleaning

```python
import sys
from pathlib import Path
import polars as pl

PROJECT_ROOT = Path.cwd().resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
```

```python
from archive.config_legacy import CSV_PATH, DATA_PATH, SAVE_FIGURE, FIGURE_DIR
from archive.data_loading_load_geodata import load_geodata
from scripts.tdlinx.data_loading import load_data
from scripts.tdlinx.cleaning import clean
# test commits
import sys
from pathlib import Path

```

```python
# Load data

edeka           = load_data()
edeka_unclean   = edeka.copy()  # Keep a copy of the uncleaned data for reference

gdf, kreis_gdf, bundesland_gdf = load_geodata()

# Clean data 
from scripts.tdlinx.cleaning import clean

edeka, cleaning_report = clean(edeka, report=True)

# Import metrics
from scripts.tdlinx.metrics import (
    compute_hhi,
    combine_hhi,
    filter_by_typ,
    market_share,
    market_share_by_kreis_and_group,
)

# Import plots
# from scripts.visualizations.plots import (
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

from scripts.tdlinx.cleaning import geocode_sheets, attach_coords_only
# edeka = geocode_sheets(edeka) # no need to geocode, all already geocoded. Just attach coords from the CSV file to the DF.
edeka = attach_coords_only(edeka)
from scripts.tdlinx.panel import build_panel, panel_summary, get_entrants, get_exits
panel = build_panel(edeka)
from scripts.tdlinx.cleaning import gate_supplied
panel = gate_supplied(panel)
panel.write_parquet(r"X:\Project-Shares\germart-lite\germart-papaio\panel")
```

```python
panel.columns
```

```python
import polars as pl
import scripts.run_pipeline as P
from scripts.runconfig import RunConfig
cfg = RunConfig.load(r"runs\baseline.toml").pipeline_config()
p = P.load_panel(cfg, shop_types=["Supermarkt", "Discount Geschäft"]).with_columns(
    pl.col("TDLinx").cast(pl.Utf8).alias("id"))
B = "Betriebsform zur Handelszentrale"

# a) what the relationship field says, and how it maps onto today's Gruppe
print(p.group_by(B).agg(pl.len().alias("rows"), pl.col("VKFL").sum().round(0).alias("vkfl"))
       .sort("rows", descending=True))
print(p.filter(pl.col("Gruppe").is_in(["Edeka", "Rewe"]))
       .group_by("Gruppe", B).len().sort(["Gruppe", "len"], descending=[False, True]))

# b) the 117 stores: relationship and buying centre, 2018-2021
ids = (p.filter(pl.col("year") == 2019, pl.col("Gruppe") == "Edeka").select("id")
       .join(p.filter(pl.col("year") == 2020, pl.col("Gruppe").is_null()).select("id"), on="id"))["id"]
print(p.filter(pl.col("id").is_in(ids.to_list()), pl.col("year").is_between(2018, 2021))
       .group_by("year", B, "Handelszentrale").len().sort(["year", "len"], descending=[False, True]))

# c) where regex classification rests on the shop's own name alone
for g, pat in {"Aldi": r"aldi", "Norma": r"norma", "Hoffmann": r"hoffmann",
               "Bartels-Langness": r"bartels|bela", "Klaas + Kock": r"klaas|k\s*\+\s*k"}.items():
    hit = p.filter(pl.col("Gruppe") == g,
                   ~pl.col("Banner").fill_null("").str.contains(f"(?i){pat}"),
                   ~pl.col("Handelszentrale").fill_null("").str.contains(f"(?i){pat}"))
    print(g, hit.height, hit.select("Name", "Name-juristisch").unique().head(5).rows())
```

```python
from scripts.visualizations.plots import (
    plot_top_chains,
    plot_universal,
    plot_bundesland,
    plot_germany,
    plot_hhi_scatter,
    plot_hhi_vs_firms,
    plot_hhi_distributions,
    plot_hhi_by_typ,
    plot_hhi_map_kreis,
    plot_hhi_map_bundesland,
    plot_hhi_map_universal,
    plot_market_share_map,
)
```

```python
nearest_competitors = pl.read_parquet(r"X:\Project-Shares\germart-lite\germart-papaio\edeka_project\\data\nearest competitors\nearest_competitors.parquet")
nearest_competitors.head()
```

# Analysis

## Run 
```python
# So we can start right away without reading, cleaning ("panel" is an already cleaned, saved parq. file)
panel = pl.read_parquet("panel")
```
```python
panel.filter(pl.col("yearc) == 2025, pl.col("AGS5") == "02000",
             pl.col("Shop Type") == "Supermarkt", pl.col("Gruppe").is_null()
             ).select("TDLinx", "Name", "Banner", "Name-juristisch", "Handelszentrale")
```

```python
panel.filter(
    pl.col("year") == 2024,
    pl.col("Bundesland") == "Bayern",
    pl.col("Shop Type").is_in(["Supermarkt", "Discount Geschäft"])
).sort(pl.col("VKFL")).select("VKFL", "Verkaufsfläche Food", "Verkaufsfläche Nonfood").show(None)
```

```python
from scripts.tdlinx.data_loading import load_geodata
gdf, kreis_gdf, bundesland_gdf = load_geodata()
```

```python
from scripts.tdlinx.metrics import compute_hhi, market_share
from scripts.visualizations.plots import plot_top_chains, plot_germany, plot_universal, plot_hhi_map_universal, plot_market_share_map
from scripts.isochrones.voronoi import build_voronoi, cells_by_chain, find_food_deserts, plot_voronoi_dominance, plot_voronoi_chain
```

## HHI, market share statistics and plotting

```python
hhi = compute_hhi(
    panel, 
    year="ALL",
)
```

```python
hhi.filter(pl.col("Shop Type") == "Supermarkt").describe()
```

```python
hhi.filter(pl.col("Shop Type") == "Supermarkt").describe()
```

```python
plot_hhi_map_universal(
    hhi, 
    year=2025,
    typ="Supermarkt", 
    gdf=bundesland_gdf,
    kreis_gdf=kreis_gdf
)
```

```python
market_sh = market_share(panel, group_col= "Gruppe")
```

```python
plot_top_chains(
    panel, name_col="Gruppe", 
    year= 2025, 
)
```

```python
plot_market_share_map(panel, gdf=gdf)
```

```python
# diagnostics for HHI
from scripts.visualizations.plots import plot_hhi_vs_firms
plot_hhi_vs_firms(hhi)
```

```python
plot_universal("Heidelberg", level="kreis", shapefile=kreis_gdf, dataframe=panel.filter(pl.col("Shop Type") == "Drogeriemarkt"), bundesland_gdf=bundesland_gdf, n_actors=5, year=2025)
```

```python
panel.filter(pl.col("Kreis").str.contains("Heidel")).unique(pl.col("Kreis"))
```

```python
plot_germany(panel.filter(pl.col("year") == 2025), bundesland_gdf, color_by="Gruppe", n_actors=8)
```

```python
plot_universal(
    "Mannheim",
    dataframe=panel,
    bundesland_gdf=bundesland_gdf,
    kreis_gdf=kreis_gdf,
    year=2025,
)

```

## Voronoi Catchment Areas

```python
build_voronoi(panel, year=2025)
```

```python
plot_voronoi_dominance(panel, year=2025, gdf=gdf, shop_type="Discount Geschäft")

```

```python
plot_voronoi_dominance(
    panel, year=2025, gdf=bundesland_gdf,
    shop_type=["Supermarkt", "Discount Geschäft"]
)
```

```python
# cells_by_chain(panel, year = 2025)
```

```python
plot_voronoi_chain(panel, year = 2025, gdf=bundesland_gdf, gruppe="Edeka")
```

```python
# find_food_deserts(panel, year = 2025, clip_to=gdf, shop_type=["Supermarkt", "Discount Geschäft"])
```

## Time Dimension Analysis

```python
from scripts.time_dimension.time_dimension import annual_by_gruppe, chain_switches
transitiion=chain_switches(panel)
transitiion.filter(pl.col("from_Gruppe") == "Kaiser's Tengelmann").show()
```

# Misc. 


## Voronoi area append

```python
from scripts.tdlinx.metrics import pair_catchment_areas_panel
catchments = pair_catchment_areas_panel(
    panel, clip_to=gdf,
    shop_type_groups={
        "grocery": ["Supermarkt", "Discount Geschäft", "Verbrauchermarkt", "SB-Warenhaus"],
        "drugstore": ["Drogeriemarkt"],
        "beverage": ["Getränkemarkt", "Angeschlossener GAM"],
    },
)

# Join back to the panel
panel_with_catchment = panel.join(
    catchments.select(["TDLinx", "year", "cell_area_km2", "area_market_share_pct", "group_name"]),
    on=["TDLinx", "year"], how="left",
)
```

```python
panel_with_catchment.filter(
    pl.col("year") == 2025
).select(
    pl.col("Gruppe"),
    pl.col("area_market_share_pct"),
).unique(
    pl.col("Gruppe")    
).sort(
    pl.col("area_market_share_pct"), 
    descending=True
).show()
```

```python
def chain_area_market_share(
    catchments: pl.DataFrame,
    panel: pl.DataFrame,
) -> pl.DataFrame:
    """
    Aggregate the per-store catchment shares into per-(Gruppe, group_name, year)
    market shares. Edeka's grocery market share in 2024 is the sum of all
    Edeka grocery stores' cell areas, divided by the total grocery cell area.
    """
    # Attach Gruppe from the panel
    catchments_with_gruppe = catchments.join(
        panel.select(["TDLinx", "year", "Gruppe"]),
        on=["TDLinx", "year"], how="left",
    )

    return (
        catchments_with_gruppe
        .group_by(["year", "group_name", "Gruppe"])
        .agg(
            pl.col("cell_area_km2").sum().alias("chain_area_km2"),
            pl.len().alias("n_stores"),
        )
        .with_columns(
            pl.col("chain_area_km2")
              .sum()
              .over(["year", "group_name"])
              .alias("group_total_km2")
        )
        .with_columns(
            (pl.col("chain_area_km2") / pl.col("group_total_km2") * 100)
            .alias("area_market_share_pct")
        )
        .sort(["year", "group_name", "area_market_share_pct"], descending=[False, False, True])
    )
```

```python
# Per-store catchment share (what fraction of Germany this store owns)
catchments = pair_catchment_areas_panel(
    panel, clip_to=gdf,
    shop_type_groups={
        "grocery":   ["Supermarkt", "Discount Geschäft"],
        "drugstore": ["Drogeriemarkt"],
        "beverage":  ["Getränkemarkt"],
    },
)

# Per-chain catchment share (what fraction of Germany this chain owns)
chain_shares = chain_area_market_share(catchments, panel)

chain_shares.filter(
    pl.col("year") == 2025, 
    pl.col("group_name") == "grocery"
).select(
    pl.col("Gruppe"),
    pl.col("area_market_share_pct")
).show()
```

```python
market_sh.filter(
    pl.col("year") == 2025
).select(
    pl.col("Gruppe"),
    pl.col("market_share_pct")
).show()
```

```python
# from functions.distances import nearest_competitors, summarise_by_gruppe 
# near = {}
# for index in sorted(panel["year"].unique().to_list()):
#     near[index] = nearest_competitors(panel, year=index, k=500)
# near[2024].show(200)

# summary = summarise_by_gruppe(near)
# summary.show()
```

```python
# # calculate and save nearest 500 competitors for each location, save to parquet files for use.
# import os

# os.makedirs("nearest_competitors", exist_ok=True)

# for year in sorted(panel["year"].unique().to_list()):
#     out_path = f"nearest_competitors/k20_year_{year}.parquet"
#     if os.path.exists(out_path):
#         print(f"{year}: cached, skipping")
#         continue
    
#     near_year = nearest_competitors(panel, year=year, k=500)
#     near_year.write_parquet(out_path)
#     print(f"{year}: wrote {len(near_year):,} rows")

# near_2024 = pl.read_parquet("nearest_competitors/k20_year_2024.parquet")
```

```python
# near_2024.sort("distance_m", descending=True).show()
```

```python
# from functions.distances import build_nearest_competitors_panel
# build_nearest_competitors_panel(panel, k=50)
```

```python
# near = pl.read_parquet("pairs_canonical.parquet")
```

```python
# import polars as pl
# import os
# import glob
    
# # Read all per-year files, add year column from filename, stack
# files = sorted(glob.glob("nearest_competitors/k20_50_year_*.parquet"))
# print(f"Found {len(files)} files")

# frames = []
# for f in files:
#     year = int(f.split("_")[-1].replace(".parquet", ""))
#     df = pl.read_parquet(f).with_columns(pl.lit(year).alias("year"))
#     frames.append(df)
#     print(f"  {year}: {len(df):,} rows")

# combined = pl.concat(frames, how="vertical")
# combined.write_parquet("nearest_competitors.parquet")
# print(f"\nTotal rows before dedup: {len(combined):,}")

```

```python
# import polars as pl
# combined = pl.read_parquet("nearest_competitors.parquet")
# def unique_pairs(near: pl.DataFrame) -> pl.DataFrame:
#     """
#     For routing purposes only — reduce (store, competitor) to unique
#     unordered pairs. Each haversine distance is still in both directions
#     in `near`; this dedupes the *pairs* so we only route each once.
#     """
#     return (
#         near
#         .with_columns([
#             pl.min_horizontal(["TDLinx", "competitor_TDLinx"]).alias("_a"),
#             pl.max_horizontal(["TDLinx", "competitor_TDLinx"]).alias("_b"),
#         ])
#         .unique(subset=["_a", "_b", "year"])
#         .select(["_a", "_b", "year", "distance_m"])
#         .rename({"_a": "store_a", "_b": "store_b"})
#     )
```

```python
# import polars as pl

# # Lazy read — doesn't load into memory immediately
# near_lazy = pl.scan_parquet("nearest_competitors.parquet")

# # Compute unique pairs in streaming mode
# unique = (
#     near_lazy
#     .with_columns([
#         pl.min_horizontal(["TDLinx", "competitor_TDLinx"]).alias("_a"),
#         pl.max_horizontal(["TDLinx", "competitor_TDLinx"]).alias("_b"),
#     ])
#     .unique(subset=["_a", "_b", "year"])
#     .select(["_a", "_b", "year", "distance_m"])
#     .rename({"_a": "store_a", "_b": "store_b"})
#     .collect(engine="streaming")
# )

# print(f"Unique pairs: {len(unique):,}")
# unique.write_parquet("unique_pairs.parquet")
```

```python
from scripts.time_dimension.time_dimension import chain_switch_matrix
chain_switch_matrix(panel).show()
```

```python
panel.filter(pl.col("Name").str.contains("Tengelmann")).describe()
```

```python

```
