# Outputs

All results of a run are under `out/<grid>[_<land>]/<run name>/`, next to the
`run_config.toml` that produced them.

## The cube

`cube/<spec>/` holds one pair of Parquet files per (side, year):

```
shares_supply_2025.parquet   hhi_supply_2025.parquet     VKFL, Umsatzpotential
shares_demand_2025.parquet   hhi_demand_2025.parquet     catchment pop, area
manifest.json                what the files were built from
```

Read it all at once:

```python
import scripts.run_pipeline as P
from scripts.runconfig import RunConfig
cfg = RunConfig.load("runs/baseline.toml").pipeline_config()
shares, hhi = P.read_cube(cfg, ["super", "super_disc"])
```

### shares

One row per (spec, year, level, region, basis, actor level, actor).

| Column | Meaning |
|---|---|
| `spec` | market definition |
| `year` | panel year |
| `level` | `DE`, `BL`, `KREIS` (a Land run has no `DE`; its `BL` row is the total) |
| `region` | `DE`, AGS2, or AGS5 |
| `basis` | `pop`, `area`, `vkfl`, `vkfl_food`, `umsatz`, `umsatz_food` |
| `actor_level` | `Gruppe` or `Chain` |
| `actor` | operator; `UNKNOWN` = unidentified stores, pooled |
| `rank` | 1 = largest named actor in the region; empty for the unidentified row |
| `value` | the actor's total on this basis (people, km², m², Mio EUR) |
| `share` | `value` / regional total, 0-1 |
| `n_stores` | the actor's stores contributing |

Shares sum to 1 per (spec, year, level, region, basis, actor level).

### hhi

One row per (spec, year, level, region, basis, actor level).

| Column | Meaning |
|---|---|
| `hhi` | 0-10,000; unidentified stores counted as separate firms - the headline |
| `hhi_pooled` | unidentified stores as one actor; always >= `hhi` |
| `cr1`, `cr3`, `cr5` | summed shares of the 1/3/5 largest named actors |
| `n_actors` | named actors present |
| `fringe_share` | share of unidentified stores |
| `n_stores` | stores contributing |
| `total` | regional total on this basis |

## CSV exports

`cube/csv/` - the selected metrics, levels, bases, actor levels and years, with
a `region_name` column:

```
hhi_<level>_<scope>[_excl].csv      e.g. hhi_kreis_all.csv
shares_<level>_<scope>[_excl].csv
```

## Figures

`figures/` (`figures/excl/` when actors are excluded). Every figure is written
in each of the run's `formats`. Names:

```
<figure>_<spec>_<actor level>_<basis>_<year>    e.g. map_hhi_super_Gruppe_pop_2025
trend_hhi_<actor level>_<basis>_<year>          one panel for all specs
```

| Figure | Shows |
|---|---|
| `map_hhi` | Kreis choropleth of HHI |
| `map_share` | Kreis choropleth of the leader's share |
| `map_leader` | which operator leads each Kreis |
| `heat_shares` | national shares, actor × year |
| `trend_hhi` | national HHI over time, one line per spec |
| `trend_shares` | national shares of the largest operators over time |
| `dist_hhi` | distribution of Kreis HHIs per year: boxes count every Kreis once; diamonds are the population-weighted median |
| `scatter_basis` | Kreis HHI, demand-side vs supply-side basis |
| `bars_bl` | HHI by Bundesland, first vs last year |
| `grid_maps` | 100 m cells coloured by the operator of their nearest store |
| `map_catchments` | store catchments as polygons for selected Kreise |

Operators keep one colour across all figures (`figures.ACTOR_COLOURS`);
unidentified is dark grey, cells reaching no store and residual "other" are
light grey.

**Grid maps** (`figures/grid/`), for the cities in `Config.cities` (Mannheim,
Hamburg):

```
grid_<city>_city_<spec>_<year>    whole city, 5 km scale bar
grid_<city>_zoom_<spec>_<year>    3 km around the centre, cell borders visible
```

**Catchment maps** (`figures/catchments/`): `catchments_<kreis name>_<spec>_<year>`
plus the polygons as `catchments_<AGS5>_<spec>_<year>.gpkg` for GIS.

Maps are drawn in the population raster's grid and CRS, so every square is a
real 100 m cell. Detached parts of a Kreis more than 10 km from its main area
(Neuwerk for Hamburg) are left off map windows; they still count in every
number.

## Rasters

With `write_rasters = true`, `out/<grid>[_<land>]/rasters/<spec>/` holds
`nearest_store_<spec>_<year>.tif` - each 100 m cell labelled with the code of
its nearest store (0 = none) - and `….lookup.parquet` mapping code → store_id.
They depend only on spec and year, so they are shared by all runs.

## Intermediate files worth knowing

Under `data/catchments/derived/<grid>/<spec>/`:

| File | Content |
|---|---|
| `assignments/assignments_<spec>_<year>.parquet` | solve node → nearest store, drive time |
| `catchments/catchments_<spec>_<year>.parquet` | per store: population, nodes, drive-time summary |
| `isochrone/isochrone_<spec>_<year>.parquet` | per node: the k nearest stores within `iso_min` minutes |

