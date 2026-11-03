# Configuration

A run is described by one TOML file in `runs/` (`scripts/runconfig.py`,
`RunConfig`). The desktop GUI writes it, the notebook panel builds it, the CLI
reads it, and a copy is saved next to the run's outputs, so any output can be
reproduced from its own directory.

## Run config (TOML)

```toml
[run]
name = "baseline"            # output folder: out/<grid>[_<land>]/<name>
note = "headline numbers"

[spec]
specs = ["super", "super_disc", "super_disc_drog"]
land = ""                    # "" = Germany; "08" = Baden-Württemberg
[spec.shop_types]            # market definitions: which formats compete
super = ["Supermarkt"]
super_disc = ["Supermarkt", "Discount Geschäft"]
super_disc_drog = ["Supermarkt", "Discount Geschäft", "Drogeriemarkt"]

[years]
mode = "range"
values = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

[report]
metrics = ["hhi", "share"]
bases = ["pop", "area", "vkfl", "umsatz"]
scope = "all"                # "all" | "food" (supply-side columns)
exclude_actors = []          # e.g. the unidentified labels, see below
levels = ["DE", "BL", "KREIS"]
actor_levels = ["Gruppe", "Chain"]
year_focus = 0               # year for single-year figures; 0 = latest

[output]
figures = ["map_hhi", "map_share", "map_leader", "heat_shares", "trend_hhi",
           "trend_shares", "dist_hhi", "scatter_basis", "bars_bl"]
formats = ["pdf", "png"]
export_csv = true
write_rasters = false        # nearest-store GeoTIFFs per spec x year

[solve]
stages = ["cube", "figures"]
force = []
osrm = "http://localhost:5000"
network = "germany-latest"
cache = ""                   # snap cache location; "" = snapped/cache
grid = "de_100m"
k = 20                       # starting candidate set size
cert_target = 0.9            # certified share of nodes to refine towards
iso_min = 20                 # isochrone radius, minutes
max_workers = 8              # parallel OSRM requests
cells_per_catchment = 0      # grid reduction; 0 = module default (8)
```

### Important Fields

| Field | Effect |
|---|---|
| `run.name` | Names the output folder. The GUI sets it to the file name on save. |
| `spec.shop_types` | Market definitions. Changing or adding one costs no routing. |
| `spec.land` | Report one Land; see *Land runs* in pipeline.md. |
| `report.bases` | `pop`/`area` need the OSRM solve; `vkfl`/`umsatz` do not. |
| `report.scope` | `food` reads the food columns for supply-side bases. |
| `report.exclude_actors` | Drops these actors from the market; outputs go to `__excl` / `excl/`. `runconfig.UNIDENTIFIED` lists the labels meaning "not identified". |
| `solve.cache` | Where the `/nearest` cache lives — shared by all runs; point it at a local SSD if the work dir is on a network share. |
| `solve.grid` | Namespace of the solve grid. Change only with a new grid. |
| `solve.k`, `cert_target` | Solve accuracy/cost trade-off; apply to new solves only. |

The pipeline's own `Config` (paths, CRSs, `ks`, cities for grid maps) is in
`run_pipeline.py`; `RunConfig.pipeline_config()` derives it for a run. Paths to
set per machine: `panel`, `ghsl`, `work`, `out`.

## Command line

```bash
python -m scripts.run_pipeline --config runs/baseline.toml
python -m scripts.run_pipeline --stages cube,figures --specs super --years 2024:2025
python -m scripts.run_pipeline --stages all --land 09 --bases pop,vkfl
python -m scripts.run_pipeline --config runs/baseline.toml --force assign
```

| Flag | Meaning |
|---|---|
| `--config FILE` | load a run config; other flags override it |
| `--stages` | `all` or a list of `stores,cells,matrix,assign,regions,cube,figures` |
| `--specs` | market definitions to run |
| `--years` | `2014:2025` or `2024,2025` |
| `--land` | AGS2 prefix, e.g. `09` |
| `--bases` | e.g. `pop,area,vkfl,umsatz` |
| `--scope` | `all` or `food` |
| `--exclude-actors` | operators to drop before shares are computed; `other` = the `UNIDENTIFIED` list |
| `--osrm` | OSRM base URL |
| `--force` | stages to recompute even if cached |
| `--status` | print what exists on disk and exit |
| `--diagnose` | print per-year panel composition and exit |
| `--save-config FILE` | write the effective configuration and exit |
| `--gui` | open the desktop GUI |
| `--migrate` / `--migrate-apply` | show / move artefacts from older layouts |

## Desktop GUI

```bash
python -m scripts.gui                     # new config
python -m scripts.gui runs/baseline.toml  # edit an existing one
```

Edit the fields, **Save…** under the run's name (the file name becomes the run
name and output folder), then **Run**. The GUI launches the pipeline as a
subprocess with `--config`, so it always runs the code on disk. It warns before
starting expensive stages that are not cached yet. Needs Tk.
 
## Notebook panel

```python
from scripts.isochrones.control_panel import panel
p = panel()
```

- **Build** tab: specs, years, Land, OSRM, run name, stages, force; *Status*,
  *Preview* (the equivalent CLI line), *Run*.
- **Explore** tab: slice the cube by spec, year, level, actor level, basis,
  scope; tables, figures, CSV.

The panel passes its whole state to the pipeline as a `RunConfig`, so what it
runs is what the printed command would run. After replacing any `scripts/`
file, restart the kernel — a running kernel keeps the old module. 

