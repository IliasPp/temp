# Pipeline

`scripts/run_pipeline.py` runs everything. Each stage writes an artefact and
reuses it whenever its inputs are unchanged, so a re-run costs only what
changed.

## Stages

| Stage | Does | Cost | Reused when |
|---|---|---|---|
| `stores` | snap every panel store to the road network | once; later only new stores | always (append-only) |
| `cells` | reduce the 100 m grid, snap it, deduplicate | once, hours | always (frozen) |
| `matrix` | per **shop type**: candidates, drive times, refinement | once per type, hours | always; new stores extend it |
| `assign` | per spec × year: nearest store, composed from the type matrices | minutes, no OSRM | the year's matrices and membership are unchanged |
| `regions` | (solve node, Kreis, pop, area) table | once | always |
| `cube` | shares, HHI, CRn | seconds | never cached — always rebuilt |
| `figures` | figures and CSV exports | seconds–minutes | never cached |

`--stages` names what you **want**; dependencies are resolved from disk or
computed. `--stages figures` on an empty directory therefore solves first.

Solve stages run only if the run needs them: a catchment basis (`pop`, `area`),
grid maps or catchment maps. A VKFL/Umsatzpotential-only run skips `stores`
through `regions` entirely.

## Re-run costs

| Change | Routing | Recomputed |
|---|---|---|
| Nothing | none | cube, figures |
| New reporting choice (bases, levels, figures, scope) | none | cube, figures |
| New or redefined market definition | none | assignments for it, cube |
| Panel re-cleaned (Gruppe rule, format fix) | none | assignments of changed years, cube |
| New panel vintage | `/nearest` for new stores; `/table` only for pairs inside candidate radii | extended matrices, affected years |
| New shop type | one solve | that type's matrix |
| Road network, raster or grid parameters | full | everything (bump `grid` or `--force`) |

## Reuse rules

- **Fingerprints are written after the artefact.** A run killed mid-way redoes
  the unfinished part and never reuses a half-written one.
- **Assignments** carry one sidecar per year (`assignments_<spec>_<year>.fp`)
  recording the matrices and that year's store membership.
- **Matrices** own their store frame: the first rows are the stores they were
  solved on, in saved order; stores new since then are appended and routed by
  `nv.extend`. Matrix files are written atomically (temp file, then rename).
- **Stores** are append-only: existing stores keep their row position, which is
  what makes extension possible. Their coordinates are the record; a re-geocode
  needs `--force stores`, which invalidates every matrix.
- **The grid** is built from the national store set and frozen. A reduction or
  candidate parameter change means a new `grid` name, since cells, mapping and
  matrices are one object split across files.
- **The cube** is rebuilt on every run and written per (spec, side, year), so a
  supply-only run never touches catchment results and a two-year run never
  touches the other ten. A change of spec definition, panel file or exclusions
  clears the spec's cube directory first: years not rebuilt go missing
  (visible) rather than staying under the old definition (silent).

`--force <stage,...>` recomputes named stages regardless.

## Land runs

`land = "08"` reports one Bundesland.

- If national matrices exist for every shop type needed, they are used: exact
  at the border, no routing, minutes.
- Otherwise the Land is solved alone on its part of the national grid. Stores
  across the border are dropped, so border Kreise lose competitors — a smoke
  test, not a published number. Solve all types nationally once and every Land
  run becomes free.
- A Land run has no Germany total: its cube carries only the `BL` and `KREIS`
  levels, and the Land's own `BL` row is the total. A `DE` in `levels` is read
  as that row; the trend and heatmap figures, `tables.py` and `diagnose_trends.py`
  use it in place of Germany.

## Directory layout

Inputs shared by every run (under `Config.work`, `data/catchments/`):

```
snapped/
  stores.parquet                 snapped stores, append-only
  rejected.txt                   stores with no road within 300 m
  snap_manifest.json             network / OSRM the snap belongs to
  cache/                         /nearest cache, keyed by coordinate hash
  <grid>/                        e.g. de_100m
    cells_raw.parquet (+ .fp)    raster cells
    cells_solve.parquet          solve nodes
    mapping.parquet              raster cell -> solve node
    map_adapt / map_dedupe       the two reduction steps
    node_regions.parquet         node x Kreis: pop, cells, area
    regions_v4.gpkg              Kreis / Land layers in the raster CRS
    grid_manifest.json
solve/<grid>[_<land>]/<shop_type>/
  matrix.parquet                     candidates + drive times + certification
  ckpt/                          resumable partial results
derived/<grid>[_<land>]/<spec>/
  assignments/assignments_<spec>_<year>.parquet (+ .fp)
  catchments/catchments_<spec>_<year>.parquet
  isochrone/isochrone_<spec>_<year>.parquet
```

Outputs per run (under `Config.out`, `outputs/catchments/`):

```
<grid>[_<land>]/
  rasters/<spec>/nearest_store_<spec>_<year>.tif (+ .lookup.parquet)   shared
  <run name>/
    run_config.toml              the exact configuration of this output
    cube/<spec>[__excl]/         shares_|hhi_ <supply|demand>_<year>.parquet
    cube/csv/                    CSV exports
    figures/[excl/]              figures; grid/ and catchments/ subfolders
```

The run name is `[run] name` inside the run's TOML (the desktop GUI sets it to
the file name when saving; a TOML copied by hand keeps the old name). Only
outputs are namespaced by it.
Snapping, solving and assignments depend on the network, grid and panel, never
on reporting choices, so every run shares them — and the snap cache in
particular is shared: it maps coordinates to road nodes, nothing more.

## Checks

```bash
python -m scripts.run_pipeline --status          # artefacts on disk, per stage
python -m scripts.run_pipeline --diagnose        # panel composition per year
python -m scripts.verify_compose --spec super_disc --land 01 --year 2025
pytest tests/
```

Each run prints, and it is worth reading:

- `coverage:` share of population assigned to no store;
- `assign[...] ... not certified this year`: population whose winner is not proven;
- `WARNING ... Kreis code(s) not in the region layer`: a missing crosswalk entry;
- `regions: EPSG:31467 -> ... via '...'`: the datum transformation applied.
