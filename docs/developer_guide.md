# Developer guide: how the code works and how to debug it

For whoever has to read, fix or extend the pipeline. The other docs say *what*
the project computes (`methodology.md`) and *how to run it*
(`configuration.md`, `operations.md`). This one says how the code is put
together, what every file on disk is, what the log lines mean, and what to do
when something breaks.

---

## 1. The whole process in one picture

```
 TDLinx workbook ──scripts/tdlinx──► panel (parquet, one row per store-year)
                                         │
 GHS-POP raster ─────────┐               │
                         ▼               ▼
                 ┌─ stage_cells ◄── stage_stores ──► snapped/stores.parquet
                 │   7.27M cells → 5.0M solve nodes   (60.8k stores, /nearest once)
                 │   snapped/<grid>/cells_solve, mapping
                 │               │
 VG5000 Gemeinden ─► region_layers ─► stage_regions ─► node_regions (node, Kreis, pop, area)
                 │               │
                 │      stage_matrix  (one per SHOP TYPE; the only OSRM-heavy step)
                 │        solve/<grid>/<type>/matrix.parquet
                 │               │
                 │      stage_assign  (per SPEC x YEAR; no OSRM; minutes)
                 │        derived/<grid>/<spec>/assignments/…
                 │               │
                 └──────► stage_cube  (per spec x side x year; seconds; never cached)
                            outputs/catchments/<grid>/<run name>/cube/…
                                         │
                       read_cube (validated) ─► stage_figures, stage_export
```

1. **Drive times are solved once per shop type, never per market definition.**
   A spec (`super_disc`) is "the nearest of the nearest supermarket and the
   nearest discounter", which is exact (`compose.py`). So adding or changing a
   spec costs no routing.
2. **Every expensive artefact is reused unless its inputs changed**, and the
   check is explicit: a sidecar fingerprint or a stored store list, written
   only after the artefact itself.

---

## 2. Where the code is

| File | Responsibility | Read first |
|---|---|---|
| `scripts/run_pipeline.py` | Paths (`Config`), every stage, cache bookkeeping, `run()`, CLI | module docstring "Reading guide", then `run()` |
| `scripts/runconfig.py` | `RunConfig`: one run as a TOML file; validation; `pipeline_config()` | class fields |
| `scripts/isochrones/network_voronoi_2.py` | Solver library: snapping, grid reduction, candidates, OSRM `/table`, certification, matrix file I/O, assignment | module docstring list |
| `scripts/isochrones/compose.py` | Combining per-type assignments (`rank1`, `topk`) and `verify` | whole file (short) |
| `scripts/isochrones/market_cube.py` | Atoms, shares, HHI/CR, validation, panel diagnostics | "Data flow" in docstring |
| `scripts/visualizations/figures.py` | Every figure; colours; grid and catchment maps | `colours`, `grid_slice` |
| `scripts/isochrones/control_panel.py` | Jupyter front-end; builds a `RunConfig` and calls `run()` | `runconfig()` |
| `scripts/gui.py` | Desktop front-end; writes a TOML, launches the CLI as a subprocess | — |
| `scripts/tdlinx/cleaning.py` | Sheet cleaning, Shop Type, Gruppe/Chain, `gate_supplied` | `_clean_sheet` |
| `scripts/tdlinx/data_loading.py`, `panel.py` | Read the workbook; build the panel parquet | — |
| `scripts/verify_compose.py` | Checks composition against a direct solve on one Land | — |
| `scripts/tables.py` | LaTeX/CSV slide tables from a cube | — |
| `scripts/diagnose_trends.py` | Decomposes national trends; label-switch audit | — |
| `scripts/snapping_report.py` | How far stores and population were moved onto roads; maps, window views with OSM roads | module docstring |
| `scripts/config.py` | Input paths for the panel build and the region layer; no side effects | — |
| `archive/` | Code no longer imported (old metrics, legacy config); see its README | — |

Front-ends never compute anything themselves: the GUI writes a TOML and runs
the CLI; the notebook panel builds a `RunConfig` and calls `run()`. So a bug
reproduced in one front-end is reproducible from the command line with the same
config.

---

## 3. What `run()` does, step by step

The numbered blocks match the `# ---- n.` comments in `run()`.

1. **Resolve the selection.** `rc.pipeline_config()` builds the `Config`. The
   run name, and so the output folder, is `[run] name` from the TOML (see §10).
   `solve` is true if any solve stage is selected, or the report needs
   catchments (`pop`/`area` basis, grid or catchment maps, label rasters).
2. **Inputs.** Three views of the panel: `panel_raw` (everything, for
   snapping), `panel_all` (formats used by any spec), `panel` (restricted to
   the Land). Region layers load from `regions_v4.gpkg`. Unknown Kreis codes
   are warned about.
3. **Solve chain** (only if `solve`). `stage_stores` → `stage_cells` →
   `stage_regions` → for each needed shop type `stage_matrix`. A Land run
   switches to the national matrices if all exist (`scfg`, `spanel`).
4. **Per spec.** `stage_assign` (or, without solve, membership from the panel)
   → `stage_cube` → grid/catchment maps if requested.
5. **Report.** `read_cube` concatenates and validates the cube → figures → CSV
   (both only when `figures` is among the stages).

Every stage function takes plain arguments and can be called alone in a
notebook. `run()` returns a `state` dict (panel, stores, cells, c2, mapping,
node_reg, per-spec assignments, shares, hhi) for interactive inspection.

---

## 4. Identifiers and data contracts

Most hard bugs in this project were id mix-ups. The rules:

| Name | Meaning | Where |
|---|---|---|
| `store_id` | TDLinx as **Utf8**, everywhere after `stage_stores` | stores.parquet, matrices, assignments, atoms |
| `cell_id` in `cells_raw` | populated **raster cell**, 0..7.27M, with `row`/`col` | cells_raw.parquet |
| `cell_id` in `cells_solve` (`c2`) | **solve node**, 0..5.0M | cells_solve, matrix rows, assignments |
| `node_id` | same as the solve-node `cell_id`, renamed | node_regions, cube internals |
| `mapping` | `src` = raster cell_id → `dst` = node id | mapping.parquet |
| `store_idx` | **position** in a matrix's own store frame, not a store id | matrix.parquet |

Contracts that must hold, and what enforces them:

- **A matrix and its store frame are one object.** The frame's ids are saved in
  the matrix footer; `matrix_frame()` rebuilds the frame in saved order and
  appends new stores at the end. `load_matrix`, `_check_aligned` and
  `matrix_frame` raise on any mismatch rather than relabel catchments.
- **A matrix and the grid are one object.** Matrix rows are `c2` rows;
  `load_matrix(cells=c2)` compares the `cell_id` column and raises on mismatch.
  Never rebuild the grid under existing matrices (`--force cells` invalidates
  all of them).
- **stores.parquet is append-only.** Existing stores keep their row order;
  that is what makes `nv.extend` possible.
- **Membership is per store-year, from the panel.** `first_year`/`last_year`
  in stores.parquet are stale once it has been topped up and are not used;
  `stores_for_year()` rewrites them per year.
- **Labels (Gruppe, Chain) are read per year from the panel at cube time.**
  The assignment never sees labels, so relabelling never invalidates it.

---

## 5. Artefact catalogue

Paths relative to `data/catchments/` (inputs, shared by all runs) and
`outputs/catchments/<grid>[_<land>]/` (outputs).

| Artefact | Written by | Contents | Reused when | Safe to delete? |
|---|---|---|---|---|
| `snapped/stores.parquet` | `stage_stores` | store_id, snapped lon/lat/x/y, `*_orig`, `snap_m`, first/last_year (stale), latest Gruppe/Chain/Shop Type (display only) | always; new panel stores are appended | **No**: every matrix indexes it. Deleting = re-solve all |
| `snapped/rejected.txt` | `stage_stores` | ids that failed to snap (no road within 300 m / no coords) | always | yes (they are retried once) |
| `snapped/snap_manifest.json` | `stage_stores` | network, OSRM URL, drop distance | checked each run, warns | yes |
| `snapped/cache/snap_<n>_<hash>.npz` | `nv.snap` | /nearest results per point set | same point count + coordinate sums | yes (costs re-snapping) |
| `snapped/<grid>/cells_raw.parquet` + `.fp` | `load_cells_cached` | raster cells: cell_id, row, col, pop, x/y (3035), lon/lat | raster file unchanged (name, size, mtime) | yes (re-read from raster, minutes) |
| `snapped/<grid>/cells_solve.parquet` | `stage_cells` | solve nodes: cell_id (node), pop, x/y, lon/lat, n_src, x/y_orig | always (frozen) | **No**: matrices index it |
| `snapped/<grid>/mapping.parquet`, `map_adapt`, `map_dedupe` | `stage_cells` | raster cell → node (and the two steps) | always | **No** |
| `snapped/<grid>/node_regions.parquet` | `stage_regions` | node_id, kreis, pop, n_cells, area_km2 | always | yes (minutes) |
| `snapped/<grid>/regions_v4.gpkg` | `region_layers` | layers `kreis` (AGS5, GEN, LABEL), `bundesland` (AGS2, …), in raster CRS | always | yes; **must** delete when shapefiles change |
| `snapped/<grid>/grid_manifest.json` | `stage_cells` | network, grid, raster fingerprint, node count | checked each run, warns | yes |
| `solve/<grid>/<type>/matrix.parquet` | `stage_matrix` | see §9.2 | always; extended for new stores | **No**: hours of routing |
| `solve/<grid>/<type>/matrix.npz` | older versions | legacy matrix | converted once to parquet | after conversion |
| `solve/<grid>/<type>/ckpt/…` | `travel_times` | partial routing results | only by a resumed solve of the same candidate set | yes once matrix.parquet exists |
| `derived/<grid>/<spec>/assignments/assignments_<spec>_<y>.parquet` + `.fp` | `stage_assign` | per node: store_id, rank, duration_s, euclid_m, pop, year | sidecar matches (matrices + that year's membership) | yes (minutes) |
| `derived/<grid>/<spec>/catchments/…` | `stage_assign` | per store: catchment_pop, n_nodes, mean/max/pop-weighted drive time | with the assignment | yes |
| `derived/<grid>/<spec>/isochrone/…` | `stage_assign` | per node: up to `iso_k` stores within `iso_min` minutes | with the assignment | yes (nothing in the cube reads it) |
| `derived/…/assignments/_partial/` | `stage_assign` | in-progress merge | never (cleared at start) | yes |
| `<grid>/<run>/run_config.toml` | `run` | the exact config of this output | — | — |
| `<grid>/<run>/cube/<spec>[__excl]/` | `stage_cube` | `{shares,hhi}_{supply,demand}_<y>.parquet`, `manifest.json` | never; rebuilt every run | yes |
| `<grid>/<run>/cube/csv/` | `stage_export` | CSV views of the cube | never | yes |
| `<grid>/<run>/figures/[excl/]`, `grid/`, `catchments/` | figure stages | PDF/PNG, catchment GPKG | never | yes |
| `<grid>/<run>/tables/` | `scripts.tables` | slide tables (.tex + .csv) | never | yes |
| `<grid>/rasters/<spec>/` | `_write_raster` | nearest-store GeoTIFF + lookup | file exists | yes |

What is reused across runs: everything in `data/catchments/`. Runs differ only in
`outputs/…/<run name>/`.

---

## 6. What a change costs

| You change… | Recomputed | OSRM |
|---|---|---|
| bases, levels, actor levels, figures, scope, exclusions | cube, figures | none |
| the panel's labels (Gruppe/Chain rules, `gate_supplied`) | cube, figures (panel fingerprint clears the cube dirs) | none |
| a spec definition, or a new spec from existing types | that spec's assignments | none |
| years selected | assignments of new years | none |
| a store's format in some year | assignments of the affected years (if the store is new to that type's matrix: an extension, then all years of that type) | none, or minutes |
| new stores in the panel | `/nearest` for them; matrices extended; all years of affected types reassigned | minutes |
| a new shop type in a spec | one matrix | hours |
| `k`, `ks`, `cert_target` | nothing, until a matrix is built from scratch (`--force matrix`) | hours per type |
| raster, OSRM network, `grid`, `cells_per_catchment` | everything (new `grid` name, or `--force cells`) | days |

---

## 7. Reading the log

Run with `python -u` (or `PYTHONUNBUFFERED=1`) so lines appear as they happen;
otherwise only warnings show until the end.

| Line | Meaning | Normal | Investigate if |
|---|---|---|---|
| `panel: 571,002 rows \| 60,832 stores \| 2014-2025 \| types [...]` | panel read | — | store count differs from the other machine (§8, panel drift) |
| `WARNING 1 panel Kreis code(s) not in the region layer -- … 00000` | stores whose AGS5 the layer lacks | `00000` = missing Gemeindekennziffer; harmless if few | a real code appears: add it to `KREIS_XWALK` |
| `output  …/<grid>/<name>` | where this run writes | your project folder and run name | wrong project or name (§8) |
| `stores: cached 60,802 snapped (30 rejected)` | no new stores | — | `N store(s) new in the panel` with large N |
| `cells: 7,269,822 raster cells -> 5,007,599 solve nodes (68.9%)` | grid reused | these numbers | different numbers: different grid/raster |
| `regions: EPSG:31467 -> ESRI:54009 via 'DHDN to WGS 84 (…)'` | datum shift used (only when regions_v4 is built) | Helmert, ~3 m | "ballpark" error |
| `=== Supermarkt: 21,462 stores traded as this type in some year ===` | stores that belong in the matrix | — | — |
| `matrix[Supermarkt]: cached (21,448 stores; arrays load on demand)` | matrix reused, not loaded | matrix count ≈ "traded as" count minus rejected | matrix count far **above** it: wrong (union) matrix (§8) |
| `matrix[X]: N store(s) new since the solve -- extending` | extension starts | after a new vintage | every run (stores.parquet changing) |
| `extend: N new store(s) fall inside the candidate radius of M/5,007,599 node(s); routing P pair(s)` | size of the extension | P in the millions = tens of minutes | — |
| `preflight: /table accepts a 5000 x 5000 matrix` | server table limit | ≥ 1000 coords | `raise --max-table-size` |
| `preflight: fastest route-average 105 km/h … vs V_MAX_KMH=180` | sanity check on the speed bound | below 180 | above 180: certification unsound |
| `travel_times: 316/316 blocks to solve, ~62M matrix entries` + progress | routing | — | `block … failed` lines |
| `rank 1: 88.50% certified` | share of nodes whose winner is proven (after a solve or extension) | compare with `cert_target` (90 %) | far below target: see methodology, certification |
| `assign[super]: 12 year(s) unchanged, reused` | sidecars match | — | — |
| `assign[super_disc]: Supermarkt merged into 12 year(s)` | stale years recomputed, one type at a time | after matrix/membership change | — |
| `assign[super] 2025: … \| 15,406 stores in spec \| X% of population not certified` | per-year proof rate of the composed winner | a few % | double digits |
| `coverage: 0.000% of population unallocated (92 people)` | people with no store at all | ≈ 0 | anything visible |
| `super 2025: 15,406 stores \| DE HHI(pop, Gruppe) 2,774` | one line per cube year | — | jumps: run `diagnose_trends` |
| `validate: OK -- …` | shares sum to 1, ranges OK | always | an AssertionError |
| `cube[super]: 24 (side, year) file pair(s)` | 12 years × supply+demand | — | 0 → RuntimeError with reasons |
| `cube[X]: spec, panel or exclusions changed … -- clearing` | old cube dir removed | after any panel rebuild | — |
| `plot_choropleth: N region(s) above the 2,500 cap` | colour cap | — | — |
| `02000: 1 detached part(s) more than 10 km … left off the map` | Neuwerk excluded from map window | Hamburg | — |
| `DeprecationWarning: is_in with a collection of the same datatype` | polars API warning | harmless | — |

---

## 8. Debugging playbook

| Symptom | Cause | Check | Fix |
|---|---|---|---|
| Run starts snapping stores | panel has stores/formats not in stores.parquet, or `rejected.txt` missing | `stores: N store(s) new`; N small = expected once | if N ≈ all stores: id dtype mismatch, compare `stores.parquet` ids with panel `TDLinx` |
| Run starts routing (`travel_times`) unexpectedly | wrong project (check `output`/`stores` paths), a matrix missing, a Land run without national matrices, or an extension | `--status`; the `matrix[...]` lines | point at the right project; copy matrices; for extensions, let it run |
| `matrix[X]: cached (N stores)` with N ≫ "traded as this type" | the file is an old **per-spec union** matrix renamed into a type folder | count shop types among `saved_store_ids()` (§9.2) | rename the folder, re-solve that type |
| `BadZipFile` / Arrow error loading a matrix | truncated copy | file size vs. the share's copy | copy again (`rsync --checksum`) |
| `Unable to allocate … (5007599, 140)` | refinement holding k=90 and k=140 arrays at once | which step (`refine: … k=140`) | page file/swap; solve one type alone; or stop refinement earlier (`cert_target`, `Config.ks`) |
| RAM exhausted after `… merged into N year(s)` | (older code) full-width isochrone frame | — | current code caps `iso_k=20`; set `Config.iso_k=0` to skip |
| `FileNotFoundError: no cube for X` | every year of the cube failed | lines `cube[X] <year>: …` above it | fix the stated reason |
| `actor_map: Chain not in the panel` | panel built before the Chain column existed | `pl.read_parquet_schema(panel)` | rebuild or copy the current panel |
| `ColumnNotFoundError: "pop"` in `catchments` | old `compose.py` that drops extra columns | `compose._stack` carries `extra` | update compose.py |
| `grid_view: … cells lie in this window by their EPSG:3035 coordinates, but none by the raster grid` | `cells_raw.parquet` from another raster | — | delete `cells_raw.parquet` and `.fp` |
| `grid_maps: no cell of this run inside the … window` | Land-only solve; city outside it | — | expected; national run for city maps |
| `regions: no datum transformation … only a ballpark one` | shapefile CRS unrecognised | the `.prj` | fix `.prj` or `Config.shapes_crs`, delete regions_v4.gpkg, `--force regions` |
| `travel_times: N checkpoint part(s) … different candidate set` | stale `ckpt/` from an earlier solve | — | delete that `ckpt` folder |
| `load_matrix: cell frame does not match` | grid rebuilt after the matrix was solved | `grid_manifest.json` | restore the old grid files, or re-solve |
| `matrix_frame: N store(s) this matrix was solved on are no longer in stores.parquet` | stores.parquet rebuilt (`--force stores`) | — | restore it, or `--force matrix` |
| Notebook shows old results after a code update | kernel holds the old module | — | restart the kernel; `gui._cube = None` after CLI rebuilds |
| Outputs land in another run's folder | `[run] name` in the TOML is not what the file name suggests | `run_config.toml` in the output folder | edit `[run] name` |
| No CSVs written | stages did not include `figures` | `stages` line in the header | add `figures` |
| `UnicodeEncodeError: 'charmap'` when redirecting output | Windows cp1252 console encoding | — | `$env:PYTHONIOENCODING="utf-8"` |
| Long silence in the log | stdout buffered; or a large matrix load/write | Task Manager / `htop` | `python -u`; see operations.md, performance |
| A national series jumps or dips | labelling change vs. real event | `python -m scripts.diagnose_trends runs/baseline.toml` | fix the classification (e.g. `gate_supplied`), rebuild panel |
| `shares do not sum to 1` | mixed cube files from different definitions | `cube/<spec>/manifest.json` | delete the cube dir and rerun `--stages cube,figures` |
| `libtk8.6.so: cannot open shared object file` | Tk not installed (Linux) | — | `pacman -S tk` / `apt install python3-tk` |

**Panel drift between machines.** Matrices, assignments and fingerprints were
built against the panel on the share. A different or older panel on another
machine changes membership, which triggers reassignments or extensions. Keep
one panel and copy it, or rebuild it the same way on both machines.

---

## 9. Inspecting artefacts by hand

All snippets assume the project root as working directory. They were run
against pipeline output before being written down here.

### 9.1 Stores, panel and membership

```python
import polars as pl
import scripts.run_pipeline as P
from scripts.runconfig import RunConfig
cfg = RunConfig.load("runs/baseline.toml").pipeline_config()

st = pl.read_parquet(cfg.snapdir / "stores.parquet")
st.filter(pl.col("store_id") == "100127")                 # one store, snapped
st.sort("snap_m", descending=True).head(10)               # worst snaps
panel = P.load_panel(cfg, shop_types=None)
P.membership(panel, ["Supermarkt"]).filter(pl.col("store_id") == "100127")   # years as Supermarkt
```

### 9.2 A matrix

```python
import numpy as np, pyarrow.parquet as pq
import scripts.isochrones.network_voronoi_2 as nv
m = cfg.solvedir("Supermarkt") / "matrix.parquet"
ids = nv.saved_store_ids(m)                               # cheap: footer only
print(len(ids), pq.read_schema(m).metadata[b"k"])         # frame size, width
# which formats have the matrix's stores had? A type matrix shows its own type
# for (nearly) every store, plus a handful under a second type: stores that
# switched format in some year. Thousands under another type = a union matrix.
print(panel.filter(pl.col("TDLinx").cast(pl.Utf8).is_in(ids.tolist()))
           .group_by("Shop Type").agg(pl.col("TDLinx").n_unique()))
z = nv.load_matrix(m)                                     # full load: ~GBs
cert = nv.certified(z["candidates"], z["durations"])
print(f"certified {cert.mean():.1%}")
```

### 9.3 Where does a location go?

```python
from pyproj import Transformer
x, y = Transformer.from_crs(4326, 3035, always_xy=True).transform(8.4660, 49.4875)  # Mannheim Q-squares
cells = pl.read_parquet(cfg.griddir / "cells_raw.parquet")
c = cells.with_columns(((pl.col("x") - x)**2 + (pl.col("y") - y)**2).alias("d")).sort("d").head(1)
node = (pl.read_parquet(cfg.griddir / "mapping.parquet")
          .filter(pl.col("src") == c["cell_id"][0])["dst"][0])
a = pl.read_parquet(cfg.derivdir("super") / "assignments" / "assignments_super_2025.parquet")
print(a.filter(pl.col("cell_id") == node))                # winning store, drive time
```

### 9.4 Why was a year recomputed?

```python
spec, y = "super_disc", 2025
fp_file = cfg.derivdir(spec) / "assignments" / f"assignments_{spec}_{y}.fp"
print(fp_file.read_text())   # "<type>:<n>:<hash of frame>:<hash of members>|..."
```

Compare the `<n>` (frame size) with the `matrix[...]: cached (N stores)` line,
and the member hash between two runs: whichever part differs is the reason.

### 9.5 The cube

```python
shares, hhi = P.read_cube(cfg, ["super"])
hhi.filter(pl.col("level") == "DE", pl.col("basis") == "pop", pl.col("actor_level") == "Gruppe")
shares.filter(pl.col("year") == 2025, pl.col("level") == "KREIS", pl.col("region") == "08222",
              pl.col("basis") == "pop", pl.col("actor_level") == "Gruppe"
              ).sort("rank", nulls_last=True)       # unidentified row (rank null) last
```

---

## 10. Known quirks in the current code

These are documented behaviour, not bugs being fixed here. Each is also
commented at the spot in the code.

1. **Run name.** The output folder is `[run] name` from the TOML. The desktop GUI
   sets it to the file stem on save; a hand-copied TOML keeps the old name and
   writes into the other run's folder.
2. **CSV export needs the `figures` stage.** `--stages cube` writes the cube
   parquet only.
3. **`--status` checks existence only.** A matrix shown `1/1` may still be
   extended; assignments shown `12/12` may still be recomputed (stale sidecar);
   excluded-actor cubes (`__excl`) are not shown.
4. **The notebook panel reports one level and one actor level** (the Explore
   tab's selection) and caches the cube for its lifetime.
5. **Figures treat only the label `UNKNOWN` as unidentified.** Other fringe
   labels (`Sonstige`, `Other`, …) are fringe in the cube but drawn as ordinary
   actors in the heatmap and share trends.
6. **`read_cube`'s error message** suggests a command without `--config`; add the
   run's config yourself.
7. **A fresh matrix solve leaves `ckpt/` behind.** Safe to delete afterwards.
8. **The snap cache key is coarse** (point count and rounded coordinate sums).
   Not relied on for stores (topped up by id) or the frozen grid.
9. **Parameters only in `Config`, not in the TOML:** `ks`, `iso_k`, `drop_over_m`,
   `cities`, `catchment_kreise`, CRSs and paths. `k`, `ks`, `cert_target`,
   `cells_per_catchment` only matter when a matrix or the grid is built anew.
10. **The isochrone frame is written for the last selected year only.**
11. **`max(1000, preflight)`**: a server limit below 1000 is not adapted to; the
    calls then fail. Fix the server.
12. **Some error texts still say `matrix.npz`** (`_check_aligned`); the file is
    `matrix.parquet` now.
13. **A type matrix holds every store that ever traded as that type**, minus
    those rejected at snapping (21,462 traded as Supermarkt, 21,448 in the
    matrix).
14. **`gate_supplied` is not part of `clean()`**; `panel.py` must call it on the
    concatenated panel.
15. **`Vertriebslinie` is consulted only by the dm and Müller rules** in
    `assign_gruppe`.
16. **polars DeprecationWarnings** about `is_in` are harmless.

---

## 11. Extending the pipeline

| Task | Where | Cost on next run |
|---|---|---|
| New market definition from existing types | `[spec.shop_types]` and `[spec] specs` in the TOML | assignments (minutes) |
| New shop type (e.g. Getränkemarkt) | add it to a spec's shop types | one matrix (hours) |
| New panel vintage | rebuild the panel; run as usual | snap new stores; extend matrices; reassign |
| New supply-side basis | `market_cube.BASES`; labels in `runconfig.BASIS_LABEL` and `figures.BASIS_LABEL` | cube only |
| New figure | `figures.plot_…`; a branch in `run_pipeline.draw_figure`; `FIG_NEEDS`; `runconfig.FIGURES` | figures only |
| Kreis reform | `market_cube.KREIS_XWALK` | cube only |
| Another city grid map | `Config.cities` | figures only |
| Different refinement ladder for one type | `Config.ks`, then `--force matrix` for that type | that matrix |
| Check composition after changing `assign` logic | `python -m scripts.verify_compose --spec super_disc --land 01 --year 2025` | one extra Land matrix |
