# Operations

## Environment

Python ≥ 3.11. Tested with polars 1.x and 2.0.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`tkinter` (desktop GUI only) cannot be pip-installed: bundled on Windows,
`apt install python3-tk` on Debian.

Set the machine's paths in `run_pipeline.Config`: `panel`, `ghsl`, `work`,
`out`; and `SHAPEFILE_PATH` in `scripts/config.py`.

Optional, for ~1 m instead of ~3 m accuracy in the DHDN datum shift:

```bash
pyproj sync --file de_adv_BETA2007.tif
```

## OSRM server

Needed only for solve stages (catchment bases, grid and catchment maps).

**Build the graph** (once per network, from a Geofabrik `.osm.pbf`; the
`osrm-extract` binary and the server must be the same version):

```bash
osrm-extract -p /opt/osrm/profiles/car.lua germany-latest.osm.pbf
osrm-partition germany-latest.osrm
osrm-customize germany-latest.osrm
```

**Serve it:**

```bash
osrm-routed --algorithm mld --mmap --max-table-size 10000 germany-latest.osrm
```

- `--algorithm mld` must match the preprocessing above.
- `--mmap` pages the ~20–25 GB graph instead of loading it; fine with enough RAM
  and an SSD, slow otherwise (see *Performance*).
- `--max-table-size` must exceed the largest `/table` request; the pipeline's
  preflight measures what the server accepts.
- On the Windows workstation OSRM runs in WSL Debian; give WSL enough memory in
  `.wslconfig` (`memory=16GB`).

**Check it:**

```bash
curl "http://127.0.0.1:5000/route/v1/driving/9.99,53.55;10.02,53.57?overview=false"
```

The snap artefacts record the network they belong to (`snap_manifest.json`);
serving a different extract under the same `network` name produces wrong
snaps without an error.

## Performance

Solve time is almost entirely OSRM `/table` work. Diagnose while it runs:

```bash
htop        # busy process: osrm-routed or python
vmstat 5    # si/so = swapping, wa = waiting on disk
```

| Observation | Cause | Remedy |
|---|---|---|
| `osrm-routed` busy on all cores | CPU-bound routing | `--threads` = cores, `max_workers` to match, large `--max-table-size` |
| `osrm-routed` idle, high `wa` | `--mmap` reading the graph from disk | graph on an SSD; or a regional extract that fits in RAM, served without `--mmap` |
| `si`/`so` non-zero | swapping | fewer concurrent jobs; smaller extract |


Structural savings, in order of size:

1. **Solve each shop type nationally once.** Every Land run then reuses the
   national matrices without routing.
2. **Supply-side runs need no solve at all** — drop `pop`/`area` from `bases`.
3. **New vintages extend, not re-solve** — keep `stores.parquet` and the
   matrices; never `--force matrix` for new data.
4. **Refinement tail.** `Config.ks` up to 140 and `cert_target` 0.9 put the last
   rounds on the hardest nodes; capping `ks` at 90 cuts time at a small cost in
   certified share (reported per year).

Solves checkpoint continuously: stop and restart and they resume.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `libtk8.6.so: cannot open shared object file` | Tk missing | install `tk` |
| Figures unchanged after updating code (notebook) | kernel holds the old module | restart the kernel |
| `grid_view: ... raster CRS/transform and the cells do not agree` | `cells_raw.parquet` from another raster | delete it; it is rebuilt from the raster |
| `grid_maps: no cell of this run inside the ... window` | Land-only solve; the city is outside the Land | expected; national run for city maps |
| `regions: no datum transformation ... only a ballpark one` | shapefile CRS unrecognised | fix the `.prj` or `Config.shapes_crs`, then `--force regions` |
| `WARNING ... Kreis code(s) not in the region layer` | Kreis reform not in the crosswalk | add to `market_cube.KREIS_XWALK` |
| `matrix_frame: ... no longer in stores.parquet` | stores rebuilt with `--force stores` | `--force matrix` for that type |
| A run re-solves unexpectedly | artefacts in an older folder layout | `--status`, then `--migrate` / `--migrate-apply` |
| `verify_compose` exits 1, `different duration: 0` | equal-time ties resolved to other stores | harmless |

## Rebuilding the panel

The panel is built outside the one-button pipeline, by `scripts/tdlinx`
(`data_loading` → `cleaning` → `panel`), and written to `Config.panel`. Any
change to it is picked up automatically on the next run: assignments of the
affected years are recomputed and the cube is cleared and rebuilt. Geocoding
of unmatched addresses (geopy / Nominatim) is a manual step, never run by the
pipeline.
