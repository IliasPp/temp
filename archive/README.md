# archive

Code no longer imported by the pipeline, kept for reference. Nothing here is
on the import path; imports inside these files point at their old locations.

| File | Was | Superseded by |
|---|---|---|
| `market_metrics.py` | `scripts/isochrones/market_metrics.py` | `isochrones/market_cube.py`, `visualizations/figures.py` |
| `tdlinx_metrics.py` | `scripts/tdlinx/metrics.py` | panel HHI/shares: `market_cube.py`; the `fe_*` panel regressions have no successor yet |
| `config_legacy.py` | `scripts/config.py` | `run_pipeline.Config`, `runconfig.py` (Land names), `figures.ACTOR_COLOURS` |
| `scripts_init_legacy.py` | `scripts/__init__.py` | — (commented-out imports of old modules) |
| `data_loading_load_geodata.py` | `load_geodata()` in `scripts/tdlinx/data_loading.py` | `run_pipeline.region_layers()` |
