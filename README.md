# edeka_project — market structure of German food retail

Market shares and concentration (HHI, CR1/CR3/CR5) of German food retailers,
2014–2025, for three market definitions, three geographic levels and four
bases of measurement, with drive-time catchments computed on the German road
network.

| Dimension | Values |
|---|---|
| Metric | HHI (plus CR1, CR3, CR5), market share |
| Years | 2014–2025 |
| Geography | Germany, Bundesland, Kreis |
| Basis | Umsatzpotential, VKFL (supply side); catchment population, catchment area (demand side) |
| Product scope | all, food only (supply-side bases) |
| Market definition (`spec`) | supermarkets; + discounters; + drugstores |
| Actor level | Gruppe (ownership / buying group), Chain |

## Quick start

```bash
pip install -r requirements.txt
# OSRM must be running (docs/operations.md) -- only for demand-side bases

python -m scripts.run_pipeline --status                     # what exists on disk
python -m scripts.run_pipeline --config runs/baseline.toml  # a saved run
python -m scripts.gui                                       # desktop GUI
```

In Jupyter:

```python
from scripts.isochrones.control_panel import panel
panel()
```

A run that only reports VKFL/Umsatzpotential never touches OSRM and finishes in
seconds. A run with catchment population or area needs the road solve once per
shop type; afterwards everything is reused (docs/pipeline.md).

## Documentation

| File | Contents |
|---|---|
| [methodology.md](methodology.md) | Market definitions, bases, catchments, HHI conventions, exactness guarantees |
| [pipeline.md](pipeline.md) | Stages, what is cached where, what a re-run costs, directory layout |
| [configuration.md](configuration.md) | Run configs (TOML), CLI flags, desktop GUI, notebook panel |
| [outputs.md](outputs.md) | Cube schema, CSV exports, figure catalogue, file naming |
| [operations.md](operations.md) | OSRM server, environment, performance, troubleshooting |
| [developer_guide.md](developer_guide.md) | How the code fits together, identifier spaces, artefact catalogue, log-line glossary, debugging playbook, extension recipes |

## Code map

```
scripts/
  run_pipeline.py        driver: stages, caching, CLI
  runconfig.py           RunConfig: one run as a TOML file (GUI, CLI, notebook)
  gui.py                 desktop GUI (tkinter), writes a TOML and launches the pipeline
  verify_compose.py      checks per-type composition against a direct solve
  config.py              legacy paths/constants used by the panel build
  isochrones/
    network_voronoi_2.py snapping, grid reduction, candidates, OSRM drive times,
                         certification, matrix extension, assignment
    compose.py           combines per-shop-type assignments into any market definition
    market_cube.py       shares, HHI, CRn over every reporting dimension
    control_panel.py     Jupyter front-end (ipywidgets)
    market_metrics.py    legacy, no longer imported by the pipeline
  tdlinx/
    data_loading.py      reads the TDLinx workbook (2023 from a cleaned export)
    cleaning.py          numeric parsing, Shop Type, Gruppe/Chain classification
    panel.py             builds the panel parquet the pipeline reads
  visualizations/
    figures.py           every figure; publication styling
runs/                    saved run configs (*.toml)
tests/                   pytest suite
```
# temp
# temp
# temp
