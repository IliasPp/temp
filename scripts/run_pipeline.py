"""
Germany-wide retail market structure pipeline -- one button.

    python -m scripts.run_pipeline --stages all
    python -m scripts.run_pipeline --stages cube,figures          # cheap re-run
    python -m scripts.run_pipeline --stages all --land 09         # Bayern
    python -m scripts.run_pipeline --status                       # what is on disk

Stages, in dependency order. Each writes its artefact and reuses it when its
inputs have not changed, so a re-run costs only what changed:

    stores   snap every panel store to the road network   [once; new stores only]
    cells    reduce the 100 m population grid and snap it [expensive, once]
    matrix   per SHOP TYPE: candidates + OSRM drive times [expensive, once;
                                                           new stores extend it]
    assign   per spec x year: composed nearest store      [minutes, no OSRM]
    regions  collapse the grid to (solve node, Kreis)     [once]
    cube     shares + HHI, one file per spec x side x year [seconds, always rebuilt]
    figures  publication PDFs/PNGs                        [seconds]

What is reused, and when
------------------------
The road work is done once per shop type; a spec is an argmin over the per-type
solves (scripts.isochrones.compose), so defining or changing a spec costs no
routing. A new panel vintage costs /nearest for its new stores and /table only
for the (node, new store) pairs inside each node's candidate radius
(`nv.extend`) -- exact, and minutes rather than hours. Closed stores and format
switches cost nothing: membership is read per store-year from the panel.

Every derived artefact carries a sidecar fingerprint of its inputs, written
*after* the artefact: a run killed half way re-does the unfinished part and
never reuses a stale one. The cube is not cached at all -- it costs seconds, and
a cache that can be stale is worse than none.

The grid is always the national one, built from the national store set. A Land
run is a view: it reuses the national matrices when they exist (exact at the
border, no routing) and falls back to a Land-only solve -- a smoke test with
biased border Kreise -- only when they do not.

Bumping any reduction or candidate parameter means bumping `grid` or deleting the
matrices: cells, mapping and matrix are one object split across three files.

Reading guide
-------------
The file is laid out in pipeline order; read it top to bottom.

    Config, manifest, fingerprints      paths, parameters, cache bookkeeping
    load_panel, region_layers           inputs (panel parquet, Kreis/Land polygons)
    stage_stores, membership,           STORES: snap once, append-only; who trades
      matrix_frame, stores_for_year       as which format in which year
    stage_cells                         GRID: raster -> reduced solve nodes
    stage_matrix (+ MatrixHandle)       MATRIX: per shop type, OSRM drive times
    stage_assign (+ LazyYears)          ASSIGN: per spec x year, nearest store
    stage_regions                       (node, Kreis) population/area table
    stage_cube, read_cube               CUBE: shares + HHI files
    draw_figure .. stage_grid_maps      FIGURES and CSV export
    migrate, artefacts, run, main       driver and CLI

`run()` is the only function that strings the stages together; every stage can
also be called on its own from a notebook with the same arguments run() uses.

Identifier spaces -- the source of most confusing bugs
------------------------------------------------------
    store_id   TDLinx as Utf8, always (`_norm_ids`). Integer TDLinx never
               survives past load_panel/stage_stores.
    cell_id    in `cells` (cells_raw): 0..n-1 over populated raster cells, with
               row/col into the raster. In `c2` (cells_solve): the SOLVE NODE id,
               0..n_nodes-1. Same column name, different id space; `mapping`
               (src = raster cell_id, dst = node id) is the only bridge.
    node_id    the c2 cell_id under another name, used in node_regions and the
               cube. An assignment's `cell_id` column is a node id.
    store_idx  inside a matrix: a POSITION in that matrix's own store frame
               (`matrix_frame`), not a store_id. Only meaningful next to the
               frame, which is why the frame's ids are saved in the matrix file.

docs/developer_guide.md has the artefact catalogue, the log-line glossary and a
symptom -> cause debugging table.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

import scripts.isochrones.network_voronoi_2 as nv
import scripts.isochrones.market_cube as C
import scripts.isochrones.compose as CO
import scripts.visualizations.figures as F
from scripts.config import SHAPEFILE_PATH
from scripts.runconfig import LAENDER, UNIDENTIFIED, RunConfig

# ==========================================================================
# config -- the only place paths and parameters live
# ==========================================================================

#: The project folder this file lives in (scripts/..). Data and outputs are
#: resolved against it, so a copied or cloned project -- edeka_project_4, a
#: checkout on another machine -- uses its own data, never a sibling's, and
#: replacing this file never re-points a run at another project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _panel_default() -> Path:
    """The panel lives next to the projects on the share (germart-papaio/panel),
    or inside the project on a machine that carries its own copy."""
    for p in (PROJECT_ROOT / "panel", PROJECT_ROOT.parent / "panel"):
        if p.exists():
            return p
    return PROJECT_ROOT.parent / "panel"


#: spec -> the shop types that compete inside it. Defined once, in runconfig,
#: so the GUI, the config file and the pipeline cannot disagree about it. A run
#: may override it: `RunConfig.shop_types` travels with the config file.
SPECS: dict[str, list[str]] = {k: list(v) for k, v in RunConfig.__dataclass_fields__
                               ["shop_types"].default_factory().items()}


#: filesystem-safe name for a shop type ("Discount Geschäft" -> discount_geschaeft)
_UML = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
                      "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})


def slug(name: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in name.translate(_UML).lower())
    return "_".join(x for x in out.split("_") if x)


@dataclass(frozen=True)
class Config:
    # Pipeline parameters. A run normally builds this from its TOML via
    # RunConfig.pipeline_config(), which overrides only the fields the TOML
    # carries (name, years, land, osrm, grid, network, k, cert_target, iso_min,
    # max_workers, cells_per_catchment, write_rasters, cache). Everything else
    # here -- paths, ks, iso_k, drop_over_m, cities, CRSs -- is set only in this
    # file: change it here, and remember that ks/k/drop_over_m/cells_per_catchment
    # only affect artefacts built AFTER the change (cached ones are not rebuilt).
    panel: Path = field(default_factory=_panel_default)
    ghsl: Path = PROJECT_ROOT / "data" / "ghsl_raster_files" / "ghs_pop_germany_clipped.tif"
    work: Path = PROJECT_ROOT / "data" / "catchments"
    out: Path = PROJECT_ROOT / "outputs" / "catchments"
    osrm: str = "http://localhost:5000"
    #: Identifier of the OSRM build the snaps belong to. Snapping maps a
    #: coordinate to a position on *this* road network, so every snapped
    #: coordinate and every cached /nearest answer is valid only for it. Bump
    #: this when you rebuild the extract from a newer Geofabrik PBF; the old
    #: snaps then sit untouched under the old name instead of being silently
    #: mixed with the new network's.
    network: str = "germany-latest"
    #: CRS of the population raster, used only when the file itself carries none.
    #: The GHS-POP R2023A 100 m tiles are World_Mollweide (ESRI:54009), and a clip
    #: of them keeps those coordinates even when the clip drops the metadata --
    #: which is the usual reason this is needed. Set it to EPSG:3035 only if the
    #: clip was actually reprojected. `gdalinfo` settles it: Mollweide puts
    #: Germany near x 0.3-1.0M / y 5.8-6.6M, EPSG:3035 near x 4.0-4.7M / y 2.7-3.6M.
    raster_crs: str | None = "ESRI:54009"
    #: CRS of the admin shapefiles (VG5000, DHDN / Gauss-Krüger zone 3), used when
    #: the .prj is missing or does not resolve to an EPSG code. See to_raster_crs.
    shapes_crs: str = "EPSG:31467"
    #: Snap cache. Defaults next to the other network-level artefacts; set it to
    #: put it anywhere. It is append-only part files (see snap_cache.SnapCache),
    #: so the share is a fine place for it -- writes are small and nothing is
    #: ever replaced. Pointing it at a local disk only buys throughput.
    cache: Path | None = None

    years: tuple[int, ...] = tuple(range(2014, 2026))
    #: Years for which the *isochrone* frame is also written. The catchment
    #: assignment keeps one row per solve node (its nearest store); the isochrone
    #: frame keeps every store reachable within `iso_min`, so it is one row per
    #: (node, reachable store) -- order 10x larger, and only the local-market
    #: robustness check reads it. It costs no OSRM calls (it re-reads the stored
    #: matrix), so restricting it is about disk and RAM, not time: add a year by
    #: re-running `assign` for it.
    iso_years: tuple[int, ...] = (2025,)
    grid: str = "de_100m"                     # namespace of the shared solve grid
    land: str | None = None                   # AGS2 prefix, None = all of Germany

    k: int = 20
    ks: tuple[int, ...] = (40, 60, 90, 140)
    cert_target: float = 0.90
    cert_rank: int = 1
    iso_min: int = 20
    #: ranks kept in the isochrone frame. The matrix is as wide as refinement
    #: made it (140), and asking for every rank builds nodes x 140 rows per shop
    #: type -- tens of GB for a frame nothing in the cube reads. 20 nearest
    #: stores within iso_min minutes answers what the frame is for. 0 skips it.
    iso_k: int = 20
    drop_over_m: float = 300.0
    cells_per_catchment: int | None = None    # None = module default
    max_workers: int = 8

    kreis_id: str = "AGS5"
    kreis_name: str = "GEN"
    #: this run's name -- the stem of its config file; namespaces `outdir` only
    name: str = "baseline"
    #: the panel's own chain/banner column, used exactly like Gruppe
    chain_col: str = "Chain"
    cities: dict[str, str] = field(default_factory=lambda: {
        "Mannheim": "08222", "Hamburg": "02000"})
    #: Kreise to draw catchment polygons for; empty = the `cities` above
    catchment_kreise: tuple[str, ...] = ()

    def __post_init__(self):
        """Expand `~` in every path field.

        `Path("~/Projects")` is a *relative* path whose first component is the
        literal character `~`; nothing expands it implicitly, so it silently
        creates a directory called `~` under wherever the process happens to be
        running. Expanding once here means a `~` in a hand-edited path or in
        `[solve] cache` from a config file does the obvious thing.
        """
        for f in ("panel", "ghsl", "work", "out", "cache"):
            v = getattr(self, f)
            if v is not None:
                object.__setattr__(self, f, Path(v).expanduser())

    # (a normal dataclass field despite sitting below __post_init__)
    #: write a per-spec, per-year GeoTIFF of the winning store
    write_rasters: bool = False

    # --- derived ---------------------------------------------------------
    @property
    def tag(self) -> str:
        """Namespace for anything that depends on the grid or the Land."""
        return self.grid + (f"_{self.land}" if self.land else "")

    @property
    def snapdir(self) -> Path:
        """Snapped stores and the /nearest cache: network-level.

        Deliberately *not* namespaced by grid or Land. A store snaps to the same
        road node whichever raster or Bundesland asked for it, so a new grid must
        not drag 45,000 stores through OSRM again.
        """
        return self.work / "snapped"

    @property
    def griddir(self) -> Path:
        """The reduced grid, per grid only.

        A Land run is a *view* on it -- the national nodes filtered to that
        Land -- not a second grid. Clipping the raster instead would shift the
        cell ids and force the whole reduce-and-snap again for a subset of cells
        that were already snapped.
        """
        return self.snapdir / self.grid

    @property
    def cachedir(self) -> Path:
        return Path(self.cache) if self.cache is not None else self.snapdir / "cache"

    def solvedir(self, shop_type: str) -> Path:
        """One travel-time matrix per shop type, per grid: it indexes the grid's
        nodes. Specs are composed from these."""
        return self.work / "solve" / self.tag / slug(shop_type)

    def derivdir(self, spec: str) -> Path:
        """Assignments and their by-products: derived from the matrices in
        minutes and with no OSRM, so cached but not precious."""
        return self.work / "derived" / self.tag / spec

    @property
    def outdir(self) -> Path:
        """Where this run's results go: out/<grid>[_<land>]/<run name>.

        The run name is the stem of its config file (RunConfig.name), so two
        market definitions -- say baseline and identified_only -- keep their
        cube, figures and CSVs apart instead of overwriting each other. Inputs
        are not namespaced this way on purpose: snapped/, solve/ and derived/
        depend on the network, the grid and the panel, never on a run's
        reporting choices, and every run shares them.
        """
        return self.out / self.tag / slug(self.name)

    def rasterdir(self, spec: str) -> Path:
        """Label rasters are a property of (spec, year), not of a run: they sit
        beside the runs, not inside one, and are written once."""
        return self.out / self.tag / "rasters" / spec

    def raster(self) -> Path:
        """The population raster. Always the national one: a Land is a filter on
        the national grid, never a clipped raster (that would shift every cell id)."""
        return self.ghsl


CFG = Config()
UNION = sorted({t for v in SPECS.values() for t in v})


def manifest(path: Path, data: dict, *, what: str) -> dict:
    """Write a fingerprint on first use, compare on every later use.

    It WARNS, it never stops the run: a "WARNING the solve grid was built with
    different inputs" line means the cached artefact is being reused anyway.
    Read the printed diff and decide; the file is then updated to the new values,
    so the warning appears only once.

    Snapped coordinates are only reusable while the thing they were snapped to is
    unchanged. Nothing downstream can detect a swapped raster or a rebuilt OSRM
    extract -- the numbers just quietly become wrong -- so the inputs are recorded
    next to the artefact and checked rather than assumed.
    """
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return data
    old = json.loads(path.read_text(encoding="utf-8"))
    diff = {k: (old.get(k), v) for k, v in data.items()
            if k in old and str(old.get(k)) != str(v)}
    if diff:
        print(f"WARNING {what} was built with different inputs:")
        for k, (a, b) in diff.items():
            print(f"    {k}: was {a!r}, now {b!r}")
        print(f"    -> {path}\n    Reusing it mixes two builds. Delete the artefacts "
              f"(or bump Config.network / Config.grid) if this change is real.")
    old.update(data)
    path.write_text(json.dumps(old, indent=2, default=str), encoding="utf-8")
    return old


def _h(xs) -> str:
    """Short, order-sensitive hash of a sequence of strings."""
    return hashlib.sha1("|".join(map(str, xs)).encode()).hexdigest()[:16]


def store_fingerprint(stores: pl.DataFrame, extra: str = "") -> str:
    """Identity of a store frame: count plus a hash of the ids, in order.

    Used in each year's assignment sidecar (`assignments_<spec>_<year>.fp`):
    when a matrix is extended its frame grows, this changes, and every year of
    every spec using that shop type is recomputed on the next run.

    `cand`/`dur` index this frame positionally, so a frame that differs by one
    inserted row is a different object entirely.
    """
    return f"{stores.height}:{_h(stores['store_id'].cast(pl.Utf8).to_list() + [extra])}"


def _fresh(path: Path, fp: str) -> bool:
    """True if the sidecar exists and records `fp`. Read-only."""
    return path.exists() and path.read_text(encoding="utf-8").strip() == fp


def _commit(path: Path, fp: str) -> None:
    """Record `fp` next to an artefact. Call only AFTER the artefact is written:
    a sidecar written first would certify a half-finished rewrite as current."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fp, encoding="utf-8")


def _fingerprint(p: Path) -> str:
    """Cheap identity of a file (name, size, mtime) or of a directory of files (a
    hash over every file in it -- a directory's own mtime does not change when a
    file inside is rewritten). A content hash of a 4 GB raster on an SMB share
    costs more than it is worth."""
    p = Path(p)
    try:
        if not p.is_dir():
            st = p.stat()
            return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
        return _h(f"{f.relative_to(p)}:{f.stat().st_size}:{int(f.stat().st_mtime)}"
                  for f in sorted(p.rglob("*")) if f.is_file())
    except OSError:
        return f"{p.name}:missing"


def cache_note(cfg: Config = CFG) -> str:
    """Where the snap cache is, and how much is in it."""
    p = cfg.cachedir
    parts = len(list(p.glob("snap_*.npz"))) if p.exists() else 0
    return f"snap cache: {p}" + (f" ({parts} file(s))" if parts else "")


def _dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


# ==========================================================================
# inputs
# ==========================================================================

def load_panel(cfg: Config = CFG, *, shop_types=UNION) -> pl.DataFrame:
    """The TDLinx panel for the run's years; `shop_types=None` keeps every format."""
    p = pl.read_parquet(cfg.panel).filter(pl.col("year").is_in(list(cfg.years)))
    if shop_types is not None:
        p = p.filter(pl.col("Shop Type").is_in(list(shop_types)))
    # one row per store-year: the cube's per-store first()/last() are only
    # deterministic under it (cleaning.drop_duplicates guarantees it per sheet)
    if (dup := p.height - p.select("TDLinx", "year").n_unique()):
        raise ValueError(f"panel: {dup:,} duplicate (TDLinx, year) rows in {cfg.panel}")
    print(f"panel: {p.height:,} rows | {p['TDLinx'].n_unique():,} stores | "
          f"{p['year'].min()}-{p['year'].max()} | types {sorted(p['Shop Type'].unique())}")
    return p


def _code(s, width: int):
    """Digits of a region code at `width`: pad short codes, truncate long ones.
    The same layer can carry AGS '9162' or ARS '091620000000'."""
    d = s.astype(str).str.replace(r"\D", "", regex=True)
    return d.where(d.str.len() > width, d.str.zfill(width)).str[:width]


def to_raster_crs(gdf, cfg: Config = CFG, *, what: str = "layer"):
    """Reproject a layer into the population raster's CRS, datum shift checked.

    The admin shapefiles are DHDN (EPSG:31467), the raster WGS84-based Mollweide.
    DHDN -> WGS84 needs a datum shift of ~100-150 m, and PROJ silently falls back
    to a "ballpark" transformation that skips it when it cannot identify the
    source datum -- typically an ESRI-style .prj that does not resolve to an EPSG
    code. At 100 m cells that moves every Kreis border by about a cell, with no
    error anywhere. So the source CRS is pinned (Config.shapes_crs when the file's
    is missing or unrecognised), the transformation is chosen explicitly, a
    ballpark one is refused, and the one applied is printed.

    Without the BeTA2007 grid the best available shift is a 7-parameter Helmert,
    accurate to ~2-3 m: negligible here. With the grid it is ~1 m:
        pyproj sync --file de_adv_BETA2007.tif
    """
    import geopandas as gpd
    import shapely
    from pyproj.transformer import TransformerGroup
    if gdf.crs is None or gdf.crs.to_epsg() is None:
        print(f"regions: {what} CRS "
              f"{'missing' if gdf.crs is None else repr(gdf.crs.name) + ' not an EPSG code'}"
              f" -- using Config.shapes_crs={cfg.shapes_crs}")
        gdf = gdf.set_crs(cfg.shapes_crs, allow_override=True)
    dst = grid_crs(cfg)
    tg = TransformerGroup(gdf.crs, dst, always_xy=True)
    if not tg.transformers or "ballpark" in tg.transformers[0].description.lower():
        raise ValueError(
            f"regions: no datum transformation from {gdf.crs.to_string()} to the raster "
            f"CRS -- only a ballpark one, ~100 m off. Check the .prj of {what} or "
            f"Config.shapes_crs, and that PROJ's database is intact.")
    t = tg.transformers[0]
    print(f"regions: {gdf.crs.to_string()} -> {dst.to_string()} via {t.description!r} "
          f"(accuracy ~{t.accuracy:g} m)")
    if not tg.best_available:
        print("regions: BeTA2007 grid not installed; the Helmert shift above is "
              "~2-3 m, fine at 100 m cells ('pyproj sync --file de_adv_BETA2007.tif' "
              "for ~1 m)")
    geom = shapely.transform(np.asarray(gdf.geometry),
                             lambda xy: np.column_stack(t.transform(xy[:, 0], xy[:, 1])))
    return gdf.set_geometry(gpd.GeoSeries(geom, index=gdf.index, crs=dst))


def region_layers(cfg: Config = CFG, panel: pl.DataFrame | None = None):
    """(kreis_gdf, bl_gdf) with AGS5/AGS2 keys, names and unique labels.

    Geometry is the Gemeinde layer dissolved on its code, so it matches the
    raster labelling exactly. Names never come from that dissolve -- it keeps an
    arbitrary Gemeinde's name, so a Landkreis would be labelled after one of its
    villages. Kreis names come from the VG5000 Kreis layer shipped next to the
    Gemeinde layer (VG5000_KRS) when it is there, otherwise from the panel's own
    `Kreis` column (the most frequent spelling per AGS5; Kreise without a store
    keep their code). Land names come from runconfig.LAENDER. Kreis names
    repeat -- München is 09162 and 09184 -- so AGS5 is the key and LABEL exists
    for titles and filenames.

    Both layers are reprojected ONCE, here, into the population raster's CRS
    (`to_raster_crs`, datum-checked), and cached in it: every later `to_crs` --
    rasterising the Kreise, map windows, boundaries -- is then a no-op.

    Cached: delete regions_v4.gpkg when the shapefiles are updated.
    """
    import geopandas as gpd
    cache = cfg.griddir / "regions_v4.gpkg"
    if cache.exists():
        kre = gpd.read_file(cache, layer="kreis")
        bl = gpd.read_file(cache, layer="bundesland")
        print(f"regions: cached {len(kre)} Kreise | {len(bl)} Bundesländer")
        return kre, bl

    gem_path = Path(SHAPEFILE_PATH)
    gem = to_raster_crs(gpd.read_file(gem_path), cfg, what=gem_path.name)
    gem["AGS5"] = _code(gem["ARS"], 12).str[:5]
    kre = gem.dissolve(by="AGS5")[["geometry"]].reset_index()
    bl = (gem.assign(AGS2=gem["AGS5"].str[:2]).dissolve(by="AGS2")[["geometry"]]
          .reset_index())

    nm = cfg.kreis_name
    krs = gem_path.with_name(gem_path.name.replace("_GEM", "_KRS"))
    names = {}
    if krs.exists():
        k = gpd.read_file(krs, ignore_geometry=True)
        col = next((c for c in ("ARS", "AGS", "RS") if c in k.columns), None)
        if col and nm in k.columns:
            names = dict(zip(_code(k[col], 5), k[nm].astype(str)))
    if not names and panel is not None and "Kreis" in panel.columns:
        n = (panel.select(C.ags5("AGS5"), pl.col("Kreis").cast(pl.Utf8))
             .drop_nulls().group_by("kreis", "Kreis").len()
             .sort(["kreis", "len", "Kreis"], descending=[False, True, False])
             .unique("kreis", keep="first", maintain_order=True))
        names = dict(zip(n["kreis"].to_list(), n["Kreis"].to_list()))
        print(f"regions: Kreis names from the panel's Kreis column ({len(names)} Kreise)")
    if not names:
        print(f"WARNING regions: no Kreis names ({krs.name} missing and no panel "
              f"Kreis column); Kreise are labelled by code")
    kre[nm] = [names.get(c, c) for c in kre["AGS5"]]
    bl[nm] = [LAENDER.get(c, c) for c in bl["AGS2"]]

    for lay, key in ((kre, "AGS5"), (bl, "AGS2")):
        dup = lay[nm].duplicated(keep=False)
        lay["LABEL"] = lay[nm].astype(str)
        lay.loc[dup, "LABEL"] = lay.loc[dup, nm].astype(str) + " (" + lay.loc[dup, key] + ")"
        assert lay[key].is_unique and lay["LABEL"].is_unique

    _dirs(cfg.griddir)
    kre.to_file(cache, layer="kreis", driver="GPKG")
    bl.to_file(cache, layer="bundesland", driver="GPKG")
    print(f"regions: {len(kre)} Kreise | {len(bl)} Bundesländer -> {cache.name}")
    return kre, bl


def raster_crs(src, cfg: Config = CFG):
    """The raster's CRS, or the configured fallback, or a usable error.

    A GeoTIFF with no CRS is not an edge case worth guessing about: every
    coordinate in the pipeline -- snapped cells, bounding boxes, the Mannheim and
    Hamburg windows -- is interpreted in it.
    """
    if src.crs is not None:
        return src.crs
    if cfg.raster_crs:
        from rasterio.crs import CRS
        print(f"raster carries no CRS; using Config.raster_crs={cfg.raster_crs!r}")
        return CRS.from_user_input(cfg.raster_crs)
    raise ValueError(
        f"{Path(cfg.ghsl).name} carries no CRS, so its coordinates cannot be "
        f"interpreted.\n"
        f"    Stamp it on the file once (best):  rio edit-info --crs EPSG:3035 "
        f"{cfg.ghsl}\n"
        f"    or set Config.raster_crs to the right value for this run.\n"
        f"    Which one: run `gdalinfo` and read the corner coordinates. Germany "
        f"sits near x 4.0-4.7M / y 2.7-3.6M in EPSG:3035, and near x 0.7-1.1M / "
        f"y 5.7-6.1M in GHS-POP's native Mollweide (ESRI:54009).")


def grid_crs(cfg: Config = CFG, raster=None):
    """The CRS the cell coordinates are in -- the raster's, whatever it is.

    Everything that draws or bounds a box in metres needs this: the grid maps
    convert a Kreis polygon into the cells' own coordinates, and a hard-coded
    EPSG:3035 would silently select no cells from a Mollweide raster.
    """
    import rasterio
    with rasterio.open(raster or cfg.ghsl) as src:
        return raster_crs(src, cfg)


def land_view(cells, c2, mapping, node_reg, cfg: Config):
    """Restrict a national solve grid to one Bundesland, by filtering.

    Everything here was already snapped for the national run: the cell ids, the
    reduction, the node coordinates and the mapping are reused unchanged, and
    only the rows outside the Land are dropped. Clipping the raster instead --
    which is what this used to do -- shifts the raster origin, so every cell id
    changes and millions of already-snapped cells go back through OSRM for no
    gain.

    Note what a Land run still is: stores outside the Land are dropped, so a
    shopper near the border is assigned to a store inside it even when a nearer
    one lies across. That is a smoke test, not a published number -- for real
    Hamburg figures, filter the national cube to region 02000.
    """
    if not cfg.land:
        return cells, c2, mapping, node_reg
    keep = node_reg.filter(pl.col("kreis").str.starts_with(cfg.land))
    nodes = keep["node_id"].unique()
    m = mapping.filter(pl.col("dst").is_in(nodes.implode()))
    c2l = c2.filter(pl.col("cell_id").is_in(nodes.implode()))
    cl = cells.filter(pl.col("cell_id").is_in(m["src"].implode()))
    print(f"land {cfg.land}: {c2l.height:,}/{c2.height:,} solve nodes, "
          f"{cl.height:,}/{cells.height:,} cells, "
          f"{keep['pop'].sum():,.0f} people -- filtered from the national grid, "
          f"nothing re-snapped")
    return cl, c2l, m, keep


def _norm_ids(df: pl.DataFrame, col: str = "store_id") -> pl.DataFrame:
    """One dtype for store ids everywhere: Utf8.

    TDLinx arrives as an integer in some frames and a string in others, and
    `compose` casts to Utf8 while `prepare_stores` keeps the panel's dtype. A
    join between the two then fails on the key rather than on anything
    meaningful, so the ids are normalised the moment they enter the pipeline.
    """
    return df.with_columns(pl.col(col).cast(pl.Utf8)) if col in df.columns else df


def _snap_stores(panel: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    keep = tuple(c for c in ("Gruppe", cfg.chain_col, "Shop Type") if c in panel.columns)
    st = nv.prepare_stores(panel, keep=keep)
    _dirs(cfg.cachedir)
    return _norm_ids(nv.snap(_norm_ids(st), base_url=cfg.osrm, cache_path=cfg.cachedir,
                             drop_over_m=cfg.drop_over_m))


def stage_stores(panel: pl.DataFrame, cfg: Config = CFG, *, force=False) -> pl.DataFrame:
    """Every panel store, snapped once for this road network. Append-only.

    Pass the *unfiltered* panel: this artefact is independent of shop type, year
    and Land, so one snap serves every run. A cached frame is topped up with the
    stores the panel has gained since -- only those go to OSRM -- and they are
    APPENDED, never sorted in: existing stores keep their positions, which is
    what lets `nv.extend` grow a matrix instead of re-solving it.

    Coordinates in this file are the record. A re-geocode of existing stores
    does not reach it; run `--force stores` for that, which invalidates every
    matrix (they would be solved against the old positions).

    Gruppe, Chain and Shop Type ride along for eyeballing only -- collapsed to
    their most recent value, so nothing downstream may use them; membership and
    labels come from the panel, per store-year. first_year/last_year are stale
    for topped-up frames for the same reason and are not used either.
    """
    _dirs(cfg.snapdir)
    p, rej = cfg.snapdir / "stores.parquet", cfg.snapdir / "rejected.txt"
    ids = set(panel["TDLinx"].cast(pl.Utf8).unique().to_list())
    # rejected.txt remembers stores that failed to snap, so they are not retried
    # (and re-billed to OSRM) on every run. --force stores retries them.
    bad = set(rej.read_text(encoding="utf-8").split()) if rej.exists() and not force \
        else set()
    if p.exists() and not force:
        st = _norm_ids(pl.read_parquet(p))
        new = ids - set(st["store_id"].to_list()) - bad
        if not new:
            print(f"stores: cached {st.height:,} snapped ({len(bad):,} rejected) -> {p}")
            return st
        print(f"stores: {len(new):,} store(s) new in the panel -- snapping only those")
        add = _snap_stores(panel.filter(pl.col("TDLinx").cast(pl.Utf8).is_in(list(new))), cfg)
        # APPEND, never re-sort: matrix_frame() relies on old stores keeping
        # their row order so a matrix can be extended instead of re-solved
        st = pl.concat([st, add], how="diagonal_relaxed").select(st.columns)
    else:
        print(cache_note(cfg))
        new = ids
        st = _snap_stores(panel, cfg)
    # a store asked for but absent after snapping was dropped by snap() (no road
    # within drop_over_m) or prepare_stores() (null coordinates)
    bad |= new - set(st["store_id"].to_list())
    tmp = p.with_name(p.stem + ".tmp.parquet")      # write-then-rename: atomic
    st.write_parquet(tmp)
    tmp.replace(p)
    rej.write_text("\n".join(sorted(bad)), encoding="utf-8")
    manifest(cfg.snapdir / "snap_manifest.json",
             {"network": cfg.network, "osrm": cfg.osrm, "drop_over_m": cfg.drop_over_m},
             what="the store snap")
    print(f"stores: {st.height:,} snapped, {len(bad):,} rejected (no road within "
          f"{cfg.drop_over_m:.0f} m, or no coordinates) -> {p}")
    return st


def rejected_ids(cfg: Config) -> list[str]:
    """Stores that failed to snap (no road within drop_over_m, or no coordinates).

    They have no catchment, so they are out of the market on BOTH sides: a store
    without catchment population must not carry sales area either. Read from
    snapped/rejected.txt, so a supply-only run (no solve) drops the same stores
    as a solve run. Stores new since the last snap are not in it yet.
    """
    p = cfg.snapdir / "rejected.txt"
    return p.read_text(encoding="utf-8").split() if p.exists() else []


def membership(panel: pl.DataFrame, types, *, id_col: str = "TDLinx",
               type_col: str = "Shop Type", year_col: str = "year") -> pl.DataFrame:
    """(store_id, year) for every year a store actually had one of `types`.

    Format is a property of the store-year, and so is membership: a store that
    converted from Supermarkt to Discounter in 2019 is in the supermarket market
    until 2018 and in the discounter one from 2019.
    """
    return (panel.filter(pl.col(type_col).is_in(list(types)))
            .select(pl.col(id_col).cast(pl.Utf8).alias("store_id"),
                    pl.col(year_col).cast(pl.Int32).alias("year"))
            .unique())


def matrix_frame(stores: pl.DataFrame, ids, saved=None) -> tuple[pl.DataFrame, int]:
    """The store frame a matrix indexes, and how many of its rows the matrix
    already covers.

    The frame belongs to the matrix, not to the current panel: its first rows are
    the stores the matrix was solved on, in the saved order, and stores of this
    type that are new since then are appended. A store that switched *into* this
    type therefore lands at the end rather than in the middle, and a store that
    left keeps its column (it is simply never a member again). Without a saved
    matrix it is every store in `ids`, in `stores` order.
    """
    want = pl.Series(list(ids), dtype=pl.Utf8)
    if saved is None:
        s = stores.filter(pl.col("store_id").is_in(want.implode()))
        if s.is_empty():
            raise ValueError("matrix_frame: no snapped store among the requested ids")
        return s, 0
    saved = pl.Series("store_id", list(saved), dtype=pl.Utf8)
    old = (saved.to_frame().with_row_index("_o")
           .join(stores, on="store_id", how="inner").sort("_o").select(stores.columns))
    if old.height != saved.len():
        raise ValueError(
            f"matrix_frame: {saved.len() - old.height:,} store(s) this matrix was "
            f"solved on are no longer in stores.parquet (it was rebuilt?). The "
            f"matrix cannot be extended; re-solve with --force matrix.")
    add = stores.filter(pl.col("store_id").is_in(want.implode())
                        & ~pl.col("store_id").is_in(saved.implode()))
    return pl.concat([old, add]), old.height


def stores_for_year(frame: pl.DataFrame, member_ids, year: int) -> pl.DataFrame:
    """The same frame, with the alive window rewritten to mean "a member this year".

    Row order and store_id are untouched -- `cand`/`dur` index this frame
    positionally; only `first_year`/`last_year` change. Membership comes from
    the panel alone: the collapsed first/last years on the frame are stale once
    the store cache has been topped up, and the panel already says which stores
    traded as what, in which year.
    """
    alive = pl.col("store_id").cast(pl.Utf8).is_in(
        pl.Series(list(member_ids), dtype=pl.Utf8).implode())
    # nv.assign() keeps a store iff first_year <= year <= last_year. Members get
    # first = last = year (alive); everyone else gets -1/-1 (never alive).
    w = pl.when(alive).then(pl.lit(year)).otherwise(pl.lit(-1))
    return frame.with_columns(w.cast(frame["first_year"].dtype).alias("first_year"),
                              w.cast(frame["last_year"].dtype).alias("last_year"))


def load_cells_cached(cfg: Config, raster: Path) -> pl.DataFrame:
    """`nv.load_cells` with its result kept on disk (a parquet round-trip instead
    of re-reading ~7M cells from a GeoTIFF on every solve-touching run)."""
    p, fpp = cfg.griddir / "cells_raw.parquet", cfg.griddir / "cells_raw.fp"
    fp = _fingerprint(raster)
    if p.exists() and _fresh(fpp, fp):
        return pl.read_parquet(p)
    cells = nv.load_cells(raster)
    _dirs(cfg.griddir)
    cells.write_parquet(p)
    _commit(fpp, fp)
    return cells


def stage_cells(stores: pl.DataFrame, cfg: Config = CFG, *, raster: Path | None = None,
                force=False):
    """(cells_orig, c2, mapping) -- the national solve grid.

    Pass the NATIONAL store frame, always. `adapt_resolution` coarsens a cell by
    its distance to the nearest store in the frame it is given, and this grid is
    shared by every run: built from one Land's stores, everything outside that
    Land would be coarsened as if it had no store nearby, for good.

    Frozen once built. Stores added later do not refine it; that costs a little
    resolution where a new store opens in a formerly empty area, and buys every
    matrix staying valid.
    """
    _dirs(cfg.snapdir, cfg.griddir)
    raster = Path(raster or cfg.ghsl)
    cp, mp = cfg.griddir / "cells_solve.parquet", cfg.griddir / "mapping.parquet"
    cells = load_cells_cached(cfg, raster)
    built = not (cp.exists() and mp.exists()) or force
    if not built:
        c2, mapping = pl.read_parquet(cp), pl.read_parquet(mp)
        print(f"cells: cached {c2.height:,} solve nodes")
    else:
        print(cache_note(cfg))
        kw = {} if cfg.cells_per_catchment is None else \
            dict(cells_per_catchment=cfg.cells_per_catchment)
        # three steps, two id remappings:
        #   cells (raster cell_id) --adapt_resolution--> c1 (fine cells keep their
        #   id; merged blocks get new ids above max(cell_id))  --snap--> same ids,
        #   positions moved onto roads  --dedupe--> c2 (node ids 0..n-1, one per
        #   distinct snapped position).  m1: raster->c1, m2: c1->node,
        #   mapping = compose(m1, m2): raster cell -> node.
        c1, m1 = nv.adapt_resolution(cells, stores, **kw)
        _dirs(cfg.cachedir)
        c1 = nv.snap(c1, base_url=cfg.osrm, cache_path=cfg.cachedir)
        c2, m2 = nv.dedupe(c1, tol_m=0.0)
        mapping = nv.compose(m1, m2)
        # write immediately: node ids are only stable given these exact inputs,
        # and every matrix is baked against them
        m1.write_parquet(cfg.griddir / "map_adapt.parquet")
        m2.write_parquet(cfg.griddir / "map_dedupe.parquet")
        mapping.write_parquet(mp)
        c2.write_parquet(cp)
    # checked on every run, built or cached: a swapped raster or network under a
    # cached grid is wrong numbers, not an error
    man = {"network": cfg.network, "grid": cfg.grid, "raster": _fingerprint(raster),
           "cells_per_catchment": cfg.cells_per_catchment}
    if built:
        man |= {"n_stores_for_resolution": stores.height, "n_nodes": c2.height}
    manifest(cfg.griddir / "grid_manifest.json", man, what="the solve grid")
    print(f"cells: {cells.height:,} raster cells -> {c2.height:,} solve nodes "
          f"({c2.height / cells.height:.1%})")
    return cells, c2, mapping


class MatrixHandle:
    """A solved matrix on disk whose arrays load only when an assignment needs them.

    A matrix refined to k=140 is ~8-9 GB in memory, and a run over all market
    definitions touches three of them. Held together, that is more than the
    workstation has -- and a run whose assignments are all cached needs none of
    it: the store frame (read from the file's small `store_id` array) is all the
    fingerprints use. So `get()` loads on first use and `drop()` releases.
    """

    def __init__(self, path: Path, c2: pl.DataFrame, arrays=None):
        self.path, self.c2, self._a = path, c2, arrays

    def get(self):
        if self._a is None:
            z = nv.load_matrix(self.path, cells=self.c2,
                               columns=("store_idx", "euclid_m", "durations"))
            self._a = (z["candidates"], z["durations"])
        return self._a

    def drop(self) -> None:
        self._a = None
        gc.collect()


def _saved_ids(mpath: Path) -> np.ndarray:
    """The store ids a matrix was solved on, without reading its arrays."""
    return nv.saved_store_ids(mpath)


def _matrix_file(d: Path) -> Path | None:
    """The matrix in a solve directory: Parquet, or a legacy .npz not yet converted."""
    for name in ("matrix.parquet", "matrix.npz"):
        if (d / name).exists():
            return d / name
    return None


def _convert_matrix(legacy: Path, mpath: Path, c2: pl.DataFrame) -> None:
    """One-time rewrite of a legacy .npz matrix as Parquet: same arrays, same
    store frame, no routing. The .npz is left in place until you delete it."""
    print(f"matrix: converting {legacy} -> {mpath.name} (once; no routing)")
    z = nv.load_matrix(legacy, cells=c2)
    frame = pl.DataFrame({"store_id": z["store_id"].astype(str)})
    nv.save_matrix(mpath, c2, frame, z["candidates"], z["durations"],
                   dist=z.get("distances"), cert=z.get("certified"))
    del z
    gc.collect()
    print(f"  done; {legacy.name} can be deleted once a run has used {mpath.name}")


def stage_matrix(name: str, c2: pl.DataFrame, stores: pl.DataFrame, ids,
                 cfg: Config = CFG, *, force=False):
    """(frame, MatrixHandle) for one store set -- in normal use, one shop type.

    `stores` is the snapped store frame, `ids` the stores that belong in this
    matrix (every store that ever traded as this type). The matrix decides its
    own frame (`matrix_frame`): stores new since the solve are appended and
    routed by `nv.extend` -- only the (node, store) pairs inside each node's
    candidate radius, exact, minutes. Nothing else ever re-solves a matrix
    except `--force matrix`.

    A cached matrix with no new stores is not loaded here at all; see
    MatrixHandle. Checkpointed throughout: kill it and re-run, it resumes.
    """
    # Three paths through this function:
    #   A. cached, no new stores    -> return a lazy handle; nothing is read
    #   B. cached, new stores       -> load, nv.extend() the new pairs, re-save
    #   C. no matrix (or --force)   -> full solve: candidates, travel_times,
    #                                  refine_until, save
    # Checkpoint folders: ckpt/temp (C, first pass), ckpt/refine_temp/k<k> (C,
    # each refinement pass), ckpt/extend_<old>_<new> (B, removed on success).
    # A fresh solve (C) leaves ckpt/temp and refine_temp behind; they are only
    # read by a resumed solve of the SAME candidate set and can be deleted once
    # matrix.parquet exists.
    d = cfg.solvedir(name)
    ckpt = d / "ckpt"
    _dirs(d, ckpt)
    mpath = d / "matrix.parquet"
    legacy = d / "matrix.npz"
    if legacy.exists() and not mpath.exists() and not force:
        _convert_matrix(legacy, mpath, c2)
    if mpath.exists() and not force:
        frame, n_old = matrix_frame(stores, ids, _saved_ids(mpath))
        if frame.height == n_old:
            print(f"matrix[{name}]: cached ({n_old:,} stores; arrays load on demand)")
            return frame, MatrixHandle(mpath, c2)
        print(f"matrix[{name}]: {frame.height - n_old:,} store(s) new since the "
              f"solve -- extending, not re-solving")
        cand, dur = MatrixHandle(mpath, c2).get()
        pf = nv.preflight(c2, base_url=cfg.osrm)
        ck = ckpt / f"extend_{n_old}_{frame.height}"
        cand, dur = nv.extend(c2, frame, cand, dur, n_old, base_url=cfg.osrm,
                              max_coords=max(1000, pf["max_coords"]),
                              max_workers=cfg.max_workers, checkpoint=ck)
        cert = nv.certified(cand, dur, rank=cfg.cert_rank)
        nv.save_matrix(mpath, c2, frame, cand, dur, cert=cert)
        nv.certification_report(cand, dur, ranks=1)
        shutil.rmtree(ck, ignore_errors=True)
        # not handed on in memory: run() keeps one handle per shop type, so
        # arrays held here would sit next to the next type's solve. Re-read
        # from disk when the assignment needs them.
        del cand, dur
        return frame, MatrixHandle(mpath, c2)

    # path C: full solve. Note max(1000, ...): if preflight measured less than
    # 1000 coordinates the server's --max-table-size is too small and the /table
    # calls below will be rejected anyway -- fix the server, not this number.
    frame, _ = matrix_frame(stores, ids)
    pf = nv.preflight(c2, base_url=cfg.osrm)
    max_coords = max(1000, pf["max_coords"])
    cand = nv.candidates(c2, frame, k=cfg.k)
    dur, _ = nv.travel_times(c2, frame, cand, base_url=cfg.osrm,
                             max_coords=max_coords, checkpoint=ckpt / "temp",
                             max_workers=cfg.max_workers, progress_every=50)
    cand, dur, cert = nv.refine_until(
        c2, frame, cand, dur, target=cfg.cert_target, rank=cfg.cert_rank,
        ks=cfg.ks, base_url=cfg.osrm, max_coords=max_coords,
        checkpoint=ckpt / "refine_temp")
    nv.save_matrix(mpath, c2, frame, cand, dur, cert=cert)
    nv.certification_report(cand, dur, ranks=min(3, cand.k))
    del cand, dur                     # see path B: re-read on demand, not held
    return frame, MatrixHandle(mpath, c2)


class LazyYears(dict):
    """year -> assignment, read from its parquet on access.

    Twelve years of 5M-row assignments per spec, for three specs, is several GB
    a run never needs at once: the cube, the maps and the coverage check each
    read one year at a time. Keys, `in`, `max()` and truthiness behave as for
    the dict of frames this replaces; only indexing reads the file.
    """

    def __getitem__(self, year):
        return pl.read_parquet(super().__getitem__(year))


def _write_raster(spec, year, a, cells, mapping, cfg: Config, raster=None) -> Path | None:
    """The winning store per 100 m cell, as a GeoTIFF, when the run asked for it.
    `a` is the assignment frame, or a callable returning it (read only if the
    raster still has to be written)."""
    if not cfg.write_rasters or cells is None or mapping is None:
        return None
    p = cfg.rasterdir(spec) / f"nearest_store_{spec}_{year}.tif"
    if p.exists():
        return p
    _dirs(cfg.rasterdir(spec))
    nv.write_label_raster(cells, a() if callable(a) else a, mapping, raster or cfg.ghsl, p)
    print(f"  raster[{spec}] {year}: {p}")
    return p


def _merge_iso(files, out: Path, *, k: int, rows_per_part: int = 10_000_000) -> None:
    """Compose per-type isochrone frames into one, a range of nodes at a time.

    Each per-type frame is ~nodes x iso_k rows (100M nationally); sorting two or
    three of them at once took >20 GB. Composition is per node, so splitting by
    cell_id range gives the same frame; each part is written as it is done.
    """
    import pyarrow.parquet as pq
    scans = [pl.scan_parquet(f) for f in files]
    st = pl.concat([s.select(pl.col("cell_id").min().alias("lo"),
                             pl.col("cell_id").max().alias("hi"), pl.len().alias("n"))
                    for s in scans]).collect()
    lo, hi = int(st["lo"].min()), int(st["hi"].max()) + 1
    edges = np.unique(np.linspace(lo, hi, max(1, int(st["n"].sum()) // rows_per_part) + 1,
                                  dtype=np.int64))
    tmp, w = out.with_name(out.stem + ".tmp.parquet"), None
    for a, b in zip(edges[:-1], edges[1:]):
        rng = pl.col("cell_id").is_between(int(a), int(b), closed="left")
        parts = [s.filter(rng).collect() for s in scans]
        if not any(p.height for p in parts):
            continue
        t = CO.topk(parts, k=k).to_arrow()
        w = w or pq.ParquetWriter(tmp, t.schema, compression="zstd")
        w.write_table(t)
        del parts, t
    if w is not None:
        w.close()
        tmp.replace(out)


def stage_assign(spec: str, c2, solved: dict, cfg: Config = CFG, *,
                 panel: pl.DataFrame, types=None,
                 cells=None, mapping=None, raster=None, force=False
                 ) -> tuple[LazyYears, dict[int, list]]:
    """Winner-take-all catchment per year, composed from the per-type solves.

    `solved` maps shop type -> (frame, MatrixHandle). Each type is assigned from
    its own matrix using the stores that traded as that type that year, and the
    results are combined by taking the nearest across types -- exact, see
    scripts.isochrones.compose.

    Reuse is per year: a year is recomputed exactly when its matrices or its
    store membership changed, recorded in a sidecar written after the files.

    Memory: the loop runs type by type, not year by year. One matrix is loaded,
    every stale year is assigned from it and merged into that year's partial
    result on disk, and the matrix is released before the next type loads. The
    nearest-of-the-nearest is associative, so merging pairwise equals composing
    all at once. Peak memory is one matrix plus one year's assignment.

    Certification is reported per year, because it is per year: with some stores
    closed or in another format that year, a node's fastest member can sit beyond
    the bound. The composed winner is proven exact when it beats min over types of
    d_k / v_max; nodes with no member candidate at all count as unproven.

    Returns the assignments (LazyYears: read from disk on access) and the store
    ids in the spec each year.
    """
    d = cfg.derivdir(spec)
    ad = d / "assignments"
    _dirs(ad, d / "catchments", d / "isochrone")
    types = list(types or SPECS[spec])
    if missing := [t for t in types if t not in solved]:
        raise ValueError(f"stage_assign[{spec}]: no solve for {', '.join(missing)}")
    mem = {t: membership(panel, [t]) for t in types}
    mfp = {t: store_fingerprint(solved[t][0]) for t in types}
    # the matrix frames hold snapped stores only: a rejected store is a member
    # in the panel but has no column, so it leaves the market (see rejected_ids)
    fids = {t: set(solved[t][0]["store_id"].to_list()) for t in types}

    # ---- phase 1: decide, per year, whether the cached file is still valid.
    # The sidecar fingerprint is, per shop type in the spec:
    #     <type>:<store_fingerprint(matrix frame)>:<hash of that year's members>
    # so a year recomputes when a matrix was extended/re-solved or when the panel
    # changed who traded as what that year. Label changes (Gruppe/Chain) do NOT
    # invalidate assignments: the assignment only knows store ids.
    paths, alive_by_year, todo, cached = LazyYears(), {}, {}, []
    for year in cfg.years:
        tids = {t: sorted(mem[t].filter(pl.col("year") == year)["store_id"].to_list())
                for t in types}
        # tids (raw panel members) stay in the fingerprint: the assignment is the
        # same with or without the rejected ids, so cached years remain valid
        alive_by_year[year] = sorted({i for t, v in tids.items() for i in v if i in fids[t]})
        if not any(tids.values()):
            print(f"  assign[{spec}] {year}: no store in the spec, skipped")
            continue
        p = ad / f"assignments_{spec}_{year}.parquet"
        fp = "|".join(f"{t}:{mfp[t]}:{_h(tids[t])}" for t in types)
        if p.exists() and not force and _fresh(p.with_suffix(".fp"), fp):
            paths[year] = p
            cached.append(year)
            _write_raster(spec, year, lambda p=p: pl.read_parquet(p), cells, mapping,
                          cfg, raster)
            continue
        todo[year] = (fp, tids)
    if cached:
        print(f"  assign[{spec}]: {len(cached)} year(s) unchanged, reused "
              f"({min(cached)}-{max(cached)})")
    if not todo:
        return paths, alive_by_year

    # ---- phase 2: type-major loop. For each shop type: load its matrix once,
    # assign every stale year from it, merge into <part>/<year>.parquet with
    # CO.rank1 (nearest of the two), release the matrix.
    #   bound[y]  per node, min over the spec's types of d_k / v_max: no store
    #             outside any candidate set can be faster than this.
    #   k_iso[y]  width of the isochrone frame for y (only for cfg.iso_years).
    part = ad / "_partial"                       # leftovers of a killed run are stale
    shutil.rmtree(part, ignore_errors=True)
    _dirs(part)
    n = c2.height
    bound = {y: np.full(n, np.inf, np.float32) for y in todo}
    k_iso = {y: 1 for y in todo}
    vmax = np.float32(nv.V_MAX_KMH / 3.6)
    for t in types:
        ys = [y for y in todo if todo[y][1][t]]
        if not ys:
            continue
        frame, mat = solved[t]
        cand, dur = mat.get()
        for y in ys:
            sy = stores_for_year(frame, todo[y][1][t], y)
            a_t = nv.assign(c2, sy, cand, dur, year=y, ranks=1)
            np.minimum(bound[y], cand.dist_m[:, -1] / vmax, out=bound[y])
            pp = part / f"{y}.parquet"
            if pp.exists():
                a_t = CO.rank1([pl.read_parquet(pp), a_t])
            a_t.write_parquet(pp)
            if y in cfg.iso_years and cfg.iso_k > 0:
                ranks = min(int(cand.k), cfg.iso_k)
                k_iso[y] = max(k_iso[y], ranks)
                nv.assign(c2, sy, cand, dur, year=y, ranks=ranks,
                          max_seconds=cfg.iso_min * 60
                          ).write_parquet(part / f"iso_{y}_{slug(t)}.parquet")
            del a_t
        del cand, dur
        mat.drop()
        print(f"  assign[{spec}]: {t} merged into {len(ys)} year(s)")

    # ---- phase 3: finalise each year: move the merged file into place, write
    # the per-store catchment summary and the isochrone frame, report the share
    # of population whose winner is not proven (winner slower than bound, or no
    # winner at all), and only THEN commit the sidecar.
    pos = c2.select("cell_id").with_row_index("_i")
    pop = c2["pop"].cast(pl.Float64).to_numpy()
    for y, (fp, _) in todo.items():
        p = ad / f"assignments_{spec}_{y}.parquet"
        (part / f"{y}.parquet").replace(p)
        a = pl.read_parquet(p)
        best = np.full(n, np.inf, np.float32)
        j = a.select("cell_id", "duration_s").join(pos, on="cell_id")
        best[j["_i"].to_numpy()] = j["duration_s"].to_numpy()
        unproven = ~(best <= bound[y])
        nv.catchments(a).write_parquet(d / "catchments" / f"catchments_{spec}_{y}.parquet")
        iso = sorted(part.glob(f"iso_{y}_*.parquet"))
        if iso:
            # merged pairwise and truncated at each step: the k nearest of a
            # union are among the per-type k nearest, so this is the same frame
            # as concatenating them all, without ever holding them all at once
            _merge_iso(iso, d / "isochrone" / f"isochrone_{spec}_{y}.parquet",
                       k=k_iso[y])
        _commit(p.with_suffix(".fp"), fp)
        print(f"  assign[{spec}] {y}: {a.height:,} nodes | "
              f"{len(alive_by_year[y]):,} stores in spec | "
              f"{pop[unproven].sum() / pop.sum():.2%} of population not certified "
              f"this year ({int(unproven.sum()):,} nodes)")
        _write_raster(spec, y, a, cells, mapping, cfg, raster)
        paths[y] = p
        del a
    shutil.rmtree(part, ignore_errors=True)
    return paths, alive_by_year


def stage_regions(cells, mapping, kreis_gdf, cfg: Config = CFG, *, raster=None,
                  force=False) -> pl.DataFrame:
    """(node_id, kreis, pop, n_cells, area_km2). Once per grid: every spec and
    year joins against this instead of re-expanding 7.27M populated cells."""
    p = cfg.griddir / "node_regions.parquet"
    if p.exists() and not force:
        print(f"regions: cached {p.name}")
        return pl.read_parquet(p)
    labels, lookup = C.rasterize_regions(kreis_gdf, raster or cfg.ghsl, cells,
                                         id_col=cfg.kreis_id, name_col=cfg.kreis_name)
    nr = C.node_regions(mapping, cells, labels, lookup, id_col=cfg.kreis_id)
    nr.write_parquet(p)
    return nr


# ==========================================================================
# cube
# ==========================================================================

def cube_dir(cfg: Config, spec: str, exclude=()) -> Path:
    """One directory per market definition: a run with and one without excluded
    actors must not overwrite each other."""
    return cfg.outdir / "cube" / (spec + ("__excl" if exclude else ""))


def top_area(cfg: Config) -> tuple[str, str, str]:
    """(level, region, name) of the run's whole reporting area.

    Germany in a national run. A Land run has no Germany: its cube carries no DE
    level, and the Land's own BL row is the total. Everything that reads "the
    national number" (trend and heatmap figures, tables, diagnostics) asks here.
    """
    return ("BL", cfg.land, LAENDER[cfg.land]) if cfg.land else ("DE", "DE", "Germany")


def cube_levels(cfg: Config) -> tuple[str, ...]:
    return ("BL", "KREIS") if cfg.land else C.LEVELS


def report_levels(rc: RunConfig, cfg: Config) -> list[str]:
    """rc.levels as this run can report them: in a Land run DE means the Land total."""
    return list(dict.fromkeys("BL" if (lv == "DE" and cfg.land) else lv for lv in rc.levels))


def stage_cube(spec: str, types, assignments, node_reg, panel, cfg: Config = CFG, *,
               alive_by_year, actor_panel=None, exclude=()) -> None:
    """Shares + HHI for one spec, written as shares|hhi _ supply|demand _ year.

    Always recomputed -- seconds per year, and nothing to keep in sync. Written
    per (side, year), so a supply-only run (VKFL, Umsatzpotential; no solve)
    never touches the catchment results, and a run over two years never touches
    the other ten.

    What the files were built from is recorded in `manifest.json` (spec
    definition, panel, exclusions). If that changes, the directory is cleared
    before writing: the years this run does not rebuild then go missing, which
    shows, rather than staying behind under the old definition, which would not.

    Supply side: the store's own VKFL/Umsatzpotential in its own Kreis, for every
    store that traded in the spec that year -- a panel fact, identical with or
    without the solve. Demand side: catchment population and area from the
    assignment, split by the Kreis each cell lies in.
    """
    d = cube_dir(cfg, spec, exclude)
    levels = cube_levels(cfg)
    man = {"types": sorted(types), "panel": _fingerprint(cfg.panel),
           "exclude": sorted(exclude), "fringe": sorted(UNIDENTIFIED),
           "levels": list(levels), "rejected": _h(sorted(rejected_ids(cfg)))}
    mp = d / "manifest.json"
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8")) != man:
        print(f"cube[{spec}]: spec, panel or exclusions changed since the last build "
              f"-- clearing {d}")
        shutil.rmtree(d)
    _dirs(d)
    mp.write_text(json.dumps(man, indent=1), encoding="utf-8")
    # files are split by side so a supply-only run (assignments=None) rewrites
    # only the *_supply_* files and leaves earlier *_demand_* files in place
    side = pl.when(pl.col("basis").is_in(["pop", "area"])).then(pl.lit("demand")) \
        .otherwise(pl.lit("supply"))
    n, errors = 0, []
    for y in cfg.years:
        a = None if assignments is None or y not in assignments else {y: assignments[y]}
        try:
            s, h = C.cube(a, node_reg if a else None, panel, spec=spec, years=[y],
                          alive_by_year=alive_by_year, exclude=exclude, fringe=UNIDENTIFIED,
                          actor_panel=actor_panel, chain_col=cfg.chain_col,
                          id_col="TDLinx", region_col=cfg.kreis_id, levels=levels)
        except ValueError as e:
            print(f"cube[{spec}] {y}: {e}")
            errors.append(f"{y}: {e}")
            continue
        C.validate(s, h)
        for sd in (("supply", "demand") if a else ("supply",)):
            s.filter(side == sd).write_parquet(d / f"shares_{sd}_{y}.parquet")
            h.filter(side == sd).write_parquet(d / f"hhi_{sd}_{y}.parquet")
            n += 1
    if not n:
        # nothing at all was built: stop here with the reason, instead of letting
        # read_cube report a missing cube and suggest the command just run
        raise RuntimeError(f"cube[{spec}]: no year could be built --\n    "
                           + "\n    ".join(errors or ["no year in the run's years"]))
    print(f"cube[{spec}]: {n} (side, year) file pair(s) -> {d}")


def read_cube(cfg: Config = CFG, specs=None, exclude=()):
    """Concatenated cube across specs -- what every figure and export reads.

    Raises with the command that would build what is missing. A cube built
    nationally is invisible to a Land run and vice versa (`de_100m` vs
    `de_100m_01`): the output namespace follows `land`.
    """
    specs = list(specs or SPECS)
    S, H, missing = [], [], []
    for x in specs:
        d = cube_dir(cfg, x, exclude)
        fs, fh = sorted(d.glob("shares_*.parquet")), sorted(d.glob("hhi_*.parquet"))
        if not fs or not fh:
            missing.append(x)
            continue
        S += [pl.read_parquet(f) for f in fs]
        H += [pl.read_parquet(f) for f in fh]
    if missing:
        # The directory also depends on the RUN NAME (cfg.outdir ends in it): a
        # cube built by runs/a.toml is invisible to runs/b.toml. The suggested
        # command below omits --config; add the run's own config when you use it.
        land = f" --land {cfg.land}" if cfg.land else ""
        excl = " --exclude-actors other" if exclude else ""
        raise FileNotFoundError(
            f"no cube for {', '.join(missing)} under {cfg.outdir / 'cube'}\n"
            f"    build it:  python -m scripts.run_pipeline --stages cube "
            f"--specs {','.join(missing)}{land}{excl}\n"
            f"    (the directory depends on grid={cfg.grid!r}, land="
            f"{cfg.land or 'None'!r} and whether actors are excluded)")
    shares = pl.concat(S, how="vertical_relaxed").sort("spec", "year", "level", "region")
    hhi = pl.concat(H, how="vertical_relaxed").sort("spec", "year", "level", "region")
    # checked again on the whole cube, not only per year as it was written: this
    # is what catches files left over from another spec definition or exclusion
    # setting, which would show up as shares that no longer sum to 1
    C.validate(shares, hhi)
    return shares, hhi


# ==========================================================================
# figures
# ==========================================================================

#: which report level each figure needs, and whether it varies by basis
FIG_NEEDS = {"map_hhi": ("KREIS", True), "map_share": ("KREIS", True),
             "map_leader": ("KREIS", True), "heat_shares": ("DE", True),
             "trend_hhi": ("DE", True),
             "trend_shares": ("DE", True), "dist_hhi": ("KREIS", True),
             "scatter_basis": ("KREIS", False), "bars_bl": ("BL", True),
             "grid_maps": (None, False), "map_catchments": (None, False)}


def fig_stem(what: str, spec: str, actor_level: str, basis: str | None, year: int) -> str:
    """One naming scheme for batch, GUI and notebook files. National trends carry
    every spec in one panel, so their name carries none."""
    return "_".join(x for x in (what, "" if what == "trend_hhi" else spec, actor_level,
                                basis or "", str(year)) if x)


def draw_figure(what: str, hhi, shares, *, spec: str, year: int, basis: str | None,
                actor_level: str, gdfs, cfg: Config = CFG, rc: RunConfig | None = None):
    """One figure from the cube. Returns an Axes, or None if it drew nothing.

    Single dispatcher so the notebook, the GUI and the batch runner draw the
    same figure from the same slice -- a figure that exists in two code paths
    eventually exists in two versions.
    """
    kreis_gdf, bl_gdf = gdfs
    lev, reg, area = top_area(cfg)
    lab = F.SPEC_LABEL.get(spec, spec)
    blab = F.BASIS_LABEL.get(basis, basis)
    if what == "map_hhi":
        return F.plot_choropleth(hhi, kreis_gdf, level="KREIS", spec=spec, year=year,
                                 basis=basis, actor_level=actor_level,
                                 id_col=cfg.kreis_id, label="HHI",
                                 title=f"{lab} — HHI by Kreis, {year}\n{blab}, {actor_level}")
    if what == "map_share":
        return F.plot_share_map(shares, kreis_gdf, level="KREIS", spec=spec, year=year,
                                basis=basis, actor_level=actor_level, id_col=cfg.kreis_id,
                                title=f"{lab} — share of the largest {actor_level} by "
                                      f"Kreis, {year}\n{blab}")
    if what == "map_leader":
        return F.plot_leader_map(shares, kreis_gdf, level="KREIS", spec=spec, year=year,
                                 basis=basis, actor_level=actor_level,
                                 id_col=cfg.kreis_id,
                                 title=f"{lab} — leading {actor_level} by Kreis, "
                                       f"{year}\n{blab}")
    if what == "heat_shares":
        return F.plot_share_heatmap(shares, spec=spec, basis=basis, level=lev, region=reg,
                                    actor_level=actor_level, top=10,
                                    title=f"{lab} — {actor_level} shares, {area}, {blab}")
    if what == "trend_hhi":
        return F.plot_trend(hhi, basis=basis, actor_level=actor_level, by="spec",
                            level=lev, region=reg,
                            title=f"HHI, {area}, {blab} ({actor_level})")
    if what == "trend_shares":
        return F.plot_share_trend(shares, level=lev, region=reg, spec=spec,
                                  basis=basis, actor_level=actor_level, top=6,
                                  title=f"{lab} — {actor_level} shares, {area}, {blab}")
    if what == "dist_hhi":
        return F.plot_distribution(hhi, level="KREIS", spec=spec, basis=basis,
                                   actor_level=actor_level,
                                   title=f"{lab} — Kreis HHI, {blab}")
    if what == "scatter_basis":
        # pair a demand-side basis against a supply-side one: the point of this
        # figure is where the catchment reading and the store reading disagree,
        # not where two sales-area measures agree
        b = list((rc or RunConfig()).report_bases())
        dem = [x for x in b if x in ("pop", "area")]
        sup = [x for x in b if x not in ("pop", "area")]
        x = (dem or b or ["pop"])[0]
        y = (sup or [z for z in b if z != x] or ["vkfl"])[0]
        if x == y:
            return None
        return F.plot_basis_agreement(hhi, spec=spec, year=year, actor_level=actor_level,
                                      x_basis=x, y_basis=y,
                                      title=f"{lab} — {F.BASIS_LABEL.get(x, x)} vs "
                                            f"{F.BASIS_LABEL.get(y, y)} HHI by Kreis, {year}")
    if what == "bars_bl":
        names = dict(zip(bl_gdf["AGS2"].astype(str), bl_gdf["LABEL"].astype(str)))
        return F.plot_region_bars(hhi, level="BL", spec=spec, basis=basis,
                                  actor_level=actor_level, names=names,
                                  title=f"{lab} — HHI by Bundesland, {blab}")
    return None


def stage_figures(shares, hhi, kreis_gdf, bl_gdf, cfg: Config = CFG,
                  rc: RunConfig | None = None) -> list[Path]:
    """Every figure the run asked for, over its selected specs, bases and
    breakdowns. Each panel is a slice of the cube, so any number in any figure
    traces back to rows in shares/hhi."""
    rc = rc or RunConfig()
    F.set_style()
    # a run with excluded actors is a different market: its figures go apart
    fig_dir = cfg.outdir / "figures" / ("excl" if rc.exclude_actors else "")
    _dirs(fig_dir)
    year = rc.year_focus or int(hhi["year"].max())
    bases = [b for b in rc.report_bases() if b in hhi["basis"].unique().to_list()]
    levels = report_levels(rc, cfg)
    written: list[Path] = []
    for what in rc.figures:
        need, per_basis = FIG_NEEDS.get(what, (None, True))
        if what in ("grid_maps", "map_catchments"):
            continue                      # need the solve, handled in run()
        need = top_area(cfg)[0] if need == "DE" else need
        if need and need not in levels:
            print(f"figures: {what} needs {need}-level coverage, not selected — skipped")
            continue
        # trend_hhi draws every spec in one panel: once, not once per spec
        for spec in (rc.specs[:1] if what == "trend_hhi" else rc.specs):
            for al in rc.actor_levels:
                for basis in (bases if per_basis else [bases[0] if bases else "pop"]):
                    ax = draw_figure(what, hhi, shares, spec=spec, year=year,
                                     basis=basis, actor_level=al,
                                     gdfs=(kreis_gdf, bl_gdf), cfg=cfg, rc=rc)
                    if ax is None:
                        continue
                    written += F.save(ax.figure,
                                      fig_dir / fig_stem(what, spec, al, basis, year),
                                      formats=rc.formats)
                    if not per_basis:
                        break
    print(f"figures: {len(written)} files -> {fig_dir}")
    return written


def stage_export(shares, hhi, cfg: Config = CFG, rc: RunConfig | None = None,
                 names: dict | None = None) -> list[Path]:
    """CSV for the selected metrics, levels and bases. Parquet stays the archive;
    CSV is what goes to co-authors, so it carries region names, not only codes."""
    # Called from run() only when "figures" is among the stages: a run with
    # --stages cube writes the cube parquet but NOT the CSVs.
    rc = rc or RunConfig()
    if not rc.export_csv:
        return []
    d = cfg.outdir / "cube" / "csv"
    _dirs(d)
    names = names or {}
    bases = list(rc.report_bases())
    have = set(hhi["basis"].unique().to_list())
    if missing := [b for b in bases if b not in have]:
        print(f"export: {', '.join(missing)} not in the cube — the panel column is "
              f"missing, so this scope cannot be reported for it")
        bases = [b for b in bases if b in have]
    out = []
    for lev in report_levels(rc, cfg):
        flt = (pl.col("level") == lev) & pl.col("spec").is_in(list(rc.specs)) \
            & pl.col("basis").is_in(bases) \
            & pl.col("actor_level").is_in(list(rc.actor_levels)) \
            & pl.col("year").is_in(list(rc.years))
        for metric, df, keys in (("hhi", hhi, ["spec", "year", "basis", "actor_level",
                                               "region"]),
                                 ("share", shares, ["spec", "year", "basis",
                                                    "actor_level", "region", "rank"])):
            if metric not in rc.metrics:
                continue
            # the scope is in the name: an "all" run and a "food" run write the
            # same levels, and without it the second silently overwrites the first
            p = (d / f"{'hhi' if metric == 'hhi' else 'shares'}_{lev.lower()}"
                 f"_{rc.scope}{'_excl' if rc.exclude_actors else ''}.csv")
            (df.filter(flt)
               .with_columns(pl.col("region").replace(names).alias("region_name"))
               .sort(keys).write_csv(p))
            out.append(p)
    print(f"export: {len(out)} CSV file(s) -> {d}")
    return out


def year_labels(panel: pl.DataFrame, year: int, col: str = "Gruppe", *,
                id_col: str = "TDLinx", year_col: str = "year") -> pl.DataFrame:
    """(store_id, <col>) as it stood in `year`. Banners switch; the map should
    show the operator of that year, not the last one on record."""
    return (panel.filter(pl.col(year_col) == year)
            .select(pl.col(id_col).cast(pl.Utf8).alias("store_id"),
                    pl.col(col).cast(pl.Utf8).alias(col))
            .group_by("store_id").agg(pl.col(col).last()))


def grid_view(cells: pl.DataFrame, stores: pl.DataFrame, raster, crs, bbox):
    """Cells inside `bbox` and the stores, both with x/y in the RASTER's CRS.
    Returns (None, None) when this run has no cell in the window.

    Selection uses the cells' own coordinates, which nv.load_cells always writes
    in EPSG:3035 whatever the raster's CRS: the window (in the raster's CRS) is
    taken to 3035, the cells inside are kept, and only then are their centres
    recomputed on the raster's grid for drawing -- the squares are the raster's
    cells, and drawn in 3035 they would be rotated against every boundary.
    Stores are reprojected from their original (unsnapped) position.
    """
    import rasterio
    import shapely
    from pyproj import Transformer
    with rasterio.open(raster) as src:
        T = src.transform
    # 1) window (raster CRS) -> EPSG:3035, densified so the reprojected outline
    #    is not a skewed quadrilateral; select cells on their stored 3035 x/y
    win = shapely.segmentize(shapely.box(*bbox), 250.0)
    wx, wy = Transformer.from_crs(crs, f"EPSG:{nv.EPSG}", always_xy=True).transform(
        *np.asarray(win.exterior.coords).T)
    sub = cells.filter(pl.col("x").is_between(float(np.min(wx)), float(np.max(wx))),
                       pl.col("y").is_between(float(np.min(wy)), float(np.max(wy))))
    if sub.is_empty():
        return None, None
    # 2) recompute the selected cells' centres in the raster CRS from row/col,
    #    then cut exactly to the window there
    c, r = sub["col"].to_numpy() + 0.5, sub["row"].to_numpy() + 0.5
    nx, ny = T.a * c + T.b * r + T.c, T.d * c + T.e * r + T.f
    x0, y0, x1, y1 = bbox
    inside = (nx >= x0) & (nx <= x1) & (ny >= y0) & (ny <= y1)
    if not inside.any():
        raise ValueError(
            f"grid_view: {sub.height:,} cells lie in this window by their EPSG:3035 "
            f"coordinates, but none by the raster grid ({raster}, CRS "
            f"{crs.to_string() if hasattr(crs, 'to_string') else crs}). The raster "
            f"CRS/transform and the cells do not agree -- cells_raw.parquet from a "
            f"different raster? First cell: 3035 ({sub['x'][0]:.0f}, {sub['y'][0]:.0f}), "
            f"raster grid ({nx[0]:.0f}, {ny[0]:.0f}); window {tuple(round(v) for v in bbox)}.")
    sub = sub.with_columns(pl.Series("x", nx), pl.Series("y", ny)).filter(pl.Series(inside))
    lon, lat = (("lon_orig", "lat_orig") if "lon_orig" in stores.columns
                else ("lon", "lat"))
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        stores[lon].to_numpy(), stores[lat].to_numpy())
    st = (stores.drop([k for k in ("x_orig", "y_orig") if k in stores.columns])
          .with_columns(pl.Series("x", np.asarray(x)), pl.Series("y", np.asarray(y))))
    return sub, st


def main_area(sel, crs, *, km: float = 10.0, id_col: str = "AGS5"):
    """A region for mapping: its polygon without parts detached far from the rest.

    Hamburg's Kreis includes Neuwerk, ~100 km out in the North Sea; drawn whole,
    the map window stretches to it and the city shrinks to a corner. Parts more
    than `km` from the largest part are left out of the map window and outline;
    river and coastal islands close by stay. Figures only -- the numbers still
    count every part.
    """
    parts = sel.to_crs(crs).explode(index_parts=False).reset_index(drop=True)
    core = parts.geometry[parts.geometry.area.idxmax()]
    keep = parts[parts.geometry.distance(core) <= km * 1000]
    if len(keep) < len(parts):
        print(f"  {sel[id_col].iloc[0]}: {len(parts) - len(keep)} detached part(s) more "
              f"than {km:g} km from the main area left off the map")
    return keep.dissolve(by=id_col).reset_index()


def _year_stores(stores, alive_by_year, year, panel, by):
    """Stores trading in the spec that year, labelled as they stood that year."""
    if alive_by_year is not None:
        stores = stores.filter(pl.col("store_id").is_in(
            pl.Series(list(alive_by_year.get(year, [])), dtype=pl.Utf8).implode()))
    if panel is not None and by in panel.columns:
        stores = (stores.drop(by) if by in stores.columns else stores).join(
            year_labels(panel, year, by), on="store_id", how="left")
    return stores


def stage_catchments(spec: str, cells, mapping, assignments, stores, kreis_gdf,
                     cfg: Config = CFG, *, alive_by_year=None, year=None, by="Gruppe",
                     kreise=None,
                     panel: pl.DataFrame | None = None, crs=None, raster=None,
                     save_polygons: bool = True) -> list[Path]:
    """Catchment polygons per Kreis, drawn and (optionally) written out.

    Everything this needs already exists after `--stages all`: the assignment
    gives every 100 m cell its winning store, and a catchment is just the outline
    around one store's cells. No re-solve, no OSRM.
    """
    F.set_style()
    fig_dir = cfg.outdir / "figures" / "catchments"
    _dirs(fig_dir)
    year = year or max(assignments)
    a = assignments[year]
    crs = crs if crs is not None else grid_crs(cfg, raster)
    stores = _year_stores(stores, alive_by_year, year, panel, by)
    targets = list(kreise or cfg.catchment_kreise or cfg.cities.values())
    written = []
    for ags5 in targets:
        sel = kreis_gdf[kreis_gdf[cfg.kreis_id].astype(str) == str(ags5)]
        if sel.empty:
            print(f"catchments: {ags5} not in the Kreis layer, skipped")
            continue
        name = str(sel["LABEL"].values[0])
        sel = main_area(sel, crs, id_col=cfg.kreis_id)
        # 2 km of context: catchments that straddle the border are clipped, not
        # dropped -- a store just outside serving inside is part of the picture
        bbox = F.region_bbox(sel, cfg.kreis_id, ags5, pad_km=2, crs=crs)
        sub, st = grid_view(cells, stores, raster or cfg.ghsl, crs, bbox)
        if sub is None:
            print(f"catchments: no cell of this run inside {name} -- a Land-only "
                  f"solve covers only its Land; skipped")
            continue
        g = F.grid_slice(sub, mapping, a, st, bbox=bbox, by=by)
        polys = F.catchment_polygons(g, crs=crs, by="actor")
        import geopandas as gpd
        clipped = gpd.clip(polys, sel.to_crs(crs))
        clipped = clipped[~clipped.geometry.is_empty]
        ax = F.plot_catchments(clipped, by="actor", region=sel, stores=None, crs=crs,
                               title=f"{name} — store catchments, {year}\n"
                                     f"{F.SPEC_LABEL.get(spec, spec)}, nearest by drive time")
        written += F.save(ax.figure, fig_dir / f"catchments_{slug(name)}"
                                               f"_{spec}_{year}")
        if save_polygons:
            p = fig_dir / f"catchments_{ags5}_{spec}_{year}.gpkg"
            clipped.to_parquet(p.with_suffix(".parquet"))
            written.append(p)
    print(f"catchments: {len(written)} file(s) -> {fig_dir}")
    return written


def stage_grid_maps(spec: str, cells, mapping, assignments, stores, kreis_gdf,
                    cfg: Config = CFG, *, alive_by_year=None, year=None, by="Gruppe",
                    panel: pl.DataFrame | None = None, crs=None,
                    raster=None) -> list[Path]:
    """The grid-assignment illustrations: one city-wide, one 3 km zoom where the
    individual 100 m squares are the point."""
    F.set_style()
    fig_dir = cfg.outdir / "figures" / "grid"
    _dirs(fig_dir)
    crs = crs if crs is not None else grid_crs(cfg, raster)
    year = year or max(assignments)
    a = assignments[year]
    stores = _year_stores(stores, alive_by_year, year, panel, by)
    written = []
    for city, ags5 in cfg.cities.items():
        if not (kreis_gdf[cfg.kreis_id].astype(str) == ags5).any():
            print(f"grid_maps: {city} ({ags5}) not in the Kreis layer, skipped")
            continue
        sel = main_area(kreis_gdf[kreis_gdf[cfg.kreis_id].astype(str) == ags5], crs,
                        id_col=cfg.kreis_id)
        for tag, bbox, scale in (
                ("city", F.region_bbox(sel, cfg.kreis_id, ags5, pad_km=3, crs=crs), 5.0),
                ("zoom", F.centre_bbox(sel, cfg.kreis_id, ags5, half_km=3, crs=crs), 1.0)):
            sub, st = grid_view(cells, stores, raster or cfg.ghsl, crs, bbox)
            if sub is None:
                print(f"grid_maps: no cell of this run inside the {city} {tag} window "
                      f"-- a Land-only solve covers only its Land; skipped")
                continue
            g = F.grid_slice(sub, mapping, a, st, bbox=bbox, by=by)
            ax = F.plot_grid(g, stores=None, boundaries=sel, scale_km=scale, crs=crs,
                             title=f"{city} — 100 m grid assigned to nearest store by "
                                   f"drive time\n{F.SPEC_LABEL.get(spec, spec)}, {year}")
            written += F.save(ax.figure, fig_dir / f"grid_{city.lower()}_{tag}_{spec}_{year}")
    print(f"grid maps: {len(written)} files -> {fig_dir}")
    return written


# ==========================================================================
# driver
# ==========================================================================

ORDER = ["stores", "cells", "matrix", "assign", "regions", "cube", "figures"]


def migrate(cfg: Config = CFG, *, apply: bool = False) -> list[tuple[Path, Path]]:
    """Move artefacts from the old layout into snapped/ solve/ derived/.

    Moves only -- nothing is recomputed. A per-spec matrix is carried over only
    when that spec is a single shop type, because only then is its store frame
    the type's frame; `load_matrix` validates that anyway and refuses a mismatch,
    so a wrong guess fails loudly rather than relabelling catchments.
    """
    old_grid = cfg.work / (cfg.grid if not cfg.land else f"{cfg.grid}_{cfg.land}")
    old_net = cfg.work / "network" / cfg.network
    old_out = cfg.outdir
    moves: list[tuple[Path, Path]] = [
        (old_net / "stores_snapped_union.parquet", cfg.snapdir / "stores.parquet"),
        (old_net / "snap_manifest.json", cfg.snapdir / "snap_manifest.json"),
        (old_net / "cache", cfg.snapdir / "cache"),
    ]
    for f in ("cells_solve.parquet", "mapping.parquet", "map_adapt.parquet",
              "map_dedupe.parquet", "node_regions.parquet", "grid_manifest.json"):
        moves.append((old_grid / f, cfg.griddir / f))
    for spec, types in SPECS.items():
        d = old_out / spec
        if len(types) == 1:
            moves.append((d / f"matrix_{spec}.npz", cfg.solvedir(types[0]) / "matrix.npz"))
        for sub in ("assignments", "catchments", "isochrone"):
            moves.append((d / sub, cfg.derivdir(spec) / sub))
    todo = [(a, b) for a, b in moves if a.exists() and not b.exists()]
    for a, b in todo:
        print(f"{'moving ' if apply else 'would move '}{a} -> {b}")
        if apply:
            b.parent.mkdir(parents=True, exist_ok=True)
            a.rename(b)
    skipped = [a for a, b in moves if a.exists() and b.exists()]
    for a in skipped:
        print(f"skipped (target exists): {a}")
    if not apply:
        print(f"\n{len(todo)} move(s). Re-run with --migrate-apply to perform them.")
    return todo


#: stages that belong to the catchment solve; any of them selected => solve
SOLVE_STAGES = {"stores", "cells", "matrix", "assign", "regions"}


def needed_stages(rc: RunConfig) -> set[str]:
    """Which stages a selection actually requires.

    `cube` does not imply the solve. A report on VKFL or Umsatzpotential is a
    filter over the panel, so the expensive stages are scheduled only when a
    demand-side basis (catchment population or area) or a grid map is asked for,
    or when a solve stage is selected outright.
    """
    want = set(rc.stages) | set(rc.force)
    if not want:
        return set()
    if rc.needs_solve() or (want & SOLVE_STAGES):
        last = max(want, key=ORDER.index)
        return set(ORDER[:ORDER.index(last) + 1])
    return want

#: rough cost of recomputing a stage, for the status table and the GUI guard
COST = {"stores": "minutes", "cells": "hours-days", "matrix": "hours",
        "assign": "minutes", "regions": "minutes", "cube": "seconds",
        "figures": "seconds"}
EXPENSIVE = ("cells", "matrix")


def artefacts(cfg: Config = CFG, specs=None) -> pl.DataFrame:
    """What exists on disk, per stage and spec. `ready` is n_present/n_expected.

    Read this before starting anything: it is the difference between a re-run
    that takes seconds and one that re-solves the country.

    Existence only: a matrix listed 1/1 may still be extended on the next run
    (new stores), and an assignment listed 12/12 may still be recomputed (stale
    sidecar). The cube row looks at the non-excluded directory only.
    """
    specs = list(specs or SPECS)
    rows = [("stores", "-", cfg.snapdir / "stores.parquet"),
            ("cells", "-", cfg.griddir / "cells_solve.parquet"),
            ("cells", "-", cfg.griddir / "mapping.parquet"),
            ("regions", "-", cfg.griddir / "node_regions.parquet")]
    out = [dict(stage=st, spec=sp, artefact=p.name, ready=f"{int(p.exists())}/1",
                done=p.exists(), cost=COST[st]) for st, sp, p in rows]
    for t in sorted({x for sp in specs for x in SPECS[sp]}):
        m = _matrix_file(cfg.solvedir(t)) or cfg.solvedir(t) / "matrix.parquet"
        out.append(dict(stage="matrix", spec=t, artefact=f"{slug(t)}/{m.name}",
                        ready=f"{int(m.exists())}/1", done=m.exists(),
                        cost=COST["matrix"]))
    for spec in specs:
        d = cfg.derivdir(spec)
        n = sum((d / "assignments" / f"assignments_{spec}_{y}.parquet").exists()
                for y in cfg.years)
        out.append(dict(stage="assign", spec=spec, artefact="assignments/*.parquet",
                        ready=f"{n}/{len(cfg.years)}", done=n == len(cfg.years),
                        cost=COST["assign"]))
        cd = cube_dir(cfg, spec)
        n = sum(any(cd.glob(f"hhi_*_{y}.parquet")) for y in cfg.years) if cd.exists() else 0
        out.append(dict(stage="cube", spec=spec, artefact=f"cube/{spec}/*",
                        ready=f"{n}/{len(cfg.years)}", done=n == len(cfg.years),
                        cost=COST["cube"]))
    fd = cfg.outdir / "figures"
    nf = len(list(fd.rglob("*.pdf"))) if fd.exists() else 0
    out.append(dict(stage="figures", spec="-", artefact="figures/*.pdf",
                    ready=f"{nf} files", done=nf > 0, cost=COST["figures"]))
    return pl.DataFrame(out)


def run(stages=("all",), specs=None, cfg: Config = CFG, force=(),
        rc: RunConfig | None = None) -> dict:
    """Execute the selected stages.

    `rc` is the authority when given: the pipeline Config is rebuilt from it and
    the stage/spec selection comes from it. That is what makes a run reproducible
    from the config file alone -- the GUI has no other channel into this function.
    """
    # ---- 1. resolve the selection -------------------------------------------
    if rc is not None:
        cfg = rc.pipeline_config(cfg)
        stages = tuple(rc.stages) if tuple(stages) == ("all",) else tuple(stages)
        specs = list(specs or rc.specs)
        force = tuple(force or rc.force)
    else:
        rc = RunConfig(specs=tuple(specs or SPECS), years=tuple(cfg.years),
                       land=cfg.land, osrm=cfg.osrm, grid=cfg.grid,
                       stages=tuple(ORDER if "all" in stages else stages),
                       force=tuple(force))
    # --stages selects what you *want*; dependencies resolve from cache or are
    # computed, so `figures` on an empty output directory solves first.
    want = set(ORDER if "all" in stages else stages)
    force = set(force)
    specs = list(specs or SPECS)
    types = {**SPECS, **rc.shop_types}
    state: dict = {}

    # VKFL and Umsatzpotential are panel facts: no cells, no matrices, no
    # assignments unless a demand-side basis, a grid map or a solve stage asks.
    solve = bool(want & SOLVE_STAGES) or bool(force & SOLVE_STAGES) or rc.needs_solve()
    if not solve:
        print("supply-side report (no catchment population or area, no grid maps): "
              "skipping the solve -- shares come from the panel.\n")

    # ---- 2. inputs ----------------------------------------------------------
    #   panel_raw  every format, every Land: what stage_stores snaps
    #   panel_all  formats used by any spec, every Land: membership, labels
    #   panel      panel_all restricted to the Land: the supply side of the cube
    union = sorted({t for v in types.values() for t in v})
    panel_raw = load_panel(cfg, shop_types=None)       # every format: for the snap
    panel_all = panel_raw.filter(pl.col("Shop Type").is_in(union))
    land = pl.lit(True) if not cfg.land else C.ags5("AGS5").str.starts_with(cfg.land)
    panel = panel_all.filter(land)
    kreis_gdf, bl_gdf = region_layers(cfg, panel_raw)
    if bad := C.unknown_kreise(panel_all, kreis_gdf[cfg.kreis_id]):
        print(f"WARNING {len(bad)} panel Kreis code(s) not in the region layer -- "
              f"their stores count in DE/BL but not in any Kreis: {', '.join(bad[:12])}"
              f"{' ...' if len(bad) > 12 else ''}. Add them to market_cube.KREIS_XWALK.")
    raster = Path(cfg.ghsl) if solve else None

    _dirs(cfg.outdir)
    rc.save(cfg.outdir / "run_config.toml", stamp=False)   # manifest next to outputs
    print(rc.summary())
    print(f"output     {cfg.outdir}\n")

    # ---- 3. solve chain (skipped for supply-only reports) ------------------
    # scfg/spanel are the config and panel the SOLVE uses. They differ from
    # cfg/panel only in a Land run that reuses the national matrices: then the
    # assignments are the national ones (derived/<grid>/, not <grid>_<land>/),
    # while the cube and figures are still written under the Land's outdir.
    stores = cells = c2 = mapping = node_reg = None
    scfg, spanel, solved = cfg, panel, {}
    if solve:
        # national store set -> national grid, always; a Land is a filter on both
        stores = stage_stores(panel_raw, cfg, force="stores" in force)
        cells, c2, mapping = stage_cells(stores, cfg, raster=raster,
                                         force="cells" in force)
        node_reg = stage_regions(cells, mapping, kreis_gdf, cfg, raster=raster,
                                 force="regions" in force)
        needed = sorted({t for sp in specs for t in types.get(sp, SPECS[sp])})
        if cfg.land:
            nat = dataclasses.replace(cfg, land=None)
            if all(_matrix_file(nat.solvedir(t)) for t in needed):
                # the national solve already covers the Land, exactly and with
                # border competition: reuse it (and its cached assignments)
                scfg, spanel = nat, panel_all
                print(f"land {cfg.land}: national matrices exist -- solving on them "
                      f"and reporting the Land; no routing, no border bias")
            else:
                inside = panel["TDLinx"].cast(pl.Utf8).unique()
                stores = stores.filter(pl.col("store_id").is_in(inside.implode()))
                cells, c2, mapping, _ = land_view(cells, c2, mapping, node_reg, cfg)
                print(f"land {cfg.land}: no national matrices -- Land-only solve. Border "
                      f"stores are dropped, so border Kreise lose competitors: smoke "
                      f"test only, never a published number.")
            node_reg = node_reg.filter(pl.col("kreis").str.starts_with(cfg.land))
        for t in needed:
            ever = membership(spanel, [t])["store_id"].unique()
            print(f"\n=== {t}: {ever.len():,} stores traded as this type in some year ===")
            solved[t] = stage_matrix(t, c2, stores, ever, scfg, force="matrix" in force)

    # ---- 4. per spec: assignments -> cube -> grid/catchment maps -----------
    for spec in specs:
        stypes = list(types.get(spec, SPECS[spec]))
        if solve:
            a, alive_by_year = stage_assign(spec, c2, solved, scfg, panel=spanel,
                                            types=stypes, cells=cells, mapping=mapping,
                                            raster=raster, force="assign" in force)
            if a:
                C.coverage(a[max(a)], node_reg)
        else:
            rej = pl.Series(rejected_ids(cfg), dtype=pl.Utf8)
            mem = membership(panel, stypes).filter(~pl.col("store_id").is_in(rej.implode()))
            alive_by_year = {y: mem.filter(pl.col("year") == y)["store_id"].to_list()
                             for y in cfg.years}
            a = None
            print(f"\n=== {spec}: "
                  f"{max((len(v) for v in alive_by_year.values()), default=0):,} "
                  f"stores (panel only) ===")
        if {"cube", "figures"} & want:
            stage_cube(spec, stypes, a, node_reg, panel, cfg, alive_by_year=alive_by_year,
                       actor_panel=spanel, exclude=rc.exclude_actors)
        if "figures" in want and solve and a:
            kw = dict(alive_by_year=alive_by_year, year=rc.year_focus, panel=spanel,
                      raster=raster)
            if "grid_maps" in rc.figures:
                stage_grid_maps(spec, cells, mapping, a, stores, kreis_gdf, cfg, **kw)
            if "map_catchments" in rc.figures:
                stage_catchments(spec, cells, mapping, a, stores, kreis_gdf, cfg, **kw)
        state[spec] = dict(stores=stores, assignments=a, alive=alive_by_year)
        gc.collect()

    # ---- 5. read the whole cube back (validated) -> figures + CSV ----------
    # `state` is returned for notebook use: the in-memory frames of this run.
    if {"cube", "figures"} & want:
        shares, hhi = read_cube(cfg, specs, rc.exclude_actors)
        state.update(shares=shares, hhi=hhi)
        if "figures" in want:
            if cfg.land:                      # maps show the Land, not Germany
                kreis_gdf = kreis_gdf[kreis_gdf["AGS5"].astype(str)
                                      .str.startswith(cfg.land)]
                bl_gdf = bl_gdf[bl_gdf["AGS2"].astype(str) == cfg.land]
            names = {**dict(zip(kreis_gdf["AGS5"].astype(str),
                                kreis_gdf["LABEL"].astype(str))),
                     **dict(zip(bl_gdf["AGS2"].astype(str),
                                bl_gdf["LABEL"].astype(str)))}
            if not cfg.land:
                names["DE"] = "Deutschland"
            stage_figures(shares, hhi, kreis_gdf, bl_gdf, cfg, rc)
            stage_export(shares, hhi, cfg, rc, names)
    state.update(panel=spanel, stores=stores, cells=cells, c2=c2, mapping=mapping,
                 node_reg=node_reg, kreis_gdf=kreis_gdf, bl_gdf=bl_gdf, cfg=cfg,
                 solve_cfg=scfg, rc=rc)
    return state


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config", default=None,
                    help="run configuration TOML (written by the GUI)")
    ap.add_argument("--gui", action="store_true", help="open the configuration GUI")
    ap.add_argument("--stages", default=None,
                    help=f"comma-separated subset of {ORDER}, or 'all'")
    ap.add_argument("--specs", default=None)
    ap.add_argument("--years", default=None, help="e.g. 2014:2025 or 2024,2025")
    ap.add_argument("--land", default=None, help="AGS2 prefix, e.g. 09 for Bayern")
    ap.add_argument("--bases", default=None,
                    help="report bases: pop,area,vkfl,umsatz (vkfl/umsatz alone "
                         "need no catchment solve)")
    ap.add_argument("--scope", default=None, choices=("all", "food"),
                    help="product scope: all (food + non-food) or food only")
    ap.add_argument("--exclude-actors", default=None,
                    help="drop these operators from the market before shares are "
                         "computed; 'other' uses the UNIDENTIFIED list")
    ap.add_argument("--osrm", default=None)
    ap.add_argument("--force", default="", help="stages to recompute even if cached")
    ap.add_argument("--migrate", action="store_true",
                    help="show the moves from the old layout to snapped/solve/derived")
    ap.add_argument("--migrate-apply", action="store_true",
                    help="perform those moves")
    ap.add_argument("--diagnose", action="store_true",
                    help="print per-year panel composition and exit")
    ap.add_argument("--status", action="store_true",
                    help="print what exists on disk and exit")
    ap.add_argument("--save-config", default=None,
                    help="write the effective configuration here and exit")
    a = ap.parse_args(argv)

    if a.gui:
        from scripts.gui import main as gui_main
        return gui_main(a.config)

    # The run name -- and so the output folder -- is `[run] name` INSIDE the
    # TOML, not the file name. The desktop GUI sets it to the file stem when it
    # saves; a TOML copied by hand keeps the name of the file it was copied from
    # and will write into that run's folder. Without --config: name "baseline".
    rc = RunConfig.load(a.config) if a.config else RunConfig(stages=tuple(ORDER))
    if a.years:                                  # 2014:2025 is inclusive at both ends
        if ":" in a.years:
            lo, hi = (int(x) for x in a.years.split(":"))
            rc.years, rc.year_mode = tuple(range(lo, hi + 1)), "range"
        else:
            rc.years = tuple(sorted({int(x) for x in a.years.split(",")}))
            rc.year_mode = "list"
    if a.stages:
        rc.stages = tuple(ORDER) if a.stages == "all" else tuple(a.stages.split(","))
    if a.specs:
        rc.specs = tuple(a.specs.split(","))
    if a.force:
        rc.force = tuple(x for x in a.force.split(",") if x)
    if a.bases:
        rc.bases = tuple(x.strip() for x in a.bases.split(",") if x.strip())
    if a.scope:
        rc.scope = a.scope
    if a.exclude_actors:
        from scripts.runconfig import UNIDENTIFIED
        rc.exclude_actors = (UNIDENTIFIED if a.exclude_actors.strip().lower() == "other"
                             else tuple(x.strip() for x in a.exclude_actors.split(",")
                                        if x.strip()))
    for key, val in (("land", a.land), ("osrm", a.osrm)):
        if val:
            setattr(rc, key, val)

    problems = rc.validate()
    for x in problems:
        print(x if x.startswith("note:") else f"ERROR: {x}")
    if rc.blocking():
        return 2

    if a.save_config:
        print("wrote", rc.save(a.save_config))
        return 0
    cfg = rc.pipeline_config()
    if a.migrate or a.migrate_apply:
        migrate(cfg, apply=a.migrate_apply)
        return 0
    if a.diagnose:
        from scripts.runconfig import UNIDENTIFIED
        types = {**SPECS, **rc.shop_types}
        p_all = load_panel(cfg, shop_types=sorted({t for s_ in rc.specs
                                                   for t in types[s_]}))
        with pl.Config(tbl_rows=30, tbl_cols=20, fmt_str_lengths=24):
            print(C.panel_diagnostics(p_all, unidentified=UNIDENTIFIED))
        return 0
    if a.status:
        with pl.Config(tbl_rows=50, fmt_str_lengths=60):
            print(artefacts(cfg, rc.specs))
        return 0
    run(cfg=cfg, rc=rc)
    return 0


if __name__ == "__main__":
    sys.exit(main())