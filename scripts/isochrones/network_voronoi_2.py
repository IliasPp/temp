"""
Network-Voronoi catchments: GHSL population cells -> nearest store by OSRM drive time.

Built for the German GHS-POP 100 m grid x ~60k stores. Measured on the real data
(2025): 7.27M populated raster cells -> 5.0M solve nodes after the reductions;
60.8k stores snapped. The clipped raster has ~35M cells in all, of which the
7.27M with positive population enter the solve.

This module is the solver library. It knows nothing about specs, years on disk,
Bundesländer or run configs -- scripts/run_pipeline.py drives it. The functions
the pipeline calls, in order:

    load_cells          raster -> populated cell centroids (EPSG:3035 x/y)
    prepare_stores      panel -> one row per store (median position)
    snap                /nearest for every point (cached on disk)
    adapt_resolution    merge cells far from any store        (approximate, bounded)
    dedupe              merge cells that snapped to one point (exact)
    compose             chain the two cell->node maps
    candidates          k nearest stores per node (straight line)
    travel_times        OSRM /table for those pairs           (the expensive step)
    refine_until        widen k where the winner is not yet proven
    extend              add new stores to a solved matrix     (only nearby pairs)
    save/load_matrix    the matrix file (Parquet; legacy .npz still readable)
    assign              per year: nearest store among those trading that year
    catchments          per-store summary of an assignment


WHY THE BATCHES FAILED (history -- kept because it explains snap())
-------------------------------------------------------------------
`radiuses=` was being sent on every /table call. OSRM rejects the *whole*
request with NoSegment if a *single* coordinate has no road inside its radius.
At 100 m GHSL resolution essentially every 900-point batch contains one such
point -- a field, a lake shore, a forest, a raster-edge artifact -- so
essentially every batch fails.

Retrying is pointless (the failure is deterministic) and probing afterwards
costs one /nearest per point in the batch. The fix is to snap once, up front,
and then never send `radiuses`:

    stores = snap(stores)     # 50k calls, seconds. Do this always.
    cells  = snap(cells)      # optional; enables the exact dedupe below.

snap() returns each point's position *on* the network plus how far it moved
(`snap_m`), so a bad geocode becomes a column you can filter instead of an
exception that kills 900 good cells with it. Without `radiuses`, OSRM always
finds some segment, and NoSegment cannot occur.


PIPELINE
--------
    7.27M populated cells (of ~35M raster cells)
      adapt_resolution  coarsen where catchments are wide              approximate, bounded
      snap              move each point onto the network               exact
      dedupe            collapse coincident snapped points  -> 5.0M    exact at tol=0
      candidates        k nearest stores each (kD-tree)
      travel_times      one /table per spatial block          <- the only expensive step
      assign            argmin over stores alive in year t     per year, ~free

Everything before travel_times exists to shrink the OSRM bill, which is the
only cost that matters. Only adapt_resolution trades accuracy for it, and it
tells you exactly how much.


USAGE
-----
    stores = snap(prepare_stores(panel))
    cells  = load_cells("ghsl_de_100m.tif")
    cells, m1 = adapt_resolution(cells, stores)     # 7.27M populated cells
    cells  = snap(cells)
    cells, m2 = dedupe(cells)                       # -> 5.0M solve nodes
    cand   = candidates(cells, stores, k=20)
    dur, _ = travel_times(cells, stores, cand, checkpoint="ckpt/")
    certification_report(cand, dur, ranks=3)
    save_matrix("matrix.parquet", cells, stores, cand, dur)

    a2019 = assign(cells, stores, cand, dur, year=2019)
    catchments(a2019)

Travel times are computed once against the union of stores over all years; a
year's assignment is then an argmin over the alive subset. No re-querying.


OSRM MUST BE STARTED WITH A RAISED TABLE LIMIT
----------------------------------------------
    osrm-routed --algorithm mld --mmap --max-table-size 10000 germany-latest.osrm

The default of 100 rejects every call here. preflight() reports the effective
value (it probes up to 10,000 coordinates; on the project server it measures
5000 x 5000). Install `orjson` -- at 10^8 matrix entries, stdlib JSON parsing is a
measurable share of wall time.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

OSRM = "http://localhost:5000"
EPSG = 3035          # ETRS89-LAEA: metric, equal-area, correct for Germany

# Must exceed every speed OSRM can actually route at. Too high => the sufficiency
# bound is merely conservative (fewer nodes certified, more refine()). Too low =>
# it is UNSOUND and certified() silently lies.
#
# 180 comes from `max_speed_for_map_matching = 180/3.6` in the car.lua in use: a
# hard ceiling the profile cannot exceed. The *routing* speeds are lower -- stock
# car.lua caps `maxspeed=none` at 140 -- which is exactly why 140 was the wrong
# number to use: it is tight against the profile rather than strictly above it,
# so a single way tagged higher, or a profile without the usual speed_reduction,
# breaks the bound with no symptom. 180 costs ~22% of the bound (and so some
# certification rate) and buys one that cannot be wrong.
#
# preflight() measures route-*average* speed, which is always well below the
# maximum link speed. Treat it as a floor, never as confirmation.
V_MAX_KMH = 180.0

try:                                     # 2-5x faster on large /table payloads
    from orjson import loads as _loads
except ImportError:
    from json import loads as _loads


# ==========================================================================
# HTTP
# ==========================================================================

class OSRMError(RuntimeError):
    """Server responded but rejected the request. Never worth a blind retry."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"OSRM {code}: {message}")
        self.code = code


def _coords(pts: np.ndarray) -> str:
    return ";".join(f"{x:.6f},{y:.6f}" for x, y in pts)


def _get(session, url: str, params: dict, timeout: float = 300.0) -> dict:
    r = session.get(url, params=params, timeout=timeout)
    try:
        payload = _loads(r.content)
    except ValueError:
        r.raise_for_status()             # not JSON -> genuine transport failure
        raise
    if payload.get("code") != "Ok":
        raise OSRMError(payload.get("code", "Unknown"), payload.get("message", ""))
    return payload


def _matrix(payload: dict, key: str) -> np.ndarray | None:
    """Rows -> float32 array. Fast C path when no pair is unreachable."""
    raw = payload.get(key)
    if raw is None:
        return None
    try:
        return np.asarray(raw, dtype=np.float32)
    except (TypeError, ValueError):      # some entry is null (unreachable)
        return np.array(
            [[np.inf if v is None else v for v in row] for row in raw],
            dtype=np.float32,
        )


def _parallel(n: int, work, *, max_workers: int, label: str, every: int = 0):
    """Run `work(lo, hi)` over contiguous index ranges on a thread pool.

    Contiguous ranges rather than one task per item: 5M futures would blow up
    memory, and a per-worker Session keeps the connection alive across a whole
    range instead of reconnecting per request.
    """
    import concurrent.futures as cf

    parts = max(1, min(n, max_workers * 8))
    edges = np.linspace(0, n, parts + 1).astype(np.int64)
    done = 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(work, int(a), int(b))
                for a, b in zip(edges[:-1], edges[1:]) if b > a]
        for i, f in enumerate(cf.as_completed(futs), 1):
            f.result()
            done = i
            if every and i % every == 0:
                print(f"  {label}: {i}/{len(futs)} chunks", flush=True)
    return done


# ==========================================================================
# 1. Inputs
# ==========================================================================

def load_cells(
    raster: str | Path, *, band: int = 1, min_pop: float = 1e-6, epsg: int = EPSG
) -> pl.DataFrame:
    """Populated cell centroids from a mosaicked, clipped GHSL raster.

    Returns cell_id, row, col, pop, x, y (metric, epsg), lon, lat (WGS84).

    The raster is never reprojected -- resampling a population *count* is lossy.
    Only the centroids are transformed.
    """
    import rasterio
    from pyproj import Transformer

    with rasterio.open(raster) as src:
        arr = src.read(band, masked=True)
        T, crs = src.transform, src.crs

    v = np.ma.filled(arr.astype(np.float32), 0.0)
    np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    flat = np.flatnonzero(v.ravel() > min_pop)
    if flat.size == 0:
        raise ValueError(f"load_cells: no cell above min_pop={min_pop} in {raster}")

    row, col = np.divmod(flat, v.shape[1])
    pop = v.ravel()[flat]
    # Cell centres straight from the affine; rasterio.transform.xy is far too
    # slow at 10^7 points.
    cf_, rf = col + 0.5, row + 0.5
    xs = T.a * cf_ + T.b * rf + T.c
    ys = T.d * cf_ + T.e * rf + T.f

    x, y = Transformer.from_crs(crs, f"EPSG:{epsg}", always_xy=True).transform(xs, ys)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)

    print(f"load_cells: {flat.size:,} populated cells, {pop.sum():,.0f} people")
    return pl.DataFrame({
        "cell_id": np.arange(flat.size, dtype=np.int64),
        "row": row.astype(np.int32), "col": col.astype(np.int32),
        "pop": pop,
        "x": np.asarray(x), "y": np.asarray(y),
        "lon": np.asarray(lon), "lat": np.asarray(lat),
    })


def prepare_stores(
    panel: pl.DataFrame, *, id_col: str = "TDLinx", lat_col: str = "lat",
    lon_col: str = "lon", year_col: str | None = "year", epsg: int = EPSG,
    move_warn_m: float = 100.0, keep: Sequence[str] = (),
) -> pl.DataFrame:
    """Collapse a long store x year panel to one row per store.

    Four things that must not be skipped:

    1. Deduplicate. A panel fed straight to the kD-tree indexes each store once
       per observed year, so query(k=20) returns the same store in 12 slots, the
       candidate set collapses to 2-3 distinct outlets, and nothing raises --
       the catchments are just wrong.
    2. Derive first_year/last_year; the panel is the only place entry and exit
       are observable.
    3. Project to metres.
    4. Flag relocations. Position is the per-store median; stores whose observed
       coordinates span more than move_warn_m are either re-geocoded or actually
       moved, and the latter means their catchment is not constant over the panel.

    `keep` carries extra store attributes through -- e.g. keep=("Gruppe",
    "Shop Type") -- so downstream chain-level metrics have something to group on.
    The most recent observed value is taken, and any attribute that is *not*
    constant within a store is reported: fine for a banner switch you know about,
    a data problem otherwise.

    Only genuinely store-level attributes belong here. Time-varying ones (VKFL,
    Umsatzpotential) must not be collapsed to one value -- join them per year
    with `store_year_attrs`, or a 2014 sales floor silently becomes the 2025
    attractiveness weight.
    """
    from pyproj import Transformer

    missing = [c for c in (id_col, lat_col, lon_col) if c not in panel.columns]
    if missing:
        raise ValueError(f"prepare_stores: missing column(s) {missing}")

    n_in = panel.height
    df = panel.filter(pl.col(lat_col).is_not_null() & pl.col(lon_col).is_not_null())
    if n_in - df.height:
        print(f"prepare_stores: dropped {n_in - df.height:,} null-coordinate row(s)")

    aggs = [pl.col(lat_col).median().alias("lat"), pl.col(lon_col).median().alias("lon"),
            (pl.col(lat_col).max() - pl.col(lat_col).min()).alias("_dlat"),
            (pl.col(lon_col).max() - pl.col(lon_col).min()).alias("_dlon")]
    has_year = year_col is not None and year_col in panel.columns
    if has_year:
        aggs += [pl.col(year_col).min().cast(pl.Int16).alias("first_year"),
                 pl.col(year_col).max().cast(pl.Int16).alias("last_year")]

    missing_keep = [c for c in keep if c not in panel.columns]
    if missing_keep:
        raise ValueError(f"prepare_stores: keep column(s) {missing_keep} not in panel")
    if has_year and keep:
        df = df.sort(year_col)          # so .last() is the most recent, not arbitrary
    for c in keep:
        aggs += [pl.col(c).last().alias(c), pl.col(c).n_unique().alias(f"_nu_{c}")]

    out = df.group_by(id_col).agg(aggs).rename({id_col: "store_id"})
    for c in keep:
        n_var = out.filter(pl.col(f"_nu_{c}") > 1).height
        if n_var:
            print(f"prepare_stores: {n_var:,} store(s) have >1 distinct {c!r} across "
                  "the panel; most recent kept. Expected for banner switches.")
    out = out.drop([f"_nu_{c}" for c in keep])
    if not has_year:
        print("prepare_stores: no year column -- every store treated as always alive")
        out = out.with_columns(pl.lit(0, pl.Int16).alias("first_year"),
                               pl.lit(9999, pl.Int16).alias("last_year"))

    span = ((pl.col("_dlat") * 111_320.0).pow(2)
            + (pl.col("_dlon") * 111_320.0
               * (pl.col("lat") * math.pi / 180.0).cos()).pow(2)).sqrt()
    out = out.with_columns(span.alias("coord_span_m")).drop("_dlat", "_dlon")
    n_moved = out.filter(pl.col("coord_span_m") > move_warn_m).height
    if n_moved:
        print(f"prepare_stores: {n_moved:,} store(s) move >{move_warn_m:.0f}m across "
              "the panel; median position used. Inspect via coord_span_m.")

    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tf.transform(out["lon"].to_numpy(), out["lat"].to_numpy())
    out = out.with_columns(pl.Series("x", np.asarray(x)), pl.Series("y", np.asarray(y)))

    print(f"prepare_stores: {n_in:,} panel rows -> {out.height:,} unique stores")
    # sort: group_by order is not stable across runs, and every downstream array
    # (cand.store_idx, the travel-time matrix, blocks) indexes stores POSITIONALLY.
    # An unsorted frame silently re-labels every column of a saved matrix.
    return (out.sort("store_id")
            .select("store_id", "lon", "lat", "x", "y",
                    "first_year", "last_year", "coord_span_m", *keep))


def store_year_attrs(
    panel: pl.DataFrame, year: int, cols: Sequence[str], *,
    id_col: str = "TDLinx", year_col: str = "year",
) -> pl.DataFrame:
    """That year's values of time-varying attributes, one row per store.

    For anything that moves over the panel (VKFL, Umsatzpotential) rather than
    collapsing it in prepare_stores::

        st24 = stores.join(store_year_attrs(panel, 2024, ["VKFL"]), on="store_id",
                           how="left")

    A Huff weight built from the wrong year's sales floor is a silent error: it
    yields perfectly plausible shares that are not the shares of the year you
    are reporting.
    """
    miss = [c for c in cols if c not in panel.columns]
    if miss:
        raise ValueError(f"store_year_attrs: {miss} not in panel")
    sub = panel.filter(pl.col(year_col) == year)
    if sub.height == 0:
        raise ValueError(f"store_year_attrs: no rows for year={year}")
    return (sub.group_by(id_col)
            .agg([pl.col(c).median() if sub.schema[c].is_numeric() else pl.col(c).last()
                  for c in cols])
            .rename({id_col: "store_id"}).sort("store_id"))


# ==========================================================================
# 2. Snapping -- the fix for the failing batches
# ==========================================================================

def snap(
    df: pl.DataFrame, *, base_url: str = OSRM, max_workers: int = 32,
    timeout: float = 10.0, epsg: int = EPSG, drop_over_m: float | None = None,
    cache_path: str | Path | None = None, flush_every: int = 200_000,
    fail_fast: int = 500,
) -> pl.DataFrame:
    """Move every point onto the road network. Replaces lon/lat/x/y, adds snap_m.

    One /nearest per point. Do this for stores always -- a store that will not
    snap is a bad geocode, and because a store is a candidate for thousands of
    cells, that one bad row corrupts a whole region. Do it for cells too if you
    want dedupe() (which needs positions to actually coincide) and a measured
    distribution of snap error.

    After this, /table calls carry no `radiuses` and cannot return NoSegment.

    drop_over_m removes points that had to travel further than that to reach a
    road. Leave it None for cells (a 3 km snap in a rural area is legitimate);
    set it around 300 for stores, where it is a data-quality flag.

    CACHING. Snapping is a one-time cost, not a per-run one -- unlike
    travel_times(), which is expensive by nature, snap() should never be the
    slow step in a repeat run. Pass `cache_path` (a directory) and:

      - if a complete cached result exists there for this exact input (matched
        by row count + a hash of the coordinates, so a changed store list or a
        re-geocode misses the cache instead of silently reusing stale snaps),
        it is loaded and returned with no network calls at all;
      - otherwise the run proceeds and flushes progress to that directory every
        `flush_every` points, and a re-run (crash, Ctrl+C, or a fresh process)
        resumes from whatever's on disk instead of starting over.

    Flushes are written to a temp file and renamed, so an interrupted write cannot
    leave a half-written cache for the next run to load.

    `fail_fast` aborts after that many CONSECUTIVE request failures. A point with
    no road near it is a legitimate miss and is simply left unsnapped; a server
    that is down, or that dies mid-run, produces the same per-point exception, and
    without this the two are indistinguishable until the very end of a multi-hour
    pass. Checking the server is up beforehand does not cover it -- the failure
    this catches is the one that happens at minute 40.
    """
    import requests
    from pyproj import Transformer

    lon0 = df["lon"].to_numpy()
    lat0 = df["lat"].to_numpy()
    n = lon0.size
    # Cache key: point count + rounded coordinate sums. hash() of ints/floats is
    # stable across processes (only str/bytes hashing is salted), so the file
    # name is reproducible. The key is coarse: a re-geocode moving a few points
    # by metres can keep the same key and reuse old snaps. The pipeline does not
    # depend on it for that -- stores.parquet is topped up by store id, and the
    # grid is frozen -- but delete snap_*.npz if you ever re-snap moved points.
    digest = hash((n, round(float(lon0[:1000].sum()), 3),
                   round(float(lat0[:1000].sum()), 3), round(float(lon0.sum()), 1)))
    cache = Path(cache_path) / f"snap_{n}_{digest & 0xffffffff:08x}.npz" if cache_path else None

    lon, lat, snap_m = lon0.copy(), lat0.copy(), np.full(n, np.nan)
    have = np.zeros(n, bool)
    if cache is not None and cache.exists():
        z = np.load(cache)
        if z["done"].all():
            print(f"snap: loaded complete cache {cache.name}, 0 requests made")
            lon, lat, snap_m = z["lon"], z["lat"], z["snap_m"]
            have[:] = True
        else:
            lon, lat, snap_m, have = z["lon"], z["lat"], z["snap_m"], z["done"]
            print(f"snap: resuming from cache, {have.sum():,}/{n:,} already snapped")

    todo_mask = ~have
    if todo_mask.any():
        url = base_url.rstrip("/") + "/nearest/v1/driving/"
        local = threading.local()
        lock = threading.Lock()
        since_flush = [0]
        n_fail = [0]                     # consecutive failures across all workers
        n_ok = [0]

        def flush():
            """Atomic: a crash mid-write must not leave a truncated cache that the
            next run silently half-loads."""
            if cache is None:
                return
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(cache.name + ".tmp.npz")
            with lock:                   # consistent snapshot, then write unlocked
                snapshot = (lon.copy(), lat.copy(), snap_m.copy(), have.copy())
            np.savez(tmp, lon=snapshot[0], lat=snapshot[1], snap_m=snapshot[2],
                     done=snapshot[3])
            tmp.replace(cache)

        def run(lo: int, hi: int):
            s = getattr(local, "s", None)
            if s is None:
                s = local.s = requests.Session()
            for i in range(lo, hi):
                if have[i]:
                    continue
                do_flush = False
                try:
                    j = _get(s, f"{url}{lon0[i]:.6f},{lat0[i]:.6f}", {"number": 1}, timeout)
                    w = j["waypoints"][0]
                    lon[i], lat[i] = w["location"]
                    snap_m[i] = w.get("distance", np.nan)
                except Exception as exc:
                    # A point with no road anywhere near it is a legitimate miss and
                    # stays have[i]=False. A dead or restarting server is not, and
                    # without this it looks identical: millions of points "fail" one
                    # at a time, in silence, and you find out an hour later.
                    with lock:
                        n_fail[0] += 1
                        if n_fail[0] >= fail_fast and n_ok[0] == 0:
                            raise RuntimeError(
                                f"snap: {n_fail[0]:,} consecutive failures and no "
                                f"successes -- OSRM at {base_url} is not answering. "
                                f"Last error: {type(exc).__name__}: {exc}"
                            ) from exc
                        if n_fail[0] >= fail_fast:
                            raise RuntimeError(
                                f"snap: {n_fail[0]:,} consecutive failures after "
                                f"{n_ok[0]:,} successes -- the server probably died "
                                f"mid-run. Progress is cached; re-run to resume. "
                                f"Last error: {type(exc).__name__}: {exc}"
                            ) from exc
                else:
                    with lock:
                        have[i] = True
                        n_fail[0] = 0
                        n_ok[0] += 1
                        since_flush[0] += 1
                        if since_flush[0] >= flush_every:
                            since_flush[0] = 0
                            do_flush = True
                if do_flush:
                    flush()              # outside the lock: this is a ~100 MB write

        t0 = time.perf_counter()
        try:
            _parallel(n, run, max_workers=max_workers, label="snap",
                      every=max(1, max_workers))
        finally:
            flush()                      # keep partial progress even on abort
        print(f"snap: {int(todo_mask.sum()):,} point(s) requested in "
              f"{time.perf_counter() - t0:.0f}s")

    ok = have
    if ok.any():
        q = np.percentile(snap_m[ok], [50, 95, 99.9])
        print(f"  snap distance m: p50={q[0]:.0f} p95={q[1]:.0f} p99.9={q[2]:.0f} "
              f"max={snap_m[ok].max():.0f}")
    if (~ok).any():
        print(f"  WARNING {int((~ok).sum()):,} point(s) failed to snap at all")

    tf = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tf.transform(lon, lat)
    # Snapped position replaces lon/lat/x/y because routing, dedupe and the
    # certification bound must all use the point OSRM actually starts from. The
    # original is kept as *_orig: it is where the population really is, and it is
    # what maps should show -- plotting snapped cells draws people strung along
    # roads with holes in between, which is an artefact of the method, not a fact
    # about settlement.
    keep_orig = {}
    if "x_orig" not in df.columns:          # don't overwrite on a second snap()
        keep_orig = {
            "lon_orig": df["lon"].to_numpy(), "lat_orig": df["lat"].to_numpy(),
            "x_orig": df["x"].to_numpy() if "x" in df.columns else None,
            "y_orig": df["y"].to_numpy() if "y" in df.columns else None,
        }
        if keep_orig["x_orig"] is None:     # unprojected input: project the original
            ox, oy = tf.transform(keep_orig["lon_orig"], keep_orig["lat_orig"])
            keep_orig["x_orig"], keep_orig["y_orig"] = np.asarray(ox), np.asarray(oy)
    out = df.with_columns(
        *[pl.Series(k, v) for k, v in keep_orig.items()],
        pl.Series("lon", lon), pl.Series("lat", lat),
        pl.Series("x", np.asarray(x)), pl.Series("y", np.asarray(y)),
        pl.Series("snap_m", snap_m),
    )
    if drop_over_m is not None:
        keep = out.filter(pl.col("snap_m").is_not_nan()
                          & (pl.col("snap_m") <= drop_over_m))
        if keep.height < out.height:
            print(f"  dropped {out.height - keep.height:,} point(s) snapping "
                  f">{drop_over_m:.0f}m")
        out = keep
    return out


# ==========================================================================
# 3. Reductions -- shrink the OSRM bill before spending it
# ==========================================================================

def adapt_resolution(
    cells: pl.DataFrame, stores: pl.DataFrame, *, cells_per_catchment: int = 8,
    max_level: int = 3, base_m: float = 100.0, epsg: int = EPSG,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Coarsen cells where catchments are wide. Returns (cells, mapping).

    Resolution should scale with the local catchment, not be uniform. Where the
    nearest store is 300 m away, 100 m cells are the right granularity. Where it
    is 8 km away, they are 80x finer than anything the assignment can resolve --
    pure OSRM cost for no information.

    Each cell gets level L = clip(log2(d1 / (cells_per_catchment * base_m)), 0,
    max_level) and merges into its 2^L x 2^L parent block, summing population and
    taking the population-weighted centroid. Centroid error is bounded by half a
    parent cell, i.e. at most d1 / (2 * cells_per_catchment) -- always a fixed
    small fraction of the distance the assignment is actually deciding.

    Population is conserved exactly. Contrast with dropping low-population cells:
    that discards population which genuinely belongs to some catchment, and
    because sparse cells are overwhelmingly rural the loss falls on rural stores,
    biasing exactly the concentration measures you are trying to estimate.
    """
    from pyproj import Transformer
    from scipy.spatial import cKDTree

    tree = cKDTree(np.column_stack([stores["x"].to_numpy(), stores["y"].to_numpy()]))
    d1, _ = tree.query(np.column_stack([cells["x"].to_numpy(),
                                        cells["y"].to_numpy()]), k=1, workers=-1)
    level = np.clip(
        np.floor(np.log2(np.maximum(d1 / (cells_per_catchment * base_m), 1.0))),
        0, max_level,
    ).astype(np.int16)
    f = (1 << level).astype(np.int32)

    keep = ["cell_id", "row", "col", "pop", "x", "y", "lon", "lat", "level", "n_src"]
    if "x_orig" in cells.columns:
        keep += ["x_orig", "y_orig"]
    df = cells.with_columns(pl.Series("level", level), pl.Series("_f", f))
    fine = (df.filter(pl.col("level") == 0)
              .with_columns(pl.lit(1, pl.UInt32).alias("n_src")).select(keep))
    rest = df.filter(pl.col("level") > 0)

    if rest.height == 0:
        m = cells.select(pl.col("cell_id").alias("src"), pl.col("cell_id").alias("dst"))
        return fine, m

    agg = (
        rest.group_by(
            pl.col("level"),
            (pl.col("row") // pl.col("_f")).alias("_pr"),
            (pl.col("col") // pl.col("_f")).alias("_pc"),
        ).agg(
            pl.col("pop").sum().alias("pop"),
            ((pl.col("x") * pl.col("pop")).sum() / pl.col("pop").sum()).alias("x"),
            ((pl.col("y") * pl.col("pop")).sum() / pl.col("pop").sum()).alias("y"),
            pl.col("row").min().alias("row"), pl.col("col").min().alias("col"),
            *_orig_centroid(cells),
            pl.col("cell_id").alias("_members"), pl.len().cast(pl.UInt32).alias("n_src"),
        ).filter(pl.col("pop") > 0)
    )
    # merged blocks get NEW cell_ids above every raster id, so the output mixes
    # untouched raster ids (level 0) and block ids; `mapping` records which
    # raster cells each block absorbed
    offset = int(cells["cell_id"].max()) + 1
    # sort before numbering: group_by order is not stable, so without this the same
    # input yields different cell_ids each run and any saved mapping/matrix mismatches
    agg = (agg.sort("level", "_pr", "_pc")
           .with_row_index("cell_id", offset=offset)
           .with_columns(pl.col("cell_id").cast(pl.Int64)))

    tf = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = tf.transform(agg["x"].to_numpy(), agg["y"].to_numpy())
    agg = agg.with_columns(pl.Series("lon", np.asarray(lon)),
                           pl.Series("lat", np.asarray(lat)))

    out = pl.concat([fine, agg.select(keep)], how="vertical")
    mapping = pl.concat([
        fine.select(pl.col("cell_id").alias("src"), pl.col("cell_id").alias("dst")),
        agg.select(pl.col("_members").alias("src"),
                   pl.col("cell_id").alias("dst")).explode("src"),
    ], how="vertical")

    pi, po = cells["pop"].sum(), out["pop"].sum()
    print(f"adapt_resolution: {cells.height:,} -> {out.height:,} cells "
          f"({out.height / cells.height:.1%}); population delta "
          f"{abs(po - pi) / pi:.1e}")
    return out, mapping


def _orig_centroid(cells: pl.DataFrame) -> list:
    """Population-weighted centroid of the pre-snap positions, if they were kept.

    Display coordinates for a merged node: it is drawn where its people are, not
    at whichever road segment OSRM happened to snap the group to.
    """
    if "x_orig" not in cells.columns:
        return []
    w = pl.col("pop").cast(pl.Float64)
    return [((pl.col("x_orig") * w).sum() / w.sum()).alias("x_orig"),
            ((pl.col("y_orig") * w).sum() / w.sum()).alias("y_orig")]


def dedupe(
    cells: pl.DataFrame, *, tol_m: float = 0.0
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collapse points OSRM cannot tell apart into one solve node.

    After snap(), many cells sit at the same position on the same street -- an
    apartment block, a village lane, a whole terrace. OSRM returns bit-identical
    durations for them, so routing each separately is pure waste. At tol_m=0 this
    is *exact*: no approximation, just not paying twice for the same answer.

    tol_m > 0 rounds positions to a grid first. That is approximate, but the
    error is bounded by tol_m: 50 m is under 4 seconds of driving, against
    catchments measured in tens of minutes.

    Only meaningful after snap(). On raw centroids nothing coincides.
    """
    if "snap_m" not in cells.columns:
        print("dedupe: cells were not snapped; positions will rarely coincide")
    step = tol_m if tol_m > 0 else 1e-6
    node = (
        cells.with_columns(((pl.col("x") / step).round() * step).alias("_gx"),
                           ((pl.col("y") / step).round() * step).alias("_gy"))
        .group_by("_gx", "_gy")
        .agg(pl.col("pop").sum().alias("pop"),
             pl.col("lon").first().alias("lon"), pl.col("lat").first().alias("lat"),
             pl.col("x").first().alias("x"), pl.col("y").first().alias("y"),
             *_orig_centroid(cells),
             pl.col("cell_id").alias("_members"), pl.len().cast(pl.UInt32).alias("n_src"))
        .sort("_gx", "_gy")                       # deterministic node numbering
        # NOTE: `cell_id` here is the solve-node id (0..n_nodes-1), a different
        # id space from the raster cell_id it is computed from
        .with_row_index("cell_id").with_columns(pl.col("cell_id").cast(pl.Int64))
    )
    mapping = node.select(pl.col("_members").alias("src"),
                          pl.col("cell_id").alias("dst")).explode("src")
    cols = ["cell_id", "pop", "x", "y", "lon", "lat", "n_src"]
    if "x_orig" in cells.columns:
        cols += ["x_orig", "y_orig"]
    out = node.select(cols)
    print(f"dedupe(tol={tol_m:.0f}m): {cells.height:,} -> {out.height:,} nodes "
          f"({out.height / cells.height:.1%})")
    return out, mapping


def compose(*maps: pl.DataFrame, strict: bool = True) -> pl.DataFrame:
    """Chain (src, dst) mappings so original cell_id -> final solve node.

    A left join is used so a gap is visible rather than silently dropped, and
    with strict=True (the default) a gap raises. This matters because the composed
    mapping is the ONLY thing tying original population to solve nodes: an
    original cell whose dst is null contributes nothing to any catchment, any
    market share or any HHI, and every join downstream is an inner join, so the
    population simply is not in the totals. Nothing else in the pipeline would
    report it.

    A gap means one of the reduction steps did not emit a row for every input --
    e.g. maps passed in the wrong order, or a mapping regenerated against a
    different `cells` frame than the one that produced the next map.
    """
    m = maps[0]
    for i, nxt in enumerate(maps[1:], 1):
        m = (m.join(nxt.rename({"src": "dst", "dst": "_d2"}), on="dst", how="left")
               .select("src", pl.col("_d2").alias("dst")))
        n_null = int(m.select(pl.col("dst").is_null().sum()).item())
        if n_null:
            msg = (f"compose: {n_null:,}/{m.height:,} source cell(s) have no node "
                   f"after map {i}; their population would vanish from every metric.")
            if strict:
                raise ValueError(msg + " Check the maps are in pipeline order and "
                                       "were produced by the same cells frame.")
            print("WARNING " + msg)
    return m


# ==========================================================================
# 4. Candidates and blocks
# ==========================================================================

@dataclass
class Candidates:
    """k Euclidean-nearest stores per node, ascending by distance."""
    store_idx: np.ndarray    # (n, k) int32, positional index into stores
    dist_m: np.ndarray       # (n, k) float32, straight-line metres
    # store_id of the frame this was built against, in frame order. Carried so a
    # reordered store frame is caught rather than silently relabelling every
    # catchment: a left-join onto `stores` (e.g. attaching VKFL) preserves height
    # but Polars does not guarantee it preserves order, and height is all the
    # shape checks can see.
    store_ids: np.ndarray | None = None

    @property
    def k(self) -> int:
        return self.store_idx.shape[1]


def _check_aligned(cells: pl.DataFrame, stores: pl.DataFrame, cand: Candidates,
                   dur: np.ndarray | None = None, *, what: str = "") -> None:
    """Assert cells/stores/cand/dur describe the same solve.

    `cand.store_idx` is (n_nodes, k) of POSITIONAL store indices, so all four
    objects are one artefact split across four variables. Mixing a frame from one
    run with a matrix from another is the most common failure here -- usually a
    stale matrix.npz next to regenerated cells -- and by default numpy either
    raises something opaque about column heights or, when the shapes happen to
    line up, silently produces catchments for the wrong stores.
    """
    n, k = cand.store_idx.shape
    if cells.height != n:
        raise ValueError(
            f"{what}: cells has {cells.height:,} rows but the candidate matrix has "
            f"{n:,}. These are from different runs -- most often a saved matrix.npz "
            f"loaded next to regenerated cells. Either reload the cells frame that "
            f"matrix was built from, or delete the matrix and re-solve. "
            f"Use nv.load_matrix(path, stores=..., cells=...) to catch this at load."
        )
    if dur is not None and dur.shape != cand.store_idx.shape:
        raise ValueError(
            f"{what}: durations {dur.shape} do not match candidates "
            f"{cand.store_idx.shape}; they must come from the same travel_times call."
        )
    hi = int(cand.store_idx.max()) if cand.store_idx.size else -1
    if hi >= stores.height:
        raise ValueError(
            f"{what}: candidate matrix references store index {hi:,} but stores has "
            f"only {stores.height:,} rows. The store frame is not the one this matrix "
            f"was built from -- rebuild it with the same prepare_stores(...) call and "
            f"filters, or re-solve."
        )
    if cand.store_ids is not None:
        now = stores["store_id"].to_numpy().astype(str)
        if not np.array_equal(cand.store_ids.astype(str), now):
            bad = int((cand.store_ids.astype(str) != now).sum()) if now.size == \
                cand.store_ids.size else -1
            raise ValueError(
                f"{what}: the store frame has the right height but not the right "
                f"ORDER ({bad if bad >= 0 else 'n/a'} position(s) differ). "
                f"cand.store_idx holds positional indices, so this would relabel "
                f"catchments with no other symptom. This usually means a frame that "
                f"went through a join (e.g. stores.join(store_year_attrs(...))) was "
                f"passed where the original prepare_stores() frame belongs. Pass the "
                f"original frame to candidates/travel_times/assign, or add "
                f"maintain_order='left' to the join."
            )


def candidates(
    cells: pl.DataFrame, stores: pl.DataFrame, *, k: int = 20, chunk: int = 2_000_000
) -> Candidates:
    """k nearest stores per node in the projected CRS.

    Simpler than a Delaunay 1-ring and, unlike it, yields the *ordered* distances
    that certified() needs for its lower bound.

    CHOOSING k. Certification needs d_k >= v_max * t_(rank). With a detour
    factor of 1.35 at 55 km/h and v_max = 140, that means d_k >~ 3.4 * d_rank,
    and since d_j ~ sqrt(j) under locally-uniform store density, k grows roughly
    linearly in rank. Measured on a uniform 300-store layout:

        k     rank1   rank2   rank3
        20    83.2%   54.2%   29.2%
        30    94.3%   77.6%   56.1%
        40    98.0%   90.1%   76.9%
        60    99.6%   98.0%   95.1%

    So k=20 is a reasonable rank-1 default, but k=20 with ranks=3 leaves ~70%
    of nodes uncertified -- at millions of nodes that sends most of the job through
    refine() a second time. If you need ranks 2-3 as a Huff choice set, budget
    k=60 and run the reductions first; if you only need the headline catchment,
    use ranks=1 and keep k=20. Real store layouts are more clustered than this
    benchmark, so treat the table as an order of magnitude and read the actual
    rate off certification_report().
    """
    from scipy.spatial import cKDTree

    for df, need, what in ((stores, ("store_id", "x", "y"), "stores"),
                           (cells, ("cell_id", "x", "y"), "cells")):
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"candidates: {what} missing {miss}; got {list(df.columns)}")
    dup = stores.height - stores["store_id"].n_unique()
    if dup:
        raise ValueError(f"candidates: {dup:,} duplicate store_id(s). Run prepare_stores "
                         "first -- a long panel indexes each store once per year.")

    sxy = np.column_stack([stores["x"].to_numpy(), stores["y"].to_numpy()])
    if not np.isfinite(sxy).all():
        raise ValueError("candidates: non-finite store coordinates")
    k = min(k, sxy.shape[0])
    tree = cKDTree(sxy)

    cxy = np.column_stack([cells["x"].to_numpy(), cells["y"].to_numpy()])
    n = cxy.shape[0]
    idx = np.empty((n, k), np.int32)
    dst = np.empty((n, k), np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        d, i = tree.query(cxy[a:b], k=k, workers=-1)
        if k == 1:
            d, i = d[:, None], i[:, None]
        idx[a:b], dst[a:b] = i, d
    return Candidates(idx, dst, stores["store_id"].to_numpy().astype(str))


def blocks(
    cand: Candidates, stores: pl.DataFrame, *, target: int = 3000
) -> list[np.ndarray]:
    """Partition nodes into blocks of ~`target`, one /table call each.

    Two competing pressures. Large source blocks amortise the HTTP round trip and
    let OSRM's many-to-many search share its forward exploration across sources.
    Small blocks keep the destination union tight, so less of the matrix is
    computed and thrown away.

    Euclidean-Voronoi groups are the natural atom -- members share almost the
    same candidate set -- but with 50k stores most are far below target, so they
    are merged in spatial order (20 km west-east strips) to keep merged blocks
    geographically compact and their destination unions small.
    """
    order_of_store = np.empty(stores.height, np.int64)
    order_of_store[np.lexsort((stores["x"].to_numpy(),
                               np.floor(stores["y"].to_numpy() / 20_000)))] = \
        np.arange(stores.height)

    key = order_of_store[cand.store_idx[:, 0]]
    order = np.argsort(key, kind="stable")
    cuts = np.flatnonzero(np.diff(key[order])) + 1

    out: list[np.ndarray] = []
    buf: list[np.ndarray] = []
    n = 0
    for g in np.split(order, cuts):
        buf.append(g)
        n += g.size
        if n >= target:
            out.append(np.concatenate(buf))
            buf, n = [], 0
    if buf:
        out.append(np.concatenate(buf))
    return out


# ==========================================================================
# 5. The solve
# ==========================================================================

def _fingerprint(cand: Candidates, n_stores: int) -> np.ndarray:
    """Cheap content hash of a candidate set, written into every checkpoint part.

    Checkpoints store *positional* row indices, so replaying a part file against a
    different candidate set writes one specification's drive times into another's
    matrix. When the new run has fewer nodes that raises IndexError; when it has
    more, it silently succeeds and corrupts the result. Neither is acceptable, so
    every part carries this and mismatched parts are refused on load.
    """
    a = cand.store_idx
    step = max(1, a.shape[0] // 4096)                 # sample, don't hash 100M ints
    return np.array([a.shape[0], a.shape[1], n_stores,
                     int(a[::step].astype(np.int64).sum()),
                     int(a[-1].astype(np.int64).sum())], dtype=np.int64)


def travel_times(
    cells: pl.DataFrame, stores: pl.DataFrame, cand: Candidates, *,
    base_url: str = OSRM, block: int = 3000, max_coords: int = 1000,
    with_distance: bool = False, max_workers: int = 8, checkpoint: str | Path | None = None,
    flush_every: int = 50, max_retries: int = 3, progress_every: int = 200,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fill an (n, k) array of drive times aligned to cand.store_idx.

    No `radiuses` is sent, so NoSegment cannot occur. If OSRM returns one anyway,
    that is a real problem (an empty extract, a broken profile) and it is raised
    rather than papered over. Only transport failures -- timeouts, connection
    resets, 5xx -- are retried, with backoff.

    Memory: n x k float32. 5M x 20 is 400 MB; 7.27M x 20 unreduced is 0.6 GB, which
    is one more reason to run the reductions first.

    With `checkpoint` set, completed blocks are appended as new part files (never
    read-concat-rewrite) and a re-run solves only the rows still missing. Blocks
    are derived deterministically from `cand`, so resume is valid only against an
    identical candidate set.
    """
    import requests

    _check_aligned(cells, stores, cand, what="travel_times")
    n, k = cand.store_idx.shape
    dur = np.full((n, k), np.inf, np.float32)
    dist = np.full((n, k), np.inf, np.float32) if with_distance else None
    have = np.zeros(n, bool)

    cxy = np.column_stack([cells["lon"].to_numpy(), cells["lat"].to_numpy()])
    sxy = np.column_stack([stores["lon"].to_numpy(), stores["lat"].to_numpy()])
    ann = "duration,distance" if with_distance else "duration"
    url = base_url.rstrip("/") + "/table/v1/driving/"

    # Checkpoint parts carry this fingerprint of the candidate set. Resuming with
    # a different store frame or k changes it, and the load loop below then
    # REFUSES to start ("checkpoint part(s) ... written for a different candidate
    # set"): delete that ckpt folder, it belongs to an older solve.
    fp = _fingerprint(cand, stores.height)
    ckpt = Path(checkpoint) if checkpoint else None
    part_no = 0
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)
        n_bad = 0
        for p in sorted(ckpt.glob("part_*.npz")):
            part_no += 1
            try:
                z = np.load(p)
            except Exception:
                print(f"  unreadable checkpoint {p.name}, ignoring")
                continue
            if "fp" not in z or not np.array_equal(z["fp"], fp):
                n_bad += 1
                continue
            dur[z["rows"]], have[z["rows"]] = z["dur"], True
            if with_distance and "dist" in z:
                dist[z["rows"]] = z["dist"]
        if n_bad:
            raise ValueError(
                f"travel_times: {n_bad:,} checkpoint part(s) in {ckpt} were written "
                f"for a different candidate set (or by a version predating this "
                f"check) and cannot be replayed against this one -- their row "
                f"indices refer to different nodes. Point `checkpoint` at a fresh "
                f"directory for this specification, or delete the stale parts if "
                f"you are sure they are obsolete."
            )
        if have.any():
            print(f"resuming: {have.sum():,}/{n:,} nodes already on disk")

    all_blocks = blocks(cand, stores, target=block)
    todo = [b for b in all_blocks if not have[b].all()]
    print(f"travel_times: {len(todo):,}/{len(all_blocks):,} blocks to solve, "
          f"~{n * k / 1e6:.0f}M matrix entries")

    lock = threading.Lock()
    pend_rows: list[np.ndarray] = []
    pend_dur: list[np.ndarray] = []
    pend_dist: list[np.ndarray] = []
    failed: list[int] = []
    local = threading.local()

    def flush():
        nonlocal part_no, pend_rows, pend_dur, pend_dist
        if ckpt is None or not pend_rows:
            return
        payload = {"rows": np.concatenate(pend_rows), "dur": np.concatenate(pend_dur),
                   "fp": fp}
        if with_distance:
            payload["dist"] = np.concatenate(pend_dist)
        np.savez(ckpt / f"part_{part_no:06d}.npz", **payload)
        part_no += 1
        pend_rows, pend_dur, pend_dist = [], [], []

    def solve_block(bi: int, rows: np.ndarray):
        s = getattr(local, "s", None)
        if s is None:
            s = local.s = requests.Session()

        # One block = a few thousand neighbouring nodes. Route them against the
        # UNION of their candidates (dests), in sub-tables that respect
        # max_coords (sources + destinations per request), then pick each node's
        # own k columns back out of the dense block result (col_of).
        dests = np.unique(cand.store_idx[rows].ravel())
        col_of = np.searchsorted(dests, cand.store_idx[rows])
        D = np.full((rows.size, dests.size), np.inf, np.float32)
        M = np.full((rows.size, dests.size), np.inf, np.float32) if with_distance else None

        dcap = max(1, min(dests.size, max_coords // 2))
        for d0 in range(0, dests.size, dcap):
            dsub = dests[d0:d0 + dcap]
            per = max(1, max_coords - dsub.size)
            for s0 in range(0, rows.size, per):
                src = cxy[rows[s0:s0 + per]]
                pts = np.vstack([src, sxy[dsub]])
                params = {
                    "sources": ";".join(map(str, range(src.shape[0]))),
                    "destinations": ";".join(
                        map(str, range(src.shape[0], src.shape[0] + dsub.size))),
                    "annotations": ann,
                }
                for attempt in range(max_retries):
                    try:
                        p = _get(s, url + _coords(pts), params)
                        break
                    except requests.RequestException:
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(2 ** attempt)
                sl = slice(s0, s0 + src.shape[0])
                dm = _matrix(p, "durations")
                if dm is None:
                    raise OSRMError("NoDurations", "the /table response carried no "
                                    "`durations` key -- check `annotations` and the "
                                    "server build")
                D[sl, d0:d0 + dsub.size] = dm
                if with_distance:
                    mm = _matrix(p, "distances")
                    if mm is None:
                        raise OSRMError("NoDistances", "with_distance=True but the "
                                        "/table response carried no `distances` key")
                    M[sl, d0:d0 + dsub.size] = mm

        d = np.take_along_axis(D, col_of, axis=1)
        m = np.take_along_axis(M, col_of, axis=1) if with_distance else None
        with lock:
            dur[rows] = d
            if with_distance:
                dist[rows] = m
            pend_rows.append(rows)
            pend_dur.append(d)
            if with_distance:
                pend_dist.append(m)
            if len(pend_rows) >= flush_every:
                flush()

    import concurrent.futures as cf
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(solve_block, i, b): i for i, b in enumerate(todo)}
        for done, fut in enumerate(cf.as_completed(futs), 1):
            try:
                fut.result()
            except Exception as exc:
                failed.append(futs[fut])
                print(f"  block {futs[fut]} failed: {type(exc).__name__}: {exc}")
            if progress_every and done % progress_every == 0:
                rate = done / (time.perf_counter() - t0)
                print(f"  {done:,}/{len(todo):,} blocks "
                      f"(eta {(len(todo) - done) / rate / 60:.0f} min)", flush=True)
    with lock:
        flush()

    # A failed block leaves its rows at inf. They are not certified, so a
    # following refine_until() re-routes them at a larger k; a matrix saved with
    # inf rows still loads, and those nodes simply have no winner (they show up
    # in the "not certified" and coverage lines of stage_assign).
    if failed:
        print(f"WARNING: {len(failed):,} block(s) still inf. Re-run with the same "
              "checkpoint to retry only those.")
    print(f"travel_times: done in {(time.perf_counter() - t0) / 60:.1f} min")
    return dur, dist


def preflight(cells: pl.DataFrame, *, base_url: str = OSRM, n_routes: int = 25) -> dict:
    """Two checks worth a minute before committing to a multi-hour solve.

    1. THE LARGEST /table THE SERVER ACTUALLY ACCEPTS. Probed with a real square
       sub-matrix (sources = destinations = half the coordinates), not a 1x1
       corner of a large coordinate list. This matters because OSRM changed what
       --max-table-size bounds: older builds compare it against the number of
       COORDINATES, newer ones against sources x destinations. A 1x1 probe passes
       the second check for any coordinate count, so it can report a limit that
       the real calls then bounce off. Probing the shape travel_times sends is the
       only way to measure the constraint travel_times will hit.

       Feed the result straight back in:
           lim = preflight(c2)["max_coords"]
           travel_times(..., max_coords=lim)

    2. HOW FAST IT ROUTES. This is a route-AVERAGE speed over a long OD pair, so
       it is a lower bound on the maximum link speed, never a measurement of it --
       turn penalties, urban segments and startup all drag it down. It can tell
       you V_MAX_KMH is definitely too low; it can never tell you it is high
       enough. For that, read the profile: the relevant constants in car.lua are
       `maxspeed_table` (stock caps `none` at 140) and `max_speed_for_map_matching`
       (180). V_MAX_KMH is set from the latter for exactly this reason.
    """
    import requests

    lon, lat = cells["lon"].to_numpy(), cells["lat"].to_numpy()
    base, limit, speeds = base_url.rstrip("/"), 0, []
    rng = np.random.default_rng(0)
    with requests.Session() as s:
        for size in (100, 500, 1_000, 2_000, 5_000, 10_000):
            pts = np.resize(np.column_stack([lon[:size], lat[:size]]), (size, 2))
            half = size // 2
            try:
                r = s.get(
                    f"{base}/table/v1/driving/{_coords(pts)}",
                    params={"sources": ";".join(map(str, range(half))),
                            "destinations": ";".join(map(str, range(half, size))),
                            "annotations": "duration"},
                    timeout=300)
                if r.status_code in (414, 400, 413) or _loads(r.content).get("code") != "Ok":
                    break
                limit = size
            except Exception:
                break
        for i, j in rng.integers(0, lon.size, (n_routes, 2)):
            try:
                r0 = _get(s, f"{base}/route/v1/driving/{lon[i]:.6f},{lat[i]:.6f};"
                             f"{lon[j]:.6f},{lat[j]:.6f}", {"overview": "false"}, 60)
                r0 = r0["routes"][0]
                speeds.append(r0["distance"] / r0["duration"] * 3.6)
            except Exception:
                pass

    out = {"max_coords": limit,
           "max_table_size_at_least": limit,          # kept: old callers read this
           "mean_route_speed_kmh_max": max(speeds) if speeds else None}
    if limit < 1_000:
        print("preflight: raise --max-table-size; the default 100 rejects every call "
              "here. Note a 414 at large sizes is the URL length, not OSRM -- the "
              "coordinates go in the path.")
    else:
        print(f"preflight: /table accepts a {limit // 2} x {limit - limit // 2} "
              f"matrix ({limit:,} coords). Pass max_coords={limit} to travel_times.")
    v = out["mean_route_speed_kmh_max"]
    if v:
        print(f"preflight: fastest route-average {v:.0f} km/h (a LOWER bound on link "
              f"speed) vs V_MAX_KMH={V_MAX_KMH:.0f}.")
        if v > V_MAX_KMH:
            print("preflight: route averages already exceed V_MAX_KMH -- certified() "
                  "is definitely UNSOUND. Raise V_MAX_KMH.")
    return out


# ==========================================================================
# 6. Was k large enough?
# ==========================================================================

def certified(
    cand: Candidates, dur: np.ndarray, *, rank: int = 1, v_max_kmh: float = V_MAX_KMH
) -> np.ndarray:
    """Boolean mask, True where the rank-th nearest store is provably exact.

    Network distance is never shorter than Euclidean, so any store outside the
    candidate set has travel time >= d_k / v_max. If the rank-th best *measured*
    time is at or below that bound, no excluded store can displace it.
    """
    # bound_i = d_k,i / v_max  (seconds): the fastest any store OUTSIDE node i's
    # candidate set could possibly be reached. certified_i <=> t_(rank),i <= bound_i.
    # in row chunks: np.partition copies its input, a full-matrix temporary
    bound = cand.dist_m[:, -1].astype(np.float64) / (v_max_kmh / 3.6)
    out = np.empty(dur.shape[0], bool)
    step = max(10_000, 50_000_000 // max(dur.shape[1], 1))
    for a in range(0, dur.shape[0], step):
        d = np.nan_to_num(dur[a:a + step], nan=np.inf)
        out[a:a + step] = np.partition(d, rank - 1, axis=1)[:, rank - 1] <= bound[a:a + step]
    return out


def certification_report(cand: Candidates, dur: np.ndarray, *, ranks: int = 3) -> None:
    """Certified share per rank. Failures cluster where they matter least --
    dense areas with many near-equidistant stores, where which of #4 and #5 takes
    rank 3 changes nothing downstream. A low rank-1 share is the real alarm."""
    for r in range(1, ranks + 1):
        print(f"  rank {r}: {certified(cand, dur, rank=r).mean():.2%} certified")


def refine(
    cells: pl.DataFrame, stores: pl.DataFrame, cand: Candidates, dur: np.ndarray,
    *, k_new: int, rank: int = 1, **kw
) -> tuple[Candidates, np.ndarray, np.ndarray]:
    """Re-solve uncertified nodes at a larger k. Returns (cand, dur, certified).

    Unrefined rows are padded by *repeating their existing k-th distance*, not
    with inf: the bound reads dist_m[:, -1], and padding it with inf would make
    every row trivially certified while certifying nothing.
    """
    _check_aligned(cells, stores, cand, dur, what="refine")
    ok = certified(cand, dur, rank=rank)
    todo = np.flatnonzero(~ok)
    n, k_old = cand.store_idx.shape
    if todo.size == 0 or k_new <= k_old:
        return cand, dur, ok
    print(f"refine: {todo.size:,}/{n:,} uncertified at k={k_old}, retrying at k={k_new}")

    pad = k_new - k_old
    idx = np.concatenate([cand.store_idx, np.zeros((n, pad), np.int32)], axis=1)
    dst = np.concatenate([cand.dist_m, np.repeat(cand.dist_m[:, -1:], pad, 1)], axis=1)
    dw = np.concatenate([dur, np.full((n, pad), np.inf, np.float32)], axis=1)

    # the uncertified rows are re-solved from scratch at k_new (their first k_old
    # pairs are routed again: simpler than merging, and these rows are a minority)
    sub = cells[todo]
    sc = candidates(sub, stores, k=k_new)
    sd, _ = travel_times(sub, stores, sc, **kw)
    idx[todo], dst[todo], dw[todo] = sc.store_idx, sc.dist_m, sd

    c2 = Candidates(idx, dst, cand.store_ids)
    ok2 = certified(c2, dw, rank=rank)
    print(f"refine: certified {ok.mean():.2%} -> {ok2.mean():.2%}")
    return c2, dw, ok2


def refine_until(
    cells: pl.DataFrame, stores: pl.DataFrame, cand: Candidates, dur: np.ndarray, *,
    target: float = 0.90, rank: int = 1, ks: Sequence[int] = (40, 60, 90, 140),
    checkpoint: str | Path | None = None, **kw,
) -> tuple[Candidates, np.ndarray, np.ndarray]:
    """Widen k until `target` of nodes are certified at `rank`, or `ks` runs out.

    One refine() pass is a guess: it widens to a fixed k and whatever certification
    rate that buys is what you report. Which k suffices depends on the local store
    density, which varies by an order of magnitude between Munich and the Bavarian
    Forest, so no single k is right everywhere. This escalates instead, and each
    pass only re-solves the nodes still uncertified -- so the expensive passes run
    on a rapidly shrinking set.

    Returns the widened (cand, dur, certified) and always leaves cand/dur mutually
    consistent, which is the property that matters: it is the caller rebinding
    only some of the three that produces a notebook whose in-memory results and
    saved matrix disagree.

    Each pass gets its OWN checkpoint subdirectory. Reusing one directory across
    passes would hit the fingerprint check in travel_times (correctly -- the
    candidate set differs every pass) and abort the run.
    """
    ok = certified(cand, dur, rank=rank)
    print(f"refine_until: start k={cand.k}, rank-{rank} certified {ok.mean():.2%}, "
          f"target {target:.0%}")
    for k_new in ks:
        if ok.mean() >= target:
            break
        if k_new <= cand.k:
            continue
        if k_new >= stores.height:
            print(f"  k={k_new} exceeds the store count ({stores.height:,}); stopping")
            break
        ck = Path(checkpoint) / f"k{k_new}" if checkpoint is not None else None
        cand, dur, ok = refine(cells, stores, cand, dur, k_new=k_new, rank=rank,
                               checkpoint=ck, **kw)
    if ok.mean() < target:
        print(f"WARNING refine_until: stopped at k={cand.k} with {ok.mean():.2%} "
              f"certified, below the {target:.0%} target. Extend `ks`, or accept and "
              f"report the rate -- uncertified does not mean wrong, it means "
              f"unproven, and failures concentrate where stores are dense and the "
              f"identity of the marginal alternative changes nothing.")
    return cand, dur, ok


def extend(
    cells: pl.DataFrame, stores: pl.DataFrame, cand: Candidates, dur: np.ndarray,
    n_old: int, *, checkpoint: str | Path | None = None, **kw,
) -> tuple[Candidates, np.ndarray]:
    """Add the stores `stores[n_old:]` to a solved matrix. Exact, and it routes
    only the pairs that can matter.

    Every row's candidates are sorted by Euclidean distance, and certified()
    bounds everything *outside* the set by d_k / v_max with d_k = dist_m[:, -1].
    A new store farther than d_k from a node is therefore already covered by that
    bound -- it cannot beat a certified winner, and it leaves an uncertified row
    exactly as unproven as before. Only new stores within d_k of a node need a
    drive time. Those pairs are routed, inserted into the row, and the row is
    re-sorted so its last column is still d_k. Existing durations are untouched,
    so every certified node stays certified and its winner can only improve.

    `stores` must be the old frame with the new stores APPENDED: rows 0..n_old-1
    are checked against cand.store_ids, because the matrix indexes them by
    position. Closed stores need nothing here -- assignment masks them per year.

    Peak memory is about twice the matrix (old and widened arrays side by side).
    """
    from scipy.spatial import cKDTree

    _check_aligned(cells, stores.head(n_old), cand, dur, what="extend")
    new = stores.slice(n_old)
    ids = stores["store_id"].to_numpy().astype(str)
    if new.is_empty():
        return Candidates(cand.store_idx, cand.dist_m, ids), dur

    nxy = np.column_stack([new["x"].to_numpy(), new["y"].to_numpy()])
    cxy = np.column_stack([cells["x"].to_numpy(), cells["y"].to_numpy()])
    dk = cand.dist_m[:, -1].astype(np.float64)
    hits = cKDTree(nxy).query_ball_point(cxy, r=dk, workers=-1, return_sorted=False)
    cnt = np.fromiter((len(h) for h in hits), np.int32, len(hits))
    rows = np.flatnonzero(cnt)
    print(f"extend: {new.height:,} new store(s) fall inside the candidate radius of "
          f"{rows.size:,}/{cnt.size:,} node(s); routing {int(cnt.sum()):,} pair(s)")
    if rows.size == 0:
        return Candidates(cand.store_idx, cand.dist_m, ids), dur

    m = int(cnt[rows].max())
    sub_idx = np.empty((rows.size, m), np.int32)
    for i, r in enumerate(rows):                   # affected rows only
        h = hits[r]
        sub_idx[i, :len(h)] = h
        sub_idx[i, len(h):] = h[0]                 # padding, masked below
    valid = np.arange(m)[None, :] < cnt[rows][:, None]
    sub_dst = np.hypot(cxy[rows, :1] - nxy[sub_idx, 0],
                       cxy[rows, 1:] - nxy[sub_idx, 1]).astype(np.float32)
    sc = Candidates(sub_idx, sub_dst, new["store_id"].to_numpy().astype(str))
    sd, _ = travel_times(cells[rows], new, sc, checkpoint=checkpoint, **kw)

    n, k = cand.store_idx.shape
    K = k + m
    idx = np.zeros((n, K), np.int32)
    dst = np.repeat(cand.dist_m[:, -1:], K, axis=1)
    dw = np.full((n, K), np.inf, np.float32)
    idx[:, :k], dst[:, :k], dw[:, :k] = cand.store_idx, cand.dist_m, dur
    kr = dst[rows, -1:]
    idx[rows, k:] = np.where(valid, sub_idx + n_old, 0)
    dst[rows, k:] = np.where(valid, sub_dst, kr)   # padding keeps d_k: bound intact
    dw[rows, k:] = np.where(valid, sd, np.inf)
    o = np.argsort(dst[rows], axis=1, kind="stable")
    idx[rows] = np.take_along_axis(idx[rows], o, axis=1)
    dst[rows] = np.take_along_axis(dst[rows], o, axis=1)
    dw[rows] = np.take_along_axis(dw[rows], o, axis=1)
    return Candidates(idx, dst, ids), dw


_MATRIX_ARRAYS = ("store_idx", "euclid_m", "durations", "distances")


def save_matrix(path, cells, stores, cand, dur, dist=None, cert=None) -> None:
    """Persist the year-invariant matrix -- the durable artefact. Per-year
    assignments are derived output, cheap to regenerate, not worth versioning.

    Parquet, one row per solve node: cell_id, pop, certified, and the n x k
    arrays as fixed-width list columns (each row's k candidates in one value).
    The store frame the indices refer to goes into the file's footer metadata,
    so saved_store_ids() reads it without touching the arrays. zstd compresses
    on every core; the zlib inside .npz is single-threaded, which made writing
    an extended matrix take tens of minutes. A ".npz" path still writes the old
    format.

    Written to a temporary file and renamed: this file is hours of OSRM work,
    and a crash mid-write must not leave a truncated matrix in place.
    """
    # On-disk layout (matrix.parquet), one row per solve node, in c2 order:
    #   cell_id   int64           solve-node id (must equal c2["cell_id"])
    #   pop       float           node population (informational)
    #   store_idx list<int32>[k]  candidate positions in the saved store frame
    #   euclid_m  list<float>[k]  straight-line metres, ascending per row
    #   durations list<float>[k]  OSRM seconds, inf = unreachable / not routed
    #   certified bool            rank-1 winner proven (at save time)
    #   footer metadata: store_id = JSON list of the frame's ids, k = width
    _check_aligned(cells, stores, cand, dur, what="save_matrix")
    path = Path(path)
    sid = stores["store_id"].to_numpy().astype(str)
    if path.suffix == ".npz":
        p = {"store_idx": cand.store_idx, "euclid_m": cand.dist_m, "durations": dur,
             "cell_id": cells["cell_id"].to_numpy(), "pop": cells["pop"].to_numpy(),
             "store_id": sid}
        if dist is not None:
            p["distances"] = dist
        if cert is not None:
            p["certified"] = cert
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **p)
        tmp.replace(path)
        return

    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    def fsl(m):
        m = np.ascontiguousarray(m)
        return pa.FixedSizeListArray.from_arrays(pa.array(m.ravel()), m.shape[1])

    cols = {"cell_id": pa.array(cells["cell_id"].to_numpy()),
            "pop": pa.array(cells["pop"].to_numpy()),
            "store_idx": fsl(cand.store_idx), "euclid_m": fsl(cand.dist_m),
            "durations": fsl(dur)}
    if dist is not None:
        cols["distances"] = fsl(dist)
    if cert is not None:
        cols["certified"] = pa.array(np.asarray(cert, dtype=bool))
    t = pa.table(cols).replace_schema_metadata(
        {b"store_id": json.dumps(sid.tolist()).encode(), b"k": str(cand.k).encode()})
    tmp = path.with_name(path.stem + ".tmp.parquet")
    pq.write_table(t, tmp, compression="zstd", row_group_size=500_000)
    tmp.replace(path)


def saved_store_ids(path) -> np.ndarray:
    """The store frame a saved matrix indexes, without reading its arrays."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as z:
            return z["store_id"].astype(str)
    import json
    import pyarrow.parquet as pq
    return np.array(json.loads(pq.read_schema(path).metadata[b"store_id"]), dtype=str)


def _read_matrix(path, columns=None) -> dict:
    """The saved arrays as numpy; `columns` limits what is read (cell_id always).

    Filled row group by row group into preallocated arrays. Reading the whole
    table first and converting it held the Arrow table, a combined copy of each
    column and the numpy result at once -- about 3x the matrix, ~25 GB for a
    national k=140 matrix. This way the peak is the arrays plus one row group.
    """
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files if columns is None or k in columns
                or k in ("cell_id", "store_id")}
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    schema, n = pf.schema_arrow, pf.metadata.num_rows
    names = [c for c in schema.names
             if columns is None or c in columns or c == "cell_id"]
    out = {}
    for c in names:
        typ = schema.field(c).type
        out[c] = (np.empty((n, typ.list_size), typ.value_type.to_pandas_dtype())
                  if c in _MATRIX_ARRAYS else np.empty(n, typ.to_pandas_dtype()))
    r = 0
    for i in range(pf.num_row_groups):
        g = pf.read_row_group(i, columns=names)
        m = g.num_rows
        for c in names:
            col = g[c].combine_chunks()
            out[c][r:r + m] = (col.flatten().to_numpy().reshape(m, -1)
                               if c in _MATRIX_ARRAYS
                               else col.to_numpy(zero_copy_only=False))
        r += m
        del g, col
    pa.default_memory_pool().release_unused()   # hand Arrow's buffers back to the OS
    out["store_id"] = saved_store_ids(path)
    return out


def load_matrix(path, stores: pl.DataFrame | None = None,
                cells: pl.DataFrame | None = None, columns=None) -> dict:
    """Inverse of save_matrix, for either format. Pass `stores`/`cells` to
    verify alignment; `columns` to read only some arrays (assignment needs
    store_idx, euclid_m and durations, not distances or certified).

    The matrix stores *positional* store indices, so it is only meaningful next
    to the exact store frame it was built from. Reusing it with a differently
    ordered frame relabels every store silently -- Edeka's catchment becomes
    Aldi's, with no error anywhere. Always pass `stores`.
    """
    out = _read_matrix(path, columns)
    saved = out["store_id"].astype(str)
    out["candidates"] = Candidates(out["store_idx"], out["euclid_m"], saved)
    if stores is not None:
        now = stores["store_id"].to_numpy().astype(str)
        if saved.shape != now.shape or not np.array_equal(saved, now):
            n_same = int((saved[:min(saved.size, now.size)]
                          == now[:min(saved.size, now.size)]).sum())
            raise ValueError(
                f"load_matrix: store frame does not match the one this matrix was "
                f"built from ({saved.size:,} saved vs {now.size:,} now, "
                f"{n_same:,} aligned). The matrix holds positional store indices, "
                f"so using it here would silently relabel catchments. Rebuild the "
                f"store frame with the same prepare_stores(...) call and filters, "
                f"or re-solve."
            )
    if cells is not None:
        saved_c = out["cell_id"]
        now_c = cells["cell_id"].to_numpy()
        if saved_c.shape != now_c.shape or not np.array_equal(saved_c, now_c):
            raise ValueError(
                f"load_matrix: cell frame does not match ({saved_c.size:,} saved vs "
                f"{now_c.size:,} now). Rows of the matrix refer to different nodes; "
                f"re-run the reductions with identical parameters or re-solve."
            )
    return out


# ==========================================================================
# 7. Assignment
# ==========================================================================

def assign(
    cells: pl.DataFrame, stores: pl.DataFrame, cand: Candidates, dur: np.ndarray, *,
    year: int | None = None, store_mask: np.ndarray | None = None, ranks: int = 1,
    max_seconds: float | None = None, chunk: int = 2_000_000,
) -> pl.DataFrame:
    """Each node's `ranks` nearest alive stores. One row per (node, rank).

    `year` filters on first_year/last_year; `store_mask` is an explicit override
    for format- or chain-specific catchments.

    `max_seconds` keeps only alternatives inside a drive-time isochrone. Combined
    with ranks=cand.k this is the natural way to build a *local market* -- every
    store a cell can reach within T minutes -- rather than a fixed top-N, which
    imposes an arbitrary ceiling on the choice set (and, downstream, a hard floor
    on any HHI computed from it). Rows are masked per chunk, so ranks=60 with a
    20-minute cap costs a fraction of the memory of an unfiltered ranks=60.

    argpartition rather than a full argsort, in row chunks: a full argsort of an
    (n, k) float32 array allocates n*k int64 alongside it, which at 7.27M x 20 is
    1.2 GB for information you discard immediately.

    Rank coverage is not uniform. A node with fewer than `ranks` reachable alive
    candidates yields fewer rows, so ranks 2-3 are thinner in sparse areas and in
    high-exit years. Check rank_coverage() before feeding this to anything that
    assumes a balanced (node x rank) panel -- a Huff denominator built on a
    truncated choice set overstates the retained alternatives' shares.
    """
    # Output: one row per (node, rank) with a finite time --
    #   cell_id (node id), rank, store_id, duration_s, euclid_m, pop[, year].
    # A node whose candidates are all closed this year has no row at all (an
    # "orphan"); stage_assign counts it as not certified.
    _check_aligned(cells, stores, cand, dur, what="assign")
    n, k = cand.store_idx.shape
    ranks = min(ranks, k)
    if store_mask is not None:
        alive = np.asarray(store_mask, bool)
    elif year is None:
        alive = np.ones(stores.height, bool)
    else:
        alive = ((stores["first_year"].to_numpy() <= year)
                 & (stores["last_year"].to_numpy() >= year))

    # ids are gathered by polars, not numpy: a numpy unicode array of 100M ids
    # (the isochrone frame) is 3+ GB of fixed-width text before it is converted
    sid = stores["store_id"].cast(pl.Utf8)
    cid, pop = cells["cell_id"].to_numpy(), cells["pop"].to_numpy()
    # keep the per-chunk temporaries bounded: `eff` and argpartition's int64
    # result are (chunk x k), so the row chunk shrinks as the matrix widens --
    # ~150k rows at k=140, i.e. ~0.3 GB of temporaries instead of several GB.
    chunk = int(min(chunk, max(50_000, 20_000_000 // max(k, 1))))
    parts: list[pl.DataFrame] = []    # one frame per chunk: never a second full copy
    rank_row = np.arange(1, ranks + 1, dtype=np.int8)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        idx = cand.store_idx[a:b]
        eff = np.where(alive[idx], dur[a:b], np.inf)
        if ranks == 1:
            pos = eff.argmin(axis=1)[:, None]
        elif ranks == k:
            pos = np.argsort(eff, axis=1)
        else:
            part = np.argpartition(eff, ranks - 1, axis=1)[:, :ranks]
            vals = np.take_along_axis(eff, part, axis=1)
            pos = np.take_along_axis(part, np.argsort(vals, axis=1), axis=1)
        t = np.take_along_axis(eff, pos, axis=1)
        m = np.isfinite(t)
        if max_seconds is not None:
            m &= t <= max_seconds
        m = m.ravel()
        parts.append(pl.DataFrame({
            "cell_id": np.repeat(cid[a:b], ranks)[m],
            "rank": np.tile(rank_row, b - a)[m],
            "store_id": sid.gather(np.take_along_axis(idx, pos, axis=1).ravel()[m]),
            "duration_s": t.ravel()[m],
            "euclid_m": np.take_along_axis(cand.dist_m[a:b], pos, axis=1).ravel()[m],
            "pop": np.repeat(pop[a:b], ranks)[m],
        }))

    out = pl.concat(parts, rechunk=False)
    if year is not None:
        out = out.with_columns(pl.lit(year, pl.Int16).alias("year"))
    return out


def orphans(cand: Candidates, dur: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """Nodes whose entire candidate set is dead in this year.

    The price of caching against the union of stores over all years: if every one
    of a node's k candidates has exited by year t, the node drops out and its
    population silently vanishes from the totals. With realistic churn this is a
    handful of rural nodes, but check rather than assume -- re-query the orphans
    against the alive-only store set and splice the results back in.
    """
    return np.flatnonzero(~np.isfinite(
        np.where(alive[cand.store_idx], dur, np.inf)).any(axis=1))


def rank_coverage(a: pl.DataFrame) -> pl.DataFrame:
    """Share of nodes and of population reaching each rank. Rank 1 should be
    ~100% of non-orphans; a sharp fall-off means the candidate set is being
    exhausted and the higher ranks are not a safe choice set."""
    base = a.filter(pl.col("rank") == 1)
    tc, tp = base.height, base["pop"].sum()
    g = ["rank"] + (["year"] if "year" in a.columns else [])
    return (a.group_by(g)
             .agg(pl.len().alias("n"), pl.col("pop").sum().alias("pop"))
             .with_columns((pl.col("n") / tc).alias("share_nodes"),
                           (pl.col("pop") / tp).alias("share_pop"))
             .sort(g))


def catchments(a: pl.DataFrame) -> pl.DataFrame:
    """Store-level catchment population and reach, from rank-1 assignments."""
    return (a.filter(pl.col("rank") == 1)
             .group_by([c for c in ("store_id", "year") if c in a.columns])
             .agg(pl.col("pop").cast(pl.Float64).sum().alias("catchment_pop"),
                  pl.len().alias("n_nodes"),
                  pl.col("duration_s").mean().alias("mean_drive_s"),
                  pl.col("duration_s").max().alias("max_drive_s"),
                  (pl.col("pop").cast(pl.Float64)
                   * pl.col("duration_s").cast(pl.Float64)).sum().alias("_pt"))
             .with_columns((pl.col("_pt") / pl.col("catchment_pop"))
                           .alias("pop_wtd_drive_s"))
             .drop("_pt")
             .sort([c for c in ("store_id", "year") if c in a.columns]))


def write_label_raster(
    cells: pl.DataFrame, assignment: pl.DataFrame, mapping: pl.DataFrame,
    template: str | Path, out_path: str | Path,
) -> None:
    """Burn rank-1 assignments back onto the original grid (int32, 0 = unassigned).

    `cells` is the *original* frame from load_cells (it carries row/col);
    `mapping` is compose(...) of every reduction applied. Codes are positional,
    with the store_id lookup written as a Parquet sidecar. Polygonise from this
    only if you genuinely need vectors -- the raster is cheaper for area and
    population arithmetic.
    """
    import rasterio

    r1 = assignment.filter(pl.col("rank") == 1)
    lookup = (pl.DataFrame({"store_id": r1["store_id"].unique(maintain_order=True)})
                .with_row_index("code", offset=1))
    burn = (mapping.join(r1.select(pl.col("cell_id").alias("dst"), "store_id"),
                         on="dst", how="inner")
                   .join(lookup, on="store_id", how="left")
                   .join(cells.select(pl.col("cell_id").alias("src"), "row", "col"),
                         on="src", how="inner"))

    with rasterio.open(template) as src:
        profile, shape = src.profile, (src.height, src.width)
    profile.update(dtype="int32", count=1, nodata=0, compress="lzw")

    grid = np.zeros(shape, np.int32)
    grid[burn["row"].to_numpy(), burn["col"].to_numpy()] = burn["code"].to_numpy()
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(grid, 1)
    lookup.write_parquet(Path(out_path).with_suffix(".lookup.parquet"))