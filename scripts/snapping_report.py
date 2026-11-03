"""
Snapping diagnostics: how far stores and population were moved onto the roads.

    python -m scripts.snapping_report runs/baseline.toml
    python -m scripts.snapping_report runs/baseline.toml --osrm http://localhost:5000
    python -m scripts.snapping_report runs/baseline.toml --roads
    python -m scripts.snapping_report runs/baseline.toml --roads-file roads.geojson

Every point the solver routes from is first moved to the nearest drivable road
by OSRM /nearest ("snapped"). This reads what the pipeline already wrote and
reports how far, for whom, and where:

  stores      snapped/stores.parquet keeps OSRM's distance (snap_m) per store;
              rejected.txt lists the stores dropped (no road within 300 m, or
              no coordinates). --osrm asks OSRM for the rejected stores'
              distance (one /nearest each).
  population  the grid files keep no per-point distance. Two sources:
              * the cell snap cache (snapped/cache/snap_<n>_*.npz, n = number of
                snapped grid points) holds OSRM's distance: exact, used if present;
              * otherwise it is recomputed from geometry: the point that was sent
                to OSRM (a 100 m cell centre, or the population-weighted centroid
                of a merged block) to the node it snapped to, in EPSG:3035 metres
                (differs from OSRM's great-circle metres by well under 1 %).
              Reported per snapped point, population-weighted, and as the total
              displacement of every 100 m cell (coarsening + snapping).

Maps:
  stores_map               stores that moved >= 50 m, and the rejected ones
  stores_kreis_share       share of a Kreis' stores that moved >= 50 m
  pop_map                  population-weighted snap distance on a 2 km grid
  pop_kreis                the same per Kreis
  road_points_germany      every snapped grid point, one 100 m pixel each: the
                           routable network as the solver sees it, wherever
                           people live (PNG only; large)
  window_<city>            6 x 6 km around each city in Config.cities: cells,
                           their snap vectors, stores before -> after, and with
                           --roads / --roads-file the OpenStreetMap roads
                           (drivable dark, others light)

Writes to outputs/catchments/<grid>/snapping/: CSV tables, PNG/PDF figures and
summary.md (the console summary, ready to paste).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

import scripts.run_pipeline as P
import scripts.isochrones.market_cube as C
import scripts.visualizations.figures as F
from scripts.runconfig import LAENDER, RunConfig

#: store distance bands, metres; the 300 m edge is Config.drop_over_m
STORE_BANDS = [10, 25, 50, 100, 300]
#: population thresholds, metres
POP_THRESH = [50, 100, 250, 500, 1000]
#: highway values OSRM's stock car profile routes on; the rest is drawn light
CAR = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
       "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
       "residential", "living_street", "service", "road"}
WINDOW_HALF_KM = 3.0

_LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _LINES.append(s)


# ---------------------------------------------------------------- helpers --

def hav(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(a, float)) for a in (lon1, lat1, lon2, lat2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6_371_008.8 * np.arcsin(np.sqrt(a))


def describe(v, w=None) -> dict:
    """Count, mean, sd and quantiles of v, optionally weighted by w."""
    v = np.asarray(v, float)
    w = np.ones_like(v) if w is None else np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return {}
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    cw = np.cumsum(w) / w.sum()
    q = lambda p: float(v[min(int(np.searchsorted(cw, p)), v.size - 1)])  # noqa: E731
    m = float(np.average(v, weights=w))
    return {"n": int(v.size), "weight": float(w.sum()), "mean": m,
            "sd": float(np.sqrt(np.average((v - m) ** 2, weights=w))),
            "p50": q(.50), "p75": q(.75), "p90": q(.90), "p95": q(.95),
            "p99": q(.99), "p99.9": q(.999), "max": float(v[-1])}


def stats(groups: dict) -> pl.DataFrame:
    """One describe() row per group; groups maps name -> (values, weights|None)."""
    rows = [{"group": k, **d} for k, (v, w) in groups.items() if (d := describe(v, w))]
    return pl.DataFrame(rows)


def md(df: pl.DataFrame, digits: int = 1) -> str:
    """A polars frame as a Markdown table (for summary.md and pasting)."""
    def f(x):
        if isinstance(x, float):
            return "" if x != x else f"{x:,.{digits}f}"
        return f"{x:,}" if isinstance(x, int) else str(x)
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    return "\n".join([head, sep] + ["| " + " | ".join(f(x) for x in r) + " |"
                                    for r in df.iter_rows()])


def band(col: str) -> pl.Expr:
    e = pl.when(pl.col(col).is_null()).then(pl.lit("rejected"))
    lo = 0
    for hi in STORE_BANDS:
        e = e.when(pl.col(col) < hi).then(pl.lit(f"{lo}-{hi} m"))
        lo = hi
    return e.otherwise(pl.lit(f">= {lo} m")).alias("band")


BAND_ORDER = ([f"{a}-{b} m" for a, b in zip([0] + STORE_BANDS[:-1], STORE_BANDS)]
              + [f">= {STORE_BANDS[-1]} m", "rejected"])


# ----------------------------------------------------------------- stores --

def store_frames(cfg, panel: pl.DataFrame, osrm: str | None):
    """(snapped stores with panel info, rejected stores with panel info)."""
    keep = [c for c in ("Name", "Straße", "Plz", "Ort", "AGS5", "Kreis") if c in panel.columns]
    info = (panel.sort("year")
            .group_by(pl.col("TDLinx").cast(pl.Utf8).alias("store_id"))
            .agg(*[pl.col(c).drop_nulls().last() for c in keep],
                 pl.col("Shop Type").last().alias("shop_type_last"),
                 pl.col("Gruppe").last().alias("gruppe_last"),
                 pl.col("lon").median().alias("lon_panel"),
                 pl.col("lat").median().alias("lat_panel"),
                 pl.col("year").min().alias("first_year_panel"),
                 pl.col("year").max().alias("last_year_panel")))
    st = (pl.read_parquet(cfg.snapdir / "stores.parquet")
          .with_columns(pl.col("store_id").cast(pl.Utf8))
          .join(info, on="store_id", how="left")
          .with_columns(band("snap_m")))

    rej_path = cfg.snapdir / "rejected.txt"
    ids = rej_path.read_text(encoding="utf-8").split() if rej_path.exists() else []
    rej = (pl.DataFrame({"store_id": ids}, schema={"store_id": pl.Utf8})
           .join(info, on="store_id", how="left")
           .with_columns(pl.when(pl.col("lon_panel").is_null() | pl.col("lat_panel").is_null())
                         .then(pl.lit("no coordinates"))
                         .otherwise(pl.lit(f"no road within {cfg.drop_over_m:.0f} m"))
                         .alias("reason")))
    if osrm and rej.height:
        rej = rej.with_columns(pl.Series("snap_m", _nearest(osrm, rej), dtype=pl.Float64))
    return st, rej


def _nearest(osrm: str, df: pl.DataFrame) -> list:
    import requests
    out, url = [], osrm.rstrip("/") + "/nearest/v1/driving/"
    with requests.Session() as s:
        for lon, lat in zip(df["lon_panel"], df["lat_panel"]):
            if lon is None or lat is None:
                out.append(None)
                continue
            try:
                j = s.get(f"{url}{lon:.6f},{lat:.6f}", params={"number": 1}, timeout=10).json()
                out.append(float(j["waypoints"][0]["distance"]) if j.get("code") == "Ok" else None)
            except Exception:
                out.append(None)
    return out


def report_stores(st, rej, out: Path):
    say("## Stores")
    n_rej = rej.height
    say(f"{st.height:,} stores snapped and kept, {n_rej:,} rejected "
        f"({rej.filter(pl.col('reason') == 'no coordinates').height:,} without coordinates). "
        "Every store with coordinates goes through OSRM /nearest; the question is how far it moved.")
    chk = hav(st["lon_orig"], st["lat_orig"], st["lon"], st["lat"]) - st["snap_m"].to_numpy()
    say(f"check: OSRM's snap_m vs. distance original->snapped position: "
        f"max |diff| {np.nanmax(np.abs(chk)):.2f} m")
    say()

    allst = pl.concat([st.select("store_id", "snap_m", "band", "shop_type_last"),
                       rej.select("store_id", pl.lit(None, pl.Float64).alias("snap_m"),
                                  pl.lit("rejected").alias("band"), "shop_type_last")])
    bands = (allst.group_by("band").agg(pl.len().alias("stores"))
             .with_columns((pl.col("stores") / pl.col("stores").sum() * 100).alias("%"))
             .with_columns(pl.col("band").replace_strict(
                 {b: i for i, b in enumerate(BAND_ORDER)}, default=99).alias("_o"))
             .sort("_o").drop("_o"))
    say("Distance bands (all stores ever in the panel with a TDLinx id):\n")
    say(md(bands))
    say()
    by_type = (allst.group_by("shop_type_last", "band").agg(pl.len().alias("n"))
               .pivot(on="band", index="shop_type_last", values="n")
               .fill_null(0).sort("shop_type_last"))
    by_type = by_type.select("shop_type_last", *[b for b in BAND_ORDER if b in by_type.columns])
    say("By shop type (format in the store's last panel year):\n")
    say(md(by_type.rename({"shop_type_last": "shop type"})))
    say()

    v = st["snap_m"].to_numpy()
    groups = {"all kept stores": (v, None)}
    for t in sorted(st["shop_type_last"].drop_nulls().unique().to_list()):
        groups[t] = (st.filter(pl.col("shop_type_last") == t)["snap_m"].to_numpy(), None)
    s = stats(groups).drop("weight")
    say("Descriptive statistics of snap distance, metres (kept stores; rejected excluded):\n")
    say(md(s))
    say()
    bl = st.with_columns(pl.col("AGS5").cast(pl.Utf8).str.slice(0, 2).alias("ags2"))
    rows = []
    for code in sorted(c for c in bl["ags2"].drop_nulls().unique().to_list() if c in LAENDER):
        x = bl.filter(pl.col("ags2") == code)["snap_m"].to_numpy()
        d = describe(x)
        rows.append({"Land": LAENDER[code], "stores": d["n"], "mean": d["mean"],
                     "p50": d["p50"], "p90": d["p90"], "max": d["max"],
                     "% >= 50 m": float((x >= 50).mean() * 100)})
    say("By Bundesland:\n")
    say(md(pl.DataFrame(rows)))
    say()
    s.write_csv(out / "stores_stats.csv")
    bands.write_csv(out / "stores_bands.csv")

    cols = [c for c in ("store_id", "snap_m", "band", "shop_type_last", "gruppe_last", "Name",
                        "Straße", "Plz", "Ort", "Kreis", "AGS5", "coord_span_m",
                        "lon_orig", "lat_orig", "lon", "lat") if c in st.columns]
    far = (st.filter(pl.col("snap_m") >= 100).sort("snap_m", descending=True).select(cols)
           .with_columns(osm_link(pl.col("lat_orig"), pl.col("lon_orig"))))
    far.write_csv(out / "stores_far_100m.csv")
    st.select(cols).write_csv(out / "stores_all.csv")
    rcols = [c for c in ("store_id", "reason", "snap_m", "shop_type_last", "gruppe_last", "Name",
                         "Straße", "Plz", "Ort", "Kreis", "first_year_panel", "last_year_panel",
                         "lon_panel", "lat_panel") if c in rej.columns]
    rej.select(rcols).with_columns(osm_link(pl.col("lat_panel"), pl.col("lon_panel"))) \
       .write_csv(out / "stores_rejected.csv")
    show = [c for c in ("store_id", "snap_m", "shop_type_last", "Name", "Ort") if c in far.columns]
    say(f"{far.height:,} stores moved >= 100 m (all in stores_far_100m.csv, with OSM links). "
        f"Largest:\n")
    say(md(far.select(show).head(15)))
    say()
    if rej.height:
        show = [c for c in ("store_id", "reason", "snap_m", "shop_type_last", "Name", "Ort")
                if c in rej.columns]
        say("Rejected stores (stores_rejected.csv):\n")
        say(md(rej.select(show).head(30)))
        say()


def osm_link(lat: pl.Expr, lon: pl.Expr) -> pl.Expr:
    return pl.when(lat.is_not_null()).then(
        pl.format("https://www.openstreetmap.org/?mlat={}&mlon={}#map=18/{}/{}",
                  lat.round(6), lon.round(6), lat.round(6), lon.round(6))).alias("osm")


# ------------------------------------------------------------- population --

def cell_frames(cfg):
    """(snapped points, 100 m cells, source) -- see the module docstring.

    snapped points: one row per point sent to OSRM (c1): pop, n_raw cells merged,
                    position sent (x1, y1), node it became (xn, yn), snap distance.
    cells:          one row per populated 100 m cell: pop, its snapped point's
                    snap distance, and its total displacement to the node.
    """
    g = cfg.griddir
    raw = pl.read_parquet(g / "cells_raw.parquet", columns=["cell_id", "row", "col", "x", "y", "pop"])
    ma = pl.read_parquet(g / "map_adapt.parquet").rename({"src": "cell_id", "dst": "c1"})
    md_ = pl.read_parquet(g / "map_dedupe.parquet").rename({"src": "c1", "dst": "node"})
    c2 = pl.read_parquet(g / "cells_solve.parquet", columns=["cell_id", "x", "y"]) \
        .rename({"cell_id": "node", "x": "xn", "y": "yn"})
    # the order the snapped points were sent to OSRM: map_adapt lists the kept
    # 100 m cells first and then the merged blocks, exactly as adapt_resolution
    # built its output frame -- which is also the row order of the snap cache
    order = ma["c1"].unique(maintain_order=True)
    w = pl.col("pop").cast(pl.Float64)
    pts = (raw.join(ma, on="cell_id")
           .group_by("c1").agg(w.sum().alias("pop"),
                               ((pl.col("x") * w).sum() / w.sum()).alias("x1"),
                               ((pl.col("y") * w).sum() / w.sum()).alias("y1"),
                               pl.len().alias("n_raw")))
    pts = (order.to_frame().with_row_index("pos").join(pts, on="c1", how="left")
           .join(md_, on="c1", how="left").join(c2, on="node", how="left")
           .with_columns(((pl.col("x1") - pl.col("xn")) ** 2
                          + (pl.col("y1") - pl.col("yn")) ** 2).sqrt().alias("snap_geom"),
                         pl.when(pl.col("n_raw") > 1).then(pl.lit("merged block"))
                         .otherwise(pl.lit("100 m cell")).alias("kind"))
           .sort("pos"))
    source = "geometry (no cell snap cache found)"
    hits = sorted(Path(cfg.cachedir).glob(f"snap_{pts.height}_*.npz"))
    if hits:
        z = np.load(hits[-1])
        pts = pts.with_columns(pl.Series("snap_osrm", z["snap_m"].astype(float)),
                               pl.Series("done", z["done"].astype(bool)))
        source = f"OSRM's distances from the snap cache {hits[-1].name}"
    snap = "snap_osrm" if hits else "snap_geom"
    pts = pts.with_columns(pl.col(snap).alias("snap"))
    cells = (raw.join(ma, on="cell_id")
             .join(pts.select("c1", "snap", "xn", "yn"), on="c1", how="left")
             .with_columns(((pl.col("x") - pl.col("xn")) ** 2
                            + (pl.col("y") - pl.col("yn")) ** 2).sqrt().alias("disp")))
    return pts, cells, source


def report_population(cfg, pts, cells, source, kreis_gdf, out: Path):
    say("## Population")
    say(f"Source of the snap distances: {source}.")
    say(f"{cells.height:,} populated 100 m cells ({cells['pop'].sum():,.0f} people) -> "
        f"{pts.height:,} points sent to OSRM -> {pts['node'].n_unique():,} solve nodes.")
    if "done" in pts.columns:
        bad = pts.filter(~pl.col("done"))
        say(f"Points OSRM could not snap at all: {bad.height:,} "
            f"({bad['pop'].sum():,.0f} people); they were routed from their own position.")
        d = (pts["snap_osrm"] - pts["snap_geom"]).abs().to_numpy()
        say(f"check: OSRM distance vs. geometric reconstruction: median |diff| "
            f"{np.nanmedian(d):.2f} m, p99 {np.nanpercentile(d, 99):.2f} m")
    say()
    v, w = pts["snap"].to_numpy(), pts["pop"].to_numpy()
    groups = {"snapped points, unweighted": (v, None),
              "snapped points, population-weighted": (v, w)}
    for k in ("100 m cell", "merged block"):
        sub = pts.filter(pl.col("kind") == k)
        groups[f"  {k}s, population-weighted"] = (sub["snap"].to_numpy(), sub["pop"].to_numpy())
    groups["100 m cells: total displacement, pop-weighted"] = (
        cells["disp"].to_numpy(), cells["pop"].to_numpy())
    s = stats(groups)
    say("Descriptive statistics, metres ('weight' = points or people):\n")
    say(md(s))
    say()
    tot = float(w.sum())
    rows = [{"threshold": f"> {t:,} m",
             "points": int((v > t).sum()), "% points": float((v > t).mean() * 100),
             "people": float(w[v > t].sum()), "% people": float(w[v > t].sum() / tot * 100)}
            for t in POP_THRESH]
    say("How many were moved further than ...:\n")
    say(md(pl.DataFrame(rows)))
    say()
    kinds = (pts.group_by("kind").agg(pl.len().alias("points"), pl.col("pop").sum().alias("people"),
                                      pl.col("n_raw").sum().alias("100 m cells"))
             .sort("kind"))
    say("Points by kind (merged blocks are areas coarsened because the nearest store is far):\n")
    say(md(kinds, 0))
    say()
    s.write_csv(out / "population_stats.csv")

    labels, lookup = C.rasterize_regions(kreis_gdf, cfg.ghsl, cells.select("cell_id", "row", "col", "pop"),
                                         id_col=cfg.kreis_id, name_col="LABEL")
    lk = lookup.select("region_code", pl.col(cfg.kreis_id).alias("AGS5"), "LABEL")
    kc = (cells.join(labels, on="cell_id", how="left").join(lk, on="region_code", how="left")
          .with_columns(pl.col("AGS5").fill_null("outside")))
    pw = pl.col("pop").cast(pl.Float64)
    kreis = (kc.group_by("AGS5", "LABEL").agg(
                pw.sum().alias("people"),
                ((pl.col("snap") * pw).sum() / pw.sum()).alias("mean_snap_m"),
                ((pl.col("disp") * pw).sum() / pw.sum()).alias("mean_displacement_m"),
                (pw.filter(pl.col("snap") > 250).sum() / pw.sum() * 100).alias("% people > 250 m"))
             .sort("mean_snap_m", descending=True))
    kreis.write_csv(out / "population_by_kreis.csv")
    land = (kc.with_columns(pl.col("AGS5").str.slice(0, 2).alias("ags2"))
            .filter(pl.col("ags2").is_in(list(LAENDER)))
            .group_by("ags2").agg(pw.sum().alias("people"),
                                  ((pl.col("snap") * pw).sum() / pw.sum()).alias("mean_snap_m"),
                                  (pw.filter(pl.col("snap") > 250).sum() / pw.sum() * 100)
                                  .alias("% people > 250 m"))
            .with_columns(pl.col("ags2").replace_strict(LAENDER, default=pl.col("ags2")).alias("Land"))
            .sort("Land").select("Land", "people", "mean_snap_m", "% people > 250 m"))
    say("By Bundesland (population-weighted):\n")
    say(md(land))
    say()
    say("Kreise with the largest population-weighted snap distance "
        "(all in population_by_kreis.csv):\n")
    say(md(kreis.head(12).select("LABEL", "people", "mean_snap_m", "% people > 250 m")))
    say()
    return kreis


# ---------------------------------------------------------------- figures --

def _germany(ax, kreis_gdf, crs=3035, lw=0.15):
    kreis_gdf.to_crs(crs).boundary.plot(ax=ax, color="#BBBBBB", linewidth=lw, zorder=1)
    ax.set_axis_off()
    ax.set_aspect("equal")


def fig_stores(st, rej, kreis_gdf, out: Path):
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    F.set_style()
    # histogram, log-spaced bins, one line per shop type
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-1, np.log10(max(3000.0, float(st["snap_m"].max() or 1) * 1.1)), 70)
    for i, (t, sub) in enumerate(sorted(st.group_by("shop_type_last"), key=lambda x: str(x[0]))):
        t = t[0] if isinstance(t, tuple) else t
        ax.hist(np.clip(sub["snap_m"].to_numpy(), 0.1, None), bins=bins, histtype="step",
                lw=1.6, color=F.PALETTE[i % len(F.PALETTE)], label=f"{t} ({sub.height:,})")
    for b in STORE_BANDS:
        ax.axvline(b, color="#999", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("snap distance, m (log scale; dotted: band edges)")
    ax.set_ylabel("stores")
    ax.set_title("Stores: distance moved onto the road network")
    ax.legend()
    F.save(fig, out / "stores_hist")

    # map of stores that moved >= 50 m, and of the rejected ones
    tf = Transformer.from_crs(4326, 3035, always_xy=True)
    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    _germany(ax, kreis_gdf)
    x, y = tf.transform(st["lon_orig"].to_numpy(), st["lat_orig"].to_numpy())
    ax.scatter(x, y, s=0.3, color="#CFCFCF", lw=0, zorder=2, label=f"all stores ({st.height:,})")
    styles = [(50, 100, "#E69F00", 6), (100, 300, "#D55E00", 12)]
    for lo, hi, col, s in styles:
        m = ((st["snap_m"] >= lo) & (st["snap_m"] < hi)).to_numpy()
        ax.scatter(x[m], y[m], s=s, color=col, lw=0, zorder=3, label=f"{lo}-{hi} m ({m.sum():,})")
    rk = rej.filter(pl.col("lon_panel").is_not_null())
    if rk.height:
        rx, ry = tf.transform(rk["lon_panel"].to_numpy(), rk["lat_panel"].to_numpy())
        ax.scatter(rx, ry, s=28, marker="x", color="#B2182B", lw=1.2, zorder=4,
                   label=f"rejected, > 300 m ({rk.height:,})")
    ax.legend(loc="upper left", fontsize=8, markerscale=1.5)
    ax.set_title("Stores moved >= 50 m to reach a road")
    F.save(fig, out / "stores_map")

    share = (st.filter(pl.col("AGS5").is_not_null())
             .group_by(pl.col("AGS5").cast(pl.Utf8).alias("region"))
             .agg(((pl.col("snap_m") >= 50).mean() * 100).alias("pct50")))
    ax = F.plot_choropleth(share, kreis_gdf.to_crs(3035), value_col="pct50", cmap="Oranges",
                           label="% of stores moved >= 50 m",
                           title="Stores moved >= 50 m, share per Kreis")
    F.save(ax.figure, out / "stores_kreis_share")


def fig_population(pts, cells, kreis, kreis_gdf, bl_gdf, out: Path, road_px: float):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    F.set_style()
    # ECDFs
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, w, lab, col in ((pts["snap"].to_numpy(), None, "snapped points", F.PALETTE[0]),
                           (pts["snap"].to_numpy(), pts["pop"].to_numpy(),
                            "people (snap distance)", F.PALETTE[1]),
                           (cells["disp"].to_numpy(), cells["pop"].to_numpy(),
                            "people (total displacement: coarsening + snap)", F.PALETTE[2])):
        ok = np.isfinite(v)
        v = v[ok]
        w = np.ones_like(v) if w is None else w[ok]
        o = np.argsort(v)
        ax.plot(np.clip(v[o], 0.1, None), np.cumsum(w[o]) / w.sum(), lw=1.8, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("metres (log scale)")
    ax.set_ylabel("cumulative share")
    ax.set_title("Population: distance moved onto the road network")
    ax.legend(loc="lower right", fontsize=8)
    F.save(fig, out / "pop_ecdf")

    # 2 km grid, population-weighted mean snap distance
    px = 2000.0
    grid = (cells.filter(pl.col("snap").is_not_null())
            .with_columns((pl.col("x") // px).cast(pl.Int64).alias("gx"),
                          (pl.col("y") // px).cast(pl.Int64).alias("gy"))
            .group_by("gx", "gy").agg(((pl.col("snap") * pl.col("pop")).sum()
                                       / pl.col("pop").sum()).alias("v")))
    gx, gy = grid["gx"].to_numpy(), grid["gy"].to_numpy()
    A = np.full((gy.max() - gy.min() + 1, gx.max() - gx.min() + 1), np.nan)
    A[gy.max() - gy, gx - gx.min()] = grid["v"].to_numpy()
    ext = (gx.min() * px, (gx.max() + 1) * px, gy.min() * px, (gy.max() + 1) * px)
    fig, ax = plt.subplots(figsize=(7.5, 9.5))
    im = ax.imshow(A, extent=ext, cmap="magma_r", norm=LogNorm(vmin=5, vmax=1000),
                   interpolation="nearest", zorder=1)
    bl_gdf.to_crs(3035).boundary.plot(ax=ax, color="#555", linewidth=0.4, zorder=2)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=0.5, label="population-weighted snap distance, m")
    ax.set_title("Population: snap distance on a 2 km grid")
    F.save(fig, out / "pop_map")

    ax = F.plot_choropleth(kreis.rename({"AGS5": "region"}), kreis_gdf.to_crs(3035),
                           value_col="mean_snap_m", cmap="magma_r",
                           label="population-weighted snap distance, m",
                           title="Population: snap distance per Kreis")
    F.save(ax.figure, out / "pop_kreis")

    # every snapped grid point at road_px metres per pixel
    c2 = pts.select("xn", "yn").unique()
    x, y = c2["xn"].to_numpy(), c2["yn"].to_numpy()
    x0, y0 = np.floor(x.min() / road_px) * road_px, np.floor(y.min() / road_px) * road_px
    nx_ = int((x.max() - x0) // road_px) + 1
    ny_ = int((y.max() - y0) // road_px) + 1
    H = np.zeros((ny_, nx_), np.uint8)
    H[ny_ - 1 - ((y - y0) // road_px).astype(int), ((x - x0) // road_px).astype(int)] = 1
    plt.imsave(out / "road_points_germany.png", H, cmap="Greys", vmin=0, vmax=1)
    say(f"road_points_germany.png: {nx_:,} x {ny_:,} px at {road_px:g} m per pixel")


# ---------------------------------------------------------------- windows --

def load_roads(bbox_ll, cache: Path, roads_file: str | None, fetch: bool):
    """[(highway, lon array, lat array)] for a lon/lat box, or [] if none."""
    if roads_file:
        import geopandas as gpd
        g = gpd.read_file(roads_file).to_crs(4326)
        out = []
        for hw, geom in zip(g.get("highway", [None] * len(g)), g.geometry):
            for part in getattr(geom, "geoms", [geom]):
                xy = np.asarray(part.coords)
                out.append((hw, xy[:, 0], xy[:, 1]))
        return out
    if not fetch:
        return []
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        import requests
        w, s, e, n = bbox_ll
        q = f'[out:json][timeout:180];way["highway"]({s},{w},{n},{e});out geom;'
        try:
            r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q},
                              timeout=240)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            say(f"roads: Overpass request failed ({type(exc).__name__}: {exc}); "
                f"drawing without roads")
            return []
        cache.write_text(json.dumps(data), encoding="utf-8")
    return [(el.get("tags", {}).get("highway"),
             np.array([p["lon"] for p in el["geometry"]]),
             np.array([p["lat"] for p in el["geometry"]]))
            for el in data.get("elements", []) if el.get("geometry")]


def fig_window(name, bbox, pts, cells, st, kreis_sel, out: Path, roads_file, fetch):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    from pyproj import Transformer
    F.set_style()
    x0, y0, x1, y1 = bbox
    to_ll = Transformer.from_crs(3035, 4326, always_xy=True)
    to_m = Transformer.from_crs(4326, 3035, always_xy=True)
    lo, la = to_ll.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    bbox_ll = (min(lo), min(la), max(lo), max(la))
    roads = load_roads(bbox_ll, out / f"roads_{name.lower()}.json", roads_file, fetch)

    fig, ax = plt.subplots(figsize=(10, 10))
    segs_car, segs_other = [], []
    for hw, lon, lat in roads:
        x, y = to_m.transform(lon, lat)
        (segs_car if hw in CAR else segs_other).append(np.column_stack([x, y]))
    if segs_other:
        ax.add_collection(LineCollection(segs_other, colors="#C8C8C8", linewidths=0.4, zorder=1))
    if segs_car:
        ax.add_collection(LineCollection(segs_car, colors="#444444", linewidths=0.8, zorder=2))
    inside = lambda df, xc, yc: df.filter(pl.col(xc).is_between(x0, x1)  # noqa: E731
                                          & pl.col(yc).is_between(y0, y1))
    p = inside(pts, "x1", "y1")
    vec = [[(a, b), (c, d)] for a, b, c, d in p.select("x1", "y1", "xn", "yn").iter_rows()]
    ax.add_collection(LineCollection(vec, colors="#2C7BB6", linewidths=0.5, alpha=0.7, zorder=3))
    c = inside(cells, "x", "y")
    sc = ax.scatter(c["x"], c["y"], c=np.clip(c["snap"].to_numpy(), 1, None), s=3,
                    cmap="magma_r", norm=LogNorm(vmin=5, vmax=500), lw=0, zorder=4)
    n = inside(p, "xn", "yn")
    ax.scatter(n["xn"], n["yn"], s=4, color="black", lw=0, zorder=5)
    s = st.with_columns(pl.Series("xo", to_m.transform(st["lon_orig"].to_numpy(),
                                                        st["lat_orig"].to_numpy())[0]),
                        pl.Series("yo", to_m.transform(st["lon_orig"].to_numpy(),
                                                        st["lat_orig"].to_numpy())[1]))
    s = inside(s, "xo", "yo")
    for r in s.iter_rows(named=True):
        ax.annotate("", xy=(r["x"], r["y"]), xytext=(r["xo"], r["yo"]), zorder=6,
                    arrowprops=dict(arrowstyle="->", color="#D55E00", lw=1.0))
    ax.scatter(s["xo"], s["yo"], s=22, facecolor="white", edgecolor="#D55E00", lw=1, zorder=7)
    if kreis_sel is not None:
        kreis_sel.to_crs(3035).boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=2)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.colorbar(sc, ax=ax, shrink=0.45, label="snap distance of the cell's point, m")
    road_note = (f"{len(segs_car):,} drivable / {len(segs_other):,} other OSM ways"
                 if roads else "no road layer (use --roads or --roads-file)")
    ax.set_title(f"{name}: 100 m cells (colour) and their snap vectors (blue) to solve "
                 f"nodes (black);\nstores: original (circle) -> snapped (arrow). {road_note}",
                 fontsize=9)
    F.save(fig, out / f"window_{name.lower()}", formats=("png", "pdf"))


# ------------------------------------------------------------------- main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Snapping diagnostics for stores and population.")
    ap.add_argument("config", nargs="?", default="runs/baseline.toml")
    ap.add_argument("--osrm", default=None,
                    help="OSRM URL: also measure the rejected stores' distance")
    ap.add_argument("--roads", action="store_true",
                    help="download OSM roads for the city windows (Overpass, cached)")
    ap.add_argument("--roads-file", default=None,
                    help="a local road layer (GeoJSON/GPKG with a 'highway' column) instead")
    ap.add_argument("--window", action="append", default=[],
                    help="extra window: name,lon,lat (6 x 6 km), repeatable")
    ap.add_argument("--road-px", type=float, default=100.0,
                    help="metres per pixel of road_points_germany.png (default 100)")
    a = ap.parse_args(argv)

    cfg = RunConfig.load(a.config).pipeline_config()
    out = cfg.out / cfg.tag / "snapping"
    out.mkdir(parents=True, exist_ok=True)
    panel = P.load_panel(cfg, shop_types=None)
    kreis_gdf, bl_gdf = P.region_layers(cfg, panel)

    say(f"# Snapping report ({cfg.tag})")
    say()
    st, rej = store_frames(cfg, panel, a.osrm)
    report_stores(st, rej, out)
    pts, cells, source = cell_frames(cfg)
    kreis = report_population(cfg, pts, cells, source, kreis_gdf, out)
    fig_stores(st, rej, kreis_gdf, out)
    fig_population(pts, cells, kreis, kreis_gdf, bl_gdf, out, a.road_px)

    wins = []
    for city, ags5 in cfg.cities.items():
        sel = kreis_gdf[kreis_gdf[cfg.kreis_id].astype(str) == ags5]
        if sel.empty:
            say(f"window: {city} ({ags5}) not in the Kreis layer, skipped")
            continue
        sel = P.main_area(sel, 3035, id_col=cfg.kreis_id)
        wins.append((city, F.centre_bbox(sel, cfg.kreis_id, ags5, half_km=WINDOW_HALF_KM,
                                          crs=3035), sel))
    from pyproj import Transformer
    to_m = Transformer.from_crs(4326, 3035, always_xy=True)
    for w in a.window:
        name, lon, lat = w.split(",")
        x, y = to_m.transform(float(lon), float(lat))
        h = WINDOW_HALF_KM * 1000
        wins.append((name, (x - h, y - h, x + h, y + h), None))
    for name, bbox, sel in wins:
        fig_window(name, bbox, pts, cells, st, sel, out, a.roads_file, a.roads)

    (out / "summary.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())