"""
Why are so many solve nodes uncertified, and where do grid points snap far?

    python -m scripts.diagnose_solve runs/baseline.toml                      # Supermarkt
    python -m scripts.diagnose_solve runs/baseline.toml --type "Discount Geschäft"
    python -m scripts.diagnose_solve runs/baseline.toml --verify 2000        # needs OSRM
    python -m scripts.diagnose_solve runs/baseline.toml --osm-sample 300     # needs internet

Reads what the solve wrote (cells_solve, node_regions, matrix.parquet); routes
nothing unless --verify is given. Writes to out/<grid>/diagnostics/<type>/:
summary.md (what is printed), CSVs and a Kreis map.

Certification (network_voronoi_2.certified): node i's winner is proven when
    t1_i <= d_k,i / v_max,
t1 the best routed time, d_k the straight-line distance of the k-th candidate.
Everything below measures how far each node is from that line and why:

  slack      v_req = d_k / t1: the straight-line speed a store OUTSIDE the
             candidate set would need to beat the winner. Certified iff
             v_req >= v_max (180 km/h).
  envelope   from every routed pair: the fastest time ever observed to a store
             at least d metres away, T_min(d). Nodes with t1 <= T_min(d_k) are
             "empirically certified": beaten only by a trip faster than any of
             ~10^8 measured ones. An observation, not a proof.
  verify     a population-weighted sample of uncertified nodes is routed to
             EVERY store within v_max * t1 (the full set the bound leaves open):
             a sound check of how often the winner actually changes.
  osm        a sample of 100 m cells that snapped far is checked against current
             OpenStreetMap (Overpass): is there a car road nearer than OSRM's
             snap, only a non-car way (track, agricultural, private), or nothing?
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import scripts.isochrones.network_voronoi_2 as nv
import scripts.run_pipeline as P
from scripts.runconfig import RunConfig

_LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def md(df: pl.DataFrame, digits: int = 1) -> str:
    def f(x):
        if isinstance(x, float):
            return "" if x != x else f"{x:,.{digits}f}"
        return f"{x:,}" if isinstance(x, int) else str(x)
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(f(x) for x in r) + " |" for r in df.iter_rows()])


def wq(v, w, ps=(0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float]:
    """Weighted quantiles of finite v."""
    ok = np.isfinite(v) & (w > 0)
    v, w = v[ok], w[ok]
    if not v.size:
        return [float("nan")] * len(ps)
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return [float(v[o][min(np.searchsorted(cw, p), v.size - 1)]) for p in ps]


# ==========================================================================
# 1. one pass over the matrix: per-node quantities + the speed envelope
# ==========================================================================

def scan(E: np.ndarray, D: np.ndarray, k0: int, *, vmax: float, bin_m: float = 250.0,
         max_m: float = 80_000.0, step: int = 200_000) -> tuple[dict, dict, dict]:
    """Per node: t1, winner column and distance, d_k (final and at k0), nearest
    distance, width. Envelope per distance bin over the plausible pairs. Pairs
    faster than v_max in a straight line are returned apart: no road allows them,
    so they are errors in the routing graph, not observations."""
    n, K = E.shape
    nb = int(max_m // bin_m) + 1
    env = {"tmin": np.full(nb, np.inf), "vmax": np.zeros(nb), "n": np.zeros(nb, np.int64)}
    r = {c: np.empty(n, np.float32) for c in ("t1", "e_win", "dk", "t1_k0", "dk0", "e1")}
    r["width"] = np.empty(n, np.int32)
    r["j1"] = np.empty(n, np.int32)
    fast = {"row": [], "col": [], "e": [], "t": []}
    V = vmax / 3.6
    for a in range(0, n, step):
        e, d = E[a:a + step], np.nan_to_num(D[a:a + step], nan=np.inf, posinf=np.inf)
        j = d.argmin(1)
        r["j1"][a:a + step] = j
        r["t1"][a:a + step] = d[np.arange(len(d)), j]
        r["e_win"][a:a + step] = e[np.arange(len(e)), j]
        r["dk"][a:a + step] = e[:, -1]
        r["t1_k0"][a:a + step] = d[:, :k0].min(1)
        r["dk0"][a:a + step] = e[:, min(k0, K) - 1]
        r["e1"][a:a + step] = e[:, 0]
        # refine() pads unrefined rows with inf times at the repeated k-th distance
        pad = np.isinf(d) & (e == e[:, -1:])
        rev = pad[:, ::-1]
        r["width"][a:a + step] = K - np.where(rev.all(1), K, rev.argmin(1))
        bad = np.isfinite(d) & (e > d * V)          # e / t > v_max (t = 0 included)
        bad &= e > 0
        if bad.any():
            rr, cc = np.nonzero(bad)
            for k_, v_ in (("row", rr + a), ("col", cc), ("e", e[rr, cc]), ("t", d[rr, cc])):
                fast[k_].append(v_)
        ok = np.isfinite(d) & (d > 0) & ~bad
        ee, tt = e[ok].astype(np.float64), d[ok].astype(np.float64)
        b = np.minimum((ee // bin_m).astype(np.int64), nb - 1)
        np.minimum.at(env["tmin"], b, tt)
        np.maximum.at(env["vmax"], b, ee / tt)
        env["n"] += np.bincount(b, minlength=nb)
    # T_min(d): fastest observed time to any store AT LEAST d away (suffix minimum)
    env["tsuf"] = np.minimum.accumulate(env["tmin"][::-1])[::-1]
    env["bin_m"] = bin_m
    fast = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in fast.items()}
    return r, env, fast


def t_min(env: dict, d: np.ndarray) -> np.ndarray:
    b = np.minimum((d // env["bin_m"]).astype(np.int64), len(env["tsuf"]) - 1)
    return env["tsuf"][b]


# ==========================================================================
# 2. report
# ==========================================================================

def report(r: dict, env: dict, pop: np.ndarray, kreis: pl.DataFrame, k0: int,
           vmax: float, out: Path) -> pl.DataFrame:
    V = vmax / 3.6
    t1, dk = r["t1"].astype(np.float64), r["dk"].astype(np.float64)
    reach = np.isfinite(t1)
    cert = reach & (t1 <= dk / V)
    cert0 = np.isfinite(r["t1_k0"]) & (r["t1_k0"] <= r["dk0"] / V)
    P_ = pop.sum()
    share = lambda m: (f"{m.mean():.1%} of nodes, {pop[m].sum() / P_:.1%} of population")  # noqa: E731

    say("## 1. Certification as stored")
    say(f"- nodes: {t1.size:,}, population {P_:,.0f}")
    say(f"- no finite time to any candidate (unreachable): {share(~reach)}")
    say(f"- certified at the first pass, k={k0}: {share(cert0)}")
    say(f"- certified at the final width: {share(cert)}")
    w = pl.DataFrame({"width": r["width"], "cert": cert, "pop": pop})
    say("\nFinal candidate width (how far refine() had to go):\n")
    say(md(w.group_by("width").agg(pl.len().alias("nodes"),
                                   (pl.col("pop").sum() / P_ * 100).alias("% pop"),
                                   (pl.col("cert").mean() * 100).alias("% certified"))
           .sort("width")))

    say("\n## 2. Slack: what an outside store would have to achieve")
    say("v_req = d_k / t1, straight-line km/h. Certified iff v_req >= "
        f"{vmax:.0f}.\n")
    with np.errstate(divide="ignore", invalid="ignore"):
        vreq = 3.6 * dk / t1
    unc = reach & ~cert
    rows = [{"v_req at least (km/h)": t,
             "% of uncertified nodes": float((vreq[unc] >= t).mean() * 100),
             "% of uncertified pop": float(pop[unc & (vreq >= t)].sum() / max(pop[unc].sum(), 1) * 100)}
            for t in (40, 60, 80, 100, 120, 140, 160)]
    say(md(pl.DataFrame(rows)))

    say("\n## 3. Why: the winner is near and slow, the bound assumes 180 km/h")
    say("Population-weighted quantiles (p10 / p25 / p50 / p75 / p90):\n")
    with np.errstate(divide="ignore", invalid="ignore"):
        veff = 3.6 * r["e_win"].astype(np.float64) / t1
    qs = []
    for name, v in (("t1, s", t1), ("straight-line km/h to winner", veff),
                    ("d_1 nearest store, m", r["e1"].astype(np.float64)),
                    ("d_k final, m", dk), ("d_k / d_1", dk / np.maximum(r["e1"], 1.0))):
        for lab, m in (("certified", cert), ("uncertified", unc)):
            qs.append({"quantity": name, "nodes": lab,
                       **dict(zip(("p10", "p25", "p50", "p75", "p90"), wq(v[m], pop[m])))})
    say(md(pl.DataFrame(qs)))

    say("\n## 4. Observed speed envelope (routed pairs at or below v_max; see 4b)")
    bm = env["bin_m"]
    erows = []
    for d in (500, 1_000, 2_000, 5_000, 10_000, 20_000, 40_000):
        i = int(d // bm)
        lo = slice(i, min(i + max(1, int(1000 // bm)), len(env["n"])))
        erows.append({"distance m": d, "pairs (1 km band)": int(env["n"][lo].sum()),
                      "max straight-line km/h": float(env["vmax"][lo].max() * 3.6),
                      "T_min(d) s": float(env["tsuf"][i]), "d/v_max s": d / V})
    say(md(pl.DataFrame(erows)))
    tlb = t_min(env, dk)
    say("\nCertified share under alternative bounds (only the first is a proof "
        "if v_max is the profile's true maximum):\n")
    alt = [("v_max = 140 km/h (sound iff car.lua caps links at 140)", t1 <= dk / (140 / 3.6)),
           ("observed envelope T_min(d_k)", t1 <= tlb),
           ("observed envelope with 20 % margin", t1 <= 0.8 * tlb)]
    say(md(pl.DataFrame([{"bound": n_, "% nodes": float((reach & m).mean() * 100),
                          "% pop": float(pop[reach & m].sum() / P_ * 100)} for n_, m in alt])))

    say("\n## 5. Where")
    nd = pl.DataFrame({"node_id": np.arange(t1.size), "pop": pop, "unc": ~cert,
                       "d1": r["e1"].astype(np.float64)})
    band = (pl.when(pl.col("d1") < 500).then(pl.lit("a <500 m"))
            .when(pl.col("d1") < 1_000).then(pl.lit("b 0.5-1 km"))
            .when(pl.col("d1") < 2_000).then(pl.lit("c 1-2 km"))
            .when(pl.col("d1") < 5_000).then(pl.lit("d 2-5 km")).otherwise(pl.lit("e >5 km")))
    say("By straight-line distance to the nearest store of this type:\n")
    say(md(nd.with_columns(band.alias("d_1")).group_by("d_1").agg(
        pl.len().alias("nodes"), (pl.col("pop").sum() / P_ * 100).alias("% pop"),
        (pl.col("pop").filter("unc").sum() / pl.col("pop").sum() * 100).alias("% pop uncertified"))
        .sort("d_1")))
    kr = (nd.join(kreis, on="node_id", how="left")
          .group_by("kreis").agg(pl.col("pop").sum().alias("people"),
                                 pl.col("area").first(), pl.col("kpop").first(),
                                 (pl.col("pop").filter("unc").sum() / pl.col("pop").sum() * 100)
                                 .alias("% pop uncertified"))
          .with_columns((pl.col("kpop") / pl.col("area")).alias("density"))
          .sort("% pop uncertified", descending=True))
    kr.write_csv(out / "uncertified_by_kreis.csv")
    say("\nBy Kreis population density (inhabited area), quintiles:\n")
    say(md(kr.drop_nulls("density").with_columns(
        (pl.col("density").rank("ordinal") * 5 / pl.col("density").len()).ceil()
        .cast(pl.Int32).alias("density quintile"))
        .group_by("density quintile").agg(
            pl.len().alias("Kreise"), pl.col("density").median().alias("median people/km2"),
            ((pl.col("% pop uncertified") * pl.col("people")).sum() / pl.col("people").sum())
            .alias("% pop uncertified")).sort("density quintile")))
    return pl.DataFrame({"t1": t1, "dk": dk, "cert": cert, "reach": reach})


def fast_pairs(fast: dict, r: dict, idx, c2, frame, pop, kreis, vmax: float, osrm: str,
               out: Path, n_route: int = 15) -> None:
    """Pairs OSRM routes faster than v_max in a straight line: where, how many
    winners rest on one, and (OSRM /route with steps) which segment does it."""
    say(f"\n## 4b. Pairs faster than {vmax:.0f} km/h in a straight line")
    if not fast["row"].size:
        say("None: no routed pair beats the bound, so v_max is consistent with the graph.")
        return
    row, col = fast["row"].astype(np.int64), fast["col"].astype(np.int64)
    sid = idx[row, col]
    win = r["j1"][row] == col                    # the node's winner IS a fast pair
    nodes, wnodes = np.unique(row), np.unique(row[win])
    with np.errstate(divide="ignore"):
        v = 3.6 * fast["e"] / fast["t"]
    say(f"- {row.size:,} pairs, {nodes.size:,} nodes ({pop[nodes].sum():,.0f} people), "
        f"{np.unique(sid).size:,} stores")
    say(f"- nodes whose WINNER is such a pair: {wnodes.size:,} ({pop[wnodes].sum():,.0f} "
        f"people). Their assignment and certification rest on an impossible time.")
    say(f"- straight-line speed: median {np.median(v):,.0f} km/h, max {np.max(v):,.0f}; "
        f"straight-line distance median {np.median(fast['e']):,.0f} m")
    fx, fy = frame["lon"].to_numpy(), frame["lat"].to_numpy()
    cx, cy = c2["lon"].to_numpy(), c2["lat"].to_numpy()
    df = pl.DataFrame({"node_id": row, "store_id": frame["store_id"].to_numpy()[sid],
                       "is_winner": win, "euclid_m": fast["e"], "dur_s": fast["t"],
                       "kmh": v, "node_lon": cx[row], "node_lat": cy[row],
                       "store_lon": fx[sid], "store_lat": fy[sid], "node_pop": pop[row]})
    df = df.join(kreis.select("node_id", "kreis"), on="node_id", how="left")
    df.write_csv(out / "fast_pairs.csv")
    by_store = (df.group_by("store_id").agg(
        pl.len().alias("pairs"), pl.col("is_winner").sum().alias("as winner"),
        pl.col("node_pop").filter("is_winner").sum().alias("people (as winner)"),
        pl.col("kmh").max().alias("max km/h"), pl.col("kreis").mode().first(),
        pl.col("store_lat").first(), pl.col("store_lon").first())
        .sort("pairs", descending=True))
    by_store.write_csv(out / "fast_pairs_by_store.csv")
    say("\nStores with the most such pairs (all in fast_pairs_by_store.csv):\n")
    say(md(by_store.head(15), 3))

    # which segment: /route with steps for each top store's fastest pair
    import requests
    top = (df.filter(pl.col("store_id").is_in(by_store.head(n_route)["store_id"].implode()))
           .sort("kmh", descending=True).unique("store_id", keep="first"))
    rows_ = []
    with requests.Session() as s:
        for t in top.iter_rows(named=True):
            url = (f"{osrm.rstrip('/')}/route/v1/driving/{t['node_lon']:.6f},{t['node_lat']:.6f};"
                   f"{t['store_lon']:.6f},{t['store_lat']:.6f}")
            try:
                j = s.get(url, params={"steps": "true", "overview": "false"}, timeout=60).json()
                route = j["routes"][0]
            except Exception as exc:
                say(f"  /route failed ({type(exc).__name__}); is OSRM up? Skipping the segment lookup.")
                break
            steps = route["legs"][0]["steps"]
            spd = [(st["distance"] / st["duration"] * 3.6 if st["duration"] > 0
                    else (np.inf if st["distance"] > 0 else 0.0)) for st in steps]
            k = int(np.argmax(spd))
            st = steps[k]
            lon, lat = st["maneuver"]["location"]
            rows_.append({"store_id": t["store_id"], "route m": round(route["distance"]),
                          "route s": round(route["duration"], 1),
                          "segment": st.get("name") or st.get("ref") or "(unnamed)",
                          "mode": st.get("mode", ""), "segment m": round(st["distance"]),
                          "segment s": round(st["duration"], 1),
                          "segment km/h": spd[k],
                          "osm": f"https://www.openstreetmap.org/#map=17/{lat:.5f}/{lon:.5f}"})
    if rows_:
        say("\nFastest step on each of those routes (OSRM /route; mode 'ferry' or 'train' "
            "points to a timetable or duration tag, 'driving' to a maxspeed tag):\n")
        say(md(pl.DataFrame(rows_), 0))


def kreis_map(kr_csv: Path, cfg, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kre, _ = P.region_layers(cfg)
    v = pl.read_csv(kr_csv, schema_overrides={"kreis": pl.Utf8})
    g = kre.merge(v.to_pandas(), left_on=cfg.kreis_id, right_on="kreis", how="left")
    ax = g.plot(column="% pop uncertified", cmap="magma_r", legend=True, figsize=(6, 8),
                edgecolor="white", linewidth=0.1, missing_kwds={"color": "#ddd"})
    ax.set_axis_off()
    ax.set_title("Population with an uncertified winner, % per Kreis")
    plt.savefig(out / "uncertified_by_kreis.png", dpi=200, bbox_inches="tight")
    plt.close()


# ==========================================================================
# 3. verification: route every store the bound leaves open, for a sample
# ==========================================================================

def ball_open(c2, frame, idx, D, rows, t1, vmax):
    """Per sampled row: stores within v_max * t1 not already routed for it."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([frame["x"].to_numpy(), frame["y"].to_numpy()]))
    cxy = np.column_stack([c2["x"].to_numpy()[rows], c2["y"].to_numpy()[rows]])
    balls = tree.query_ball_point(cxy, r=t1[rows] * vmax / 3.6, workers=-1)
    return [np.setdiff1d(np.asarray(b, np.int32), idx[r][np.isfinite(D[r])])
            for b, r in zip(balls, rows)]


def verify(c2, frame, idx, D, t1, pop, rows, *, vmax, osrm, max_workers) -> dict:
    new = ball_open(c2, frame, idx, D, rows, t1, vmax)
    m = max(1, max(len(x) for x in new))
    sidx = np.empty((rows.size, m), np.int32)
    for i, x in enumerate(new):           # empty or short rows padded with the winner
        fill = x if x.size else idx[rows[i], :1]
        sidx[i] = np.resize(fill, m)
    valid = np.arange(m)[None, :] < np.array([len(x) for x in new])[:, None]
    fx, fy = frame["x"].to_numpy(), frame["y"].to_numpy()
    cx, cy = c2["x"].to_numpy()[rows], c2["y"].to_numpy()[rows]
    dist = np.hypot(cx[:, None] - fx[sidx], cy[:, None] - fy[sidx]).astype(np.float32)
    o = np.argsort(dist, axis=1)
    sidx, dist, valid = (np.take_along_axis(a, o, 1) for a in (sidx, dist, valid))
    pf = nv.preflight(c2[rows], base_url=osrm)
    if pf["max_coords"] < 1000:
        raise RuntimeError("verify: OSRM rejects even a 50 x 50 /table (preflight). Is "
                           "osrm-routed up, idle, and started with --max-table-size 10000?")
    # small blocks: one block is one thread's run of /table calls; 25 neighbouring
    # nodes keep each call small and spread the work over every worker
    sd, _ = nv.travel_times(c2[rows], frame, nv.Candidates(sidx, dist, frame["store_id"].to_numpy().astype(str)),
                            base_url=osrm, max_coords=pf["max_coords"], block=25,
                            max_workers=max_workers, progress_every=20)
    # a failed block leaves its rows at inf: those rows are NOT verified
    done = ~(valid & np.isinf(sd)).any(1)
    best = np.where(valid, sd, np.inf).min(1)
    changed = done & (best < t1[rows] - 0.5)    # 0.5 s: equal-time ties are not changes
    return {"rows": int(done.sum()), "failed": int((~done).sum()), "pairs": int(valid.sum()),
            "changed": int(changed.sum()), "pop": float(pop[rows][done].sum()),
            "pop_changed": float(pop[rows][changed].sum()),
            "max_gain_s": float(np.max(t1[rows][changed] - best[changed], initial=0.0))}


# ==========================================================================
# 4. far snaps against current OpenStreetMap
# ==========================================================================

#: highway values the stock car.lua routes on, and access values it refuses
CAR_HW = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
          "secondary", "secondary_link", "tertiary", "tertiary_link", "unclassified",
          "residential", "living_street", "service", "road"}
NO_CAR = {"no", "private", "agricultural", "forestry", "emergency", "psv", "delivery",
          "customers"}


def car_ok(tags: dict) -> bool:
    """Approximately stock car.lua: a car highway class, and the most specific
    access tag present not on the blacklist."""
    if tags.get("highway") not in CAR_HW:
        return False
    for k in ("motorcar", "motor_vehicle", "vehicle", "access"):
        if k in tags:
            return tags[k] not in NO_CAR
    return True


def classify(resp: dict, x: float, y: float, snap: float, to3035) -> dict:
    """Nearest car road / any way to (x, y) in EPSG:3035, and the verdict."""
    from shapely.geometry import LineString, Point
    p = Point(x, y)
    d_car = d_any = np.inf
    near_tags: dict = {}
    n_bld = 0
    for el in resp.get("elements", []):
        if el.get("type") == "count":
            n_bld = int(el.get("tags", {}).get("ways", 0))
            continue
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        lx, ly = to3035([q["lon"] for q in g], [q["lat"] for q in g])
        d = LineString(np.column_stack([lx, ly])).distance(p)
        t = el.get("tags", {})
        if car_ok(t):
            d_car = min(d_car, d)
        if d < d_any:
            d_any, near_tags = d, t
    if d_car + 50 < snap:
        verdict = "car road nearer than OSRM's snap"
    elif d_any <= 100:
        verdict = "only a non-car way nearby"
    elif n_bld:
        verdict = "buildings, no mapped way within 100 m"
    else:
        verdict = "no way, no building within 100 m"
    acc = next((f"{k}={near_tags[k]}" for k in ("motorcar", "motor_vehicle", "vehicle", "access")
                if k in near_tags), "")
    return {"d_car": d_car, "d_any": d_any, "buildings_100m": n_bld, "verdict": verdict,
            "nearest_way": f"{near_tags.get('highway', '')} {acc}".strip()}


#: public Overpass instances, tried in order; the first answering one is kept
OVERPASS = ("https://overpass-api.de/api/interpreter",
            "https://overpass.private.coffee/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter")


def overpass(session, q: str, state: dict) -> dict | None:
    """One query. 429/50x: wait and retry; other refusals: next instance.
    Prints the status and the server's own message, so a refusal is legible."""
    for _ in range(len(OVERPASS) * 3):
        url = OVERPASS[state["i"] % len(OVERPASS)]
        try:
            r = session.post(url, data={"data": q}, timeout=90)
        except Exception as exc:
            print(f"  {url}: {type(exc).__name__}; next instance")
            state["i"] += 1
            continue
        if r.ok:
            return r.json()
        msg = " ".join(r.text.split())[:160]
        if r.status_code in (429, 502, 503, 504):
            state["wait"] = min(state.get("wait", 5) * 2, 120)
            print(f"  {url}: HTTP {r.status_code} (busy), waiting {state['wait']} s: {msg}")
            time.sleep(state["wait"])
        else:
            print(f"  {url}: HTTP {r.status_code}, trying the next instance: {msg}")
            state["i"] += 1
    return None


def osm_check(cfg, n: int, min_snap: float, out: Path, seed: int = 0) -> None:
    import requests
    from pyproj import Transformer
    from scripts.snapping_report import cell_frames
    say(f"\n## 7. Grid points that snapped > {min_snap:.0f} m, against current OSM")
    pts, _, source = cell_frames(cfg)
    say(f"(snap distances: {source})")
    far = pts.filter(pl.col("snap") > min_snap)
    say(md(far.group_by("kind").agg(pl.len().alias("points"),
                                    pl.col("pop").sum().alias("people"))
           .with_columns((pl.col("people") / pts["pop"].sum() * 100).alias("% of all people"))))
    say("Merged blocks snap far by construction (a coarse block's centroid often "
        "lies between settlements); the check below uses 100 m cells only.\n")
    cand = far.filter(pl.col("kind") == "100 m cell")
    if cand.is_empty():
        return
    w = cand["pop"].to_numpy().astype(float)
    pick = np.random.default_rng(seed).choice(cand.height, min(n, cand.height), replace=False,
                                              p=w / w.sum())
    s = cand[np.sort(pick)]
    to_ll = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    t3035 = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
    lon, lat = to_ll.transform(s["x1"].to_numpy(), s["y1"].to_numpy())
    cache = out / "osm_cache"
    cache.mkdir(exist_ok=True)
    rows, fails, state = [], 0, {"i": 0}
    session = requests.Session()
    # Overpass instances refuse anonymous scripted clients; say who is asking
    session.headers["User-Agent"] = "edeka-project diagnose_solve (research; python-requests)"
    for i, row in enumerate(s.iter_rows(named=True)):
        f = cache / f"{row['c1']}.json"
        if f.exists():
            resp = json.loads(f.read_text(encoding="utf-8"))
        else:
            R = int(min(max(row["snap"] + 200, 300), 2000))
            q = (f"[out:json][timeout:60];way(around:{R},{lat[i]:.6f},{lon[i]:.6f})[highway];"
                 f"out tags geom;way(around:100,{lat[i]:.6f},{lon[i]:.6f})[building];out count;")
            resp = overpass(session, q, state)
            if resp is None:
                fails += 1
                if fails >= 5:
                    say(f"Overpass refused 5 points in a row; stopping at {i}/{s.height}. "
                        f"Answers so far are cached in {cache}: re-run to resume.")
                    break
                continue
            fails, state["wait"] = 0, 5
            f.write_text(json.dumps(resp), encoding="utf-8")
            time.sleep(1.0)                       # Overpass fair use
        rows.append({"c1": row["c1"], "lon": lon[i], "lat": lat[i], "pop": row["pop"],
                     "snap_m": row["snap"],
                     **classify(resp, row["x1"], row["y1"], row["snap"], t3035)})
        if (i + 1) % 50 == 0:
            print(f"  osm: {i + 1}/{s.height}")
    if not rows:
        return
    res = pl.DataFrame(rows)
    res.write_csv(out / "far_snaps_osm.csv")
    say(f"Sample of {res.height} cells, drawn population-weighted:\n")
    say(md(res.group_by("verdict").agg(pl.len().alias("cells"), pl.col("pop").sum().alias("people"),
                                       pl.col("snap_m").median().alias("median snap m"))
           .with_columns((pl.col("people") / pl.col("people").sum() * 100).alias("% of sample pop"))
           .sort("people", descending=True)))
    say("\nNearest way where it is not a car road:\n")
    say(md(res.filter(pl.col("verdict") == "only a non-car way nearby")
           .group_by("nearest_way").agg(pl.len().alias("cells")).sort("cells", descending=True)
           .head(12), 0))


# ==========================================================================

def certification(cfg, a, out: Path, osrm: str) -> None:
    """Sections 1-6: read the matrix, report, and optionally verify."""
    mpath = cfg.solvedir(a.type) / "matrix.parquet"
    c2 = pl.read_parquet(cfg.griddir / "cells_solve.parquet")
    nr = pl.read_parquet(cfg.griddir / "node_regions.parquet")
    # each node's main Kreis, and Kreis population / inhabited area for density
    kt = nr.group_by("kreis").agg(pl.col("pop").sum().alias("kpop"),
                                  pl.col("area_km2").sum().alias("area"))
    kreis = (nr.sort(["node_id", "pop"], descending=[False, True])
             .unique("node_id", keep="first", maintain_order=True)
             .select("node_id", "kreis").join(kt, on="kreis", how="left"))

    z = nv._read_matrix(mpath, ("euclid_m", "durations", "store_idx"))
    if not np.array_equal(z["cell_id"], c2["cell_id"].to_numpy()):
        raise ValueError(f"{mpath} does not index this grid's solve nodes")
    E, D = z["euclid_m"], z["durations"]
    say(f"matrix {mpath}: {E.shape[0]:,} nodes x {E.shape[1]} candidates; "
        f"v_max {nv.V_MAX_KMH:.0f} km/h; first pass k={cfg.k}\n")
    pop = c2["pop"].cast(pl.Float64).to_numpy()
    r, env, fast = scan(E, D, cfg.k, vmax=nv.V_MAX_KMH)
    res = report(r, env, pop, kreis, cfg.k, nv.V_MAX_KMH, out)
    kreis_map(out / "uncertified_by_kreis.csv", cfg, out)
    stores = P._norm_ids(pl.read_parquet(cfg.snapdir / "stores.parquet"))
    saved = nv.saved_store_ids(mpath)
    frame, _ = P.matrix_frame(stores, saved, saved)
    fast_pairs(fast, r, z["store_idx"], c2, frame, pop, kreis, nv.V_MAX_KMH, osrm, out)

    if a.verify:
        unc = np.flatnonzero(res["reach"].to_numpy() & ~res["cert"].to_numpy())
        t1 = res["t1"].to_numpy()
        w = pop[unc] / pop[unc].sum()
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(unc, min(a.verify, unc.size), replace=False, p=w))
        # what certifying ALL uncertified nodes by routing their open ball would cost
        est = rng.choice(unc, min(50_000, unc.size), replace=False)
        sizes = np.array([len(x) for x in ball_open(c2, frame, z["store_idx"], D, est, t1,
                                                    nv.V_MAX_KMH)])
        say("\n## 6. Verification against every store the bound leaves open")
        say(f"Open stores per uncertified node (beyond its candidates): median "
            f"{np.median(sizes):.0f}, p90 {np.percentile(sizes, 90):.0f}; certifying all "
            f"{unc.size:,} uncertified nodes this way routes ~{sizes.mean() * unc.size:,.0f} pairs.")
        v = verify(c2, frame, z["store_idx"], D, t1, pop, rows, vmax=nv.V_MAX_KMH, osrm=osrm,
                   max_workers=cfg.max_workers)
        if v["failed"]:
            say(f"WARNING {v['failed']:,} sampled nodes could not be routed and are left out.")
        if not v["rows"]:
            say("No sampled node was routed completely: no verdict.")
            return
        ub = 3.0 / v["rows"] if not v["changed"] else None
        say(f"Sample: {v['rows']:,} uncertified nodes verified (population-weighted), "
            f"{v['pairs']:,} pairs.")
        say(f"Winner changed: {v['changed']:,} nodes, {v['pop_changed']:,.0f} of "
            f"{v['pop']:,.0f} people; largest gain {v['max_gain_s']:.0f} s."
            + (f" No change: 95 % upper bound on the change rate {ub:.2%}." if ub else ""))
    del E, D, z


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("config")
    ap.add_argument("--type", default="Supermarkt", help="shop type whose matrix to read")
    ap.add_argument("--verify", type=int, default=0,
                    help="route this many uncertified nodes against every open store (OSRM)")
    ap.add_argument("--osm-sample", type=int, default=0,
                    help="check this many far-snapped cells against OSM (Overpass)")
    ap.add_argument("--min-snap", type=float, default=250.0)
    ap.add_argument("--osrm", default=None)
    ap.add_argument("--osm-only", action="store_true",
                    help="only the OSM check (section 7); skip reading the matrix")
    a = ap.parse_args(argv)

    rc = RunConfig.load(a.config)
    cfg = dataclasses.replace(rc.pipeline_config(), land=None)    # the national solve
    osrm = a.osrm or cfg.osrm
    out = cfg.out / cfg.tag / "diagnostics" / P.slug(a.type)
    out.mkdir(parents=True, exist_ok=True)
    say(f"# Solve diagnostics: {a.type}, grid {cfg.grid}")

    if not a.osm_only:
        certification(cfg, a, out, osrm)

    if a.osm_sample:
        osm_check(cfg, a.osm_sample, a.min_snap, out)
    (out / ("summary_osm.md" if a.osm_only else "summary.md")).write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
