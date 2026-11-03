"""
Compose per-shop-type solves into any market definition.

The assignment is a minimum over a set of stores, and a minimum over a union is
the minimum of the per-set minima:

    nearest(S ∪ D)      = argmin( nearest(S), nearest(D) )
    nearest(S ∪ D ∪ Dr) = argmin( nearest(S), nearest(D), nearest(Dr) )

So the road work is done once per *shop type* and every market definition is an
argmin over the results -- no routing, no candidate sets, no OSRM. A spec stops
being a solve and becomes a reporting concept.

This holds for the isochrone frame too, which is the part worth checking rather
than assuming: the k nearest stores of a union are always a subset of the union
of the per-type k nearest, so merging the per-type lists and truncating to k
gives the exact top-k for any combination of types.

Certification composes as well. If no supermarket outside the candidate set can
beat the nearest supermarket, and the same holds for discounters, then neither
can beat the minimum of the two. Per-type candidate sets are in fact *safer*
than one union set: each type gets its own k nearest of that type, so the
nearest supermarket can never be crowded out of the candidate set by a dense
cluster of drugstores.

What this buys is re-runs. Adding, removing or redefining a spec costs nothing;
a store that changes format simply appears in a different type's solve from that
year on; a new shop type costs one solve rather than three new specs.

    from scripts.isochrones import compose
    a = compose.rank1({t: assign_for_type[t] for t in ("Supermarkt", "Discount Geschäft")})
    iso = compose.topk(frames, k=20, max_seconds=1200)

Where the pipeline uses it: run_pipeline.stage_assign merges the per-type
assignments of a year PAIRWISE (rank1([merged_so_far, next_type])) so only one
matrix is in memory at a time -- valid because "nearest of" is associative --
and builds the isochrone frame the same way with topk. verify_compose.py checks
the result against a direct solve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

COLS = ("cell_id", "store_id", "duration_s")


def _stack(frames: Mapping[str, pl.DataFrame] | Sequence[pl.DataFrame]
           ) -> tuple[pl.DataFrame, list[str]]:
    """One frame with the columns the composition needs, types unified.

    Any other column the frames all share is carried through -- `pop` above all,
    which is a property of the cell rather than of the store, identical in every
    per-type frame, and which `nv.catchments` needs downstream. `rank` is not
    carried: it is recomputed for the composed set.

    Ties are broken on `store_id`, so the order the frames arrive in never
    changes the result -- without that, a composed run would not reproduce and
    would not match a direct solve.
    """
    vals = [f for f in (frames.values() if isinstance(frames, Mapping) else frames)
            if f is not None and not f.is_empty()]
    if not vals:
        raise ValueError("compose: nothing to compose -- every frame was empty")
    for f in vals:
        missing = [c for c in COLS if c not in f.columns]
        if missing:
            raise ValueError(f"compose: assignment frame lacks {', '.join(missing)}")
    shared = set(vals[0].columns)
    for f in vals[1:]:
        shared &= set(f.columns)
    extra = [c for c in vals[0].columns if c in shared and c not in (*COLS, "rank")]
    parts = [f.select(pl.col("cell_id"),
                      pl.col("store_id").cast(pl.Utf8),
                      pl.col("duration_s").cast(pl.Float64),
                      *[pl.col(c) for c in extra]) for f in vals]
    return pl.concat(parts, how="vertical_relaxed"), extra


def rank1(frames: Mapping[str, pl.DataFrame] | Sequence[pl.DataFrame]) -> pl.DataFrame:
    """Winner-take-all assignment for the union of the frames' store sets."""
    df, extra = _stack(frames)
    return (df.sort("cell_id", "duration_s", "store_id")
            .unique(subset=["cell_id"], keep="first", maintain_order=True)
            .with_columns(pl.lit(1, pl.Int32).alias("rank"))
            .select("cell_id", "store_id", "rank", "duration_s", *extra))


def topk(frames: Mapping[str, pl.DataFrame] | Sequence[pl.DataFrame], *, k: int,
         max_seconds: float | None = None) -> pl.DataFrame:
    """The k nearest stores of the union, per cell -- the isochrone frame.

    Exact as long as each per-type frame holds at least k of its own type, which
    is what solving per type at rank k guarantees.
    """
    df, extra = _stack(frames)
    if max_seconds is not None:
        df = df.filter(pl.col("duration_s") <= max_seconds)
    return (df.sort("cell_id", "duration_s", "store_id")
            .with_columns((pl.int_range(pl.len()).over("cell_id") + 1)
                          .cast(pl.Int32).alias("rank"))
            .filter(pl.col("rank") <= k)
            .select("cell_id", "store_id", "rank", "duration_s", *extra))


def verify(direct: pl.DataFrame, composed: pl.DataFrame, *, tol: float = 1e-6) -> dict:
    """Compare a composed assignment against one solved directly.

    Run this once on a Bundesland before trusting the composition. It either
    matches cell for cell or the argument above is wrong somewhere -- there is no
    grey zone, so a `store_mismatch` above zero is a finding, not a tolerance to
    widen. (A handful of exact duration ties resolved to different stores of
    equal distance would show as `store_mismatch` with `duration_mismatch` zero:
    still worth reading, but harmless.)
    """
    d = direct.filter(pl.col("rank") == 1).select(
        "cell_id", pl.col("store_id").cast(pl.Utf8).alias("s_direct"),
        pl.col("duration_s").cast(pl.Float64).alias("t_direct"))
    c = composed.filter(pl.col("rank") == 1).select(
        "cell_id", pl.col("store_id").cast(pl.Utf8).alias("s_comp"),
        pl.col("duration_s").cast(pl.Float64).alias("t_comp"))
    j = d.join(c, on="cell_id", how="full", coalesce=True)
    out = {
        "cells_direct": d.height,
        "cells_composed": c.height,
        "only_direct": j.filter(pl.col("s_comp").is_null()).height,
        "only_composed": j.filter(pl.col("s_direct").is_null()).height,
    }
    both = j.filter(pl.col("s_direct").is_not_null() & pl.col("s_comp").is_not_null())
    out["compared"] = both.height
    out["store_mismatch"] = both.filter(pl.col("s_direct") != pl.col("s_comp")).height
    out["duration_mismatch"] = both.filter(
        (pl.col("t_direct") - pl.col("t_comp")).abs() > tol).height
    out["max_abs_dt"] = float(
        both.select((pl.col("t_direct") - pl.col("t_comp")).abs().max()).item() or 0.0)
    out["exact"] = (out["only_direct"] == 0 and out["only_composed"] == 0
                    and out["duration_mismatch"] == 0)
    out["sample"] = both.filter(pl.col("s_direct") != pl.col("s_comp")).head(5)
    return out


def report(v: dict) -> str:
    """One-screen summary of `verify`."""
    head = ("composition is exact" if v["exact"] and not v["store_mismatch"]
            else "composition MATCHES on duration but picks other equal-distance stores"
            if v["exact"] else "composition DIFFERS from the direct solve")
    lines = [head,
             f"  cells: {v['cells_direct']:,} direct, {v['cells_composed']:,} composed, "
             f"{v['compared']:,} compared",
             f"  only in direct: {v['only_direct']:,} | only in composed: "
             f"{v['only_composed']:,}",
             f"  different store: {v['store_mismatch']:,} | different duration: "
             f"{v['duration_mismatch']:,} (max |dt| {v['max_abs_dt']:.3f}s)"]
    if v["store_mismatch"]:
        lines.append(str(v["sample"]))
    return "\n".join(lines)