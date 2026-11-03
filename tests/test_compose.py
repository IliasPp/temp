"""Composition must equal a direct solve, cell for cell."""
import numpy as np, polars as pl
import scripts.isochrones.compose as X



def test_composition_matches_direct_solve():
    rng = np.random.default_rng(4)
    NC, NS = 4000, 60
    TYPES = ["Supermarkt", "Discount Geschäft", "Drogeriemarkt"]
    stype = np.array([TYPES[i % 3] for i in range(NS)])
    cx, cy = rng.uniform(0, 50_000, NC), rng.uniform(0, 50_000, NC)
    sx, sy = rng.uniform(0, 50_000, NS), rng.uniform(0, 50_000, NS)
    # "travel time" with a little noise so it is not a pure Euclidean tie-fest
    D = np.hypot(cx[:, None] - sx[None, :], cy[:, None] - sy[None, :]) / 12.0
    D += rng.uniform(0, 0.5, D.shape)
    SID = np.array([f"S{i:03d}" for i in range(NS)])


    def solve(mask, k):
        """A direct solve over the stores in `mask`: the k nearest, per cell."""
        idx = np.flatnonzero(mask)
        d = D[:, idx]
        order = np.argsort(d, axis=1, kind="stable")[:, :k]
        rows = []
        for r in range(k):
            rows.append(pl.DataFrame({
                "cell_id": np.arange(NC), "store_id": SID[idx][order[:, r]],
                "rank": np.full(NC, r + 1, dtype=np.int32),
                "duration_s": d[np.arange(NC), order[:, r]]}))
        return pl.concat(rows).sort("cell_id", "rank")


    K = 20
    per_type = {t: solve(stype == t, K) for t in TYPES}

    # ---- rank 1 over every combination of types ---------------------------------
    for spec, types in (("super", ["Supermarkt"]),
                        ("super_disc", ["Supermarkt", "Discount Geschäft"]),
                        ("super_disc_drog", TYPES)):
        direct = solve(np.isin(stype, types), 1)
        comp = X.rank1({t: per_type[t].filter(pl.col("rank") == 1) for t in types})
        v = X.verify(direct, comp)
        assert v["exact"] and v["store_mismatch"] == 0, X.report(v)
        print(f"{spec:<16} rank-1 composed == direct ({v['compared']:,} cells)")

    # ---- the isochrone frame, which is the part worth checking ------------------
    for spec, types in (("super_disc", ["Supermarkt", "Discount Geschäft"]),
                        ("super_disc_drog", TYPES)):
        direct = solve(np.isin(stype, types), K)
        comp = X.topk({t: per_type[t] for t in types}, k=K)
        j = (direct.join(comp, on=["cell_id", "rank"], how="inner",
                         suffix="_c"))
        assert j.height == direct.height, (j.height, direct.height)
        assert (j["store_id"] == j["store_id_c"]).all(), "top-k order differs"
        assert float((j["duration_s"] - j["duration_s_c"]).abs().max()) < 1e-9
        print(f"{spec:<16} top-{K} composed == direct ({j.height:,} rows)")

    # a cutoff applies to the composed frame exactly as to a direct one
    cut = 900.0
    direct = solve(np.isin(stype, TYPES), K).filter(pl.col("duration_s") <= cut)
    comp = X.topk(per_type, k=K, max_seconds=cut)
    assert comp.height == direct.height
    print(f"isochrone cutoff {cut:.0f}s: {comp.height:,} rows both ways")

    # ---- ties break deterministically, whatever order the frames arrive in ------
    tie = {"a": pl.DataFrame({"cell_id": [1], "store_id": ["Z9"], "rank": [1],
                              "duration_s": [100.0]}),
           "b": pl.DataFrame({"cell_id": [1], "store_id": ["A1"], "rank": [1],
                              "duration_s": [100.0]})}
    one = X.rank1(tie)["store_id"][0]
    two = X.rank1({"b": tie["b"], "a": tie["a"]})["store_id"][0]
    assert one == two == "A1", (one, two)
    print("exact ties break on store_id, independent of frame order")

    # ---- verify() reports a real difference rather than hiding it --------------
    bad = X.rank1({t: per_type[t].filter(pl.col("rank") == 1) for t in TYPES[:2]})
    v = X.verify(solve(np.isin(stype, TYPES), 1), bad)
    assert not v["exact"] and v["store_mismatch"] > 0
    print("verify() flags a wrong composition:", X.report(v).splitlines()[0])

    print("\nALL CHECKS PASSED")
