"""Format and banner switchers: membership and labels must follow the year.

A store that trades as a Supermarkt until 2019 and as a Discount Geschäft after
belongs to the supermarket market only until 2019 -- its catchment AND its sales
area must leave in the same year. A store whose Gruppe changes must be counted
under the operator of that year.
"""
import numpy as np, polars as pl

import scripts.run_pipeline as P
import scripts.isochrones.market_cube as C



def test_membership_and_labels_follow_the_year(tmp_path):
    YEARS = [2018, 2019, 2020, 2021]
    rows = []
    for y in YEARS:
        # A: always a supermarket, Gruppe changes in 2020 (banner switch)
        rows.append(dict(TDLinx="A", year=y, AGS5="08111",
                         Gruppe="Rewe" if y < 2020 else "Edeka",
                         Chain="Rewe" if y < 2020 else "Edeka",
                         **{"Shop Type": "Supermarkt"}, VKFL=1000.0))
        # B: supermarket until 2019, discounter from 2020 (format switch)
        rows.append(dict(TDLinx="B", year=y, AGS5="08111", Gruppe="Schwarz", Chain="Lidl",
                         **{"Shop Type": "Supermarkt" if y < 2020 else "Discount Geschäft"},
                         VKFL=800.0))
        # C: always a discounter
        rows.append(dict(TDLinx="C", year=y, AGS5="08111", Gruppe="Aldi", Chain="Aldi Süd",
                         **{"Shop Type": "Discount Geschäft"}, VKFL=700.0))
    panel = pl.DataFrame(rows)

    # what prepare_stores would collapse to (most recent value)
    def spec_stores(frame, types=("Supermarkt",)):
        """Stores that ever traded in the spec, in frame order (what a matrix indexes)."""
        return P.matrix_frame(frame, P.membership(panel, types)["store_id"].unique())[0]


    stores = pl.DataFrame({"store_id": ["A", "B", "C"],
                           "first_year": [2018, 2018, 2018], "last_year": [2021, 2021, 2021],
                           "Gruppe": ["Edeka", "Schwarz", "Aldi"],
                           "Shop Type": ["Supermarkt", "Discount Geschäft", "Discount Geschäft"]})

    # ---- membership follows the panel, per year ---------------------------------
    mem = P.membership(panel, ["Supermarkt"])
    by_year = {y: set(mem.filter(pl.col("year") == y)["store_id"]) for y in YEARS}
    assert by_year == {2018: {"A", "B"}, 2019: {"A", "B"}, 2020: {"A"}, 2021: {"A"}}, by_year
    print("membership by year:", {y: sorted(v) for y, v in by_year.items()})

    # the panel is the only source of membership; the frame's collapsed Shop Type
    # would have said B was never a supermarket
    assert sorted(spec_stores(stores)["store_id"]) == ["A", "B"]
    print("matrix frame: union over years, not the last format on record")

    # ---- the alive window is rewritten per year, order preserved ----------------
    sy = P.stores_for_year(spec_stores(stores), by_year[2020], 2020)
    assert sy["store_id"].to_list() == ["A", "B"], "row order must not move"
    assert sy.filter(pl.col("store_id") == "B")["first_year"][0] == -1, "B is out in 2020"
    assert sy.filter(pl.col("store_id") == "A")["first_year"][0] == 2020
    print("stores_for_year: B dead in 2020, A alive, order intact")

    # ---- the cube: supply side leaves in the same year the catchment does -------
    node_reg = pl.DataFrame({"node_id": [1, 2], "kreis": ["08111", "08111"],
                             "pop": [100.0, 100.0], "n_cells": [10, 10],
                             "area_km2": [0.1, 0.1]})
    def assign(winners):          # nodes split evenly between the given stores
        return pl.DataFrame({"cell_id": [1, 2], "store_id": winners,
                             "rank": np.ones(2, dtype=np.int32)})
    assigns = {2019: assign(["A", "B"]), 2020: assign(["A", "A"])}
    alive = {2019: ["A", "B"], 2020: ["A"]}

    sh, hh = C.cube(assigns, node_reg, panel, spec="super",
                    stores=spec_stores(stores), alive_by_year=alive)
    v = {(r["year"], r["basis"], r["actor"]): r["value"] for r in
         sh.filter(pl.col("level") == "DE", pl.col("actor_level") == "Gruppe").iter_rows(named=True)}
    assert v[(2019, "vkfl", "Schwarz")] == 800.0, "B counts in 2019"
    assert (2020, "vkfl", "Schwarz") not in v, "B's sales area must leave in 2020"
    assert v[(2020, "vkfl", "Edeka")] == 1000.0
    print("cube: supply side follows membership, not the collapsed type")

    # ---- banner switch: A is Rewe in 2019 and Edeka in 2020 ---------------------
    assert v[(2019, "pop", "Rewe")] == 100.0 and (2019, "pop", "Edeka") not in v
    assert v[(2020, "pop", "Edeka")] == 200.0 and (2020, "pop", "Rewe") not in v
    print("cube: Gruppe is the operator of that year")

    # ---- grid maps label with the year's Gruppe ---------------------------------
    lab = P.year_labels(panel, 2019, "Gruppe")
    assert dict(zip(lab["store_id"], lab["Gruppe"]))["A"] == "Rewe"
    lab = P.year_labels(panel, 2021, "Gruppe")
    assert dict(zip(lab["store_id"], lab["Gruppe"]))["A"] == "Edeka"
    print("year_labels: maps show the operator of the mapped year")


    # ---- Chain comes from the panel's own column, like Gruppe -------------------
    mix = pl.DataFrame({"TDLinx": ["A", "B", "C"], "year": [2020] * 3,
                        "Gruppe": ["Edeka", "Edeka", None],
                        "Chain": ["Edeka", "Müller Markt", None]})
    am = C.actor_map(mix).sort("store_id")
    assert am["Chain"].to_list() == ["Edeka", "Müller Markt", "UNKNOWN"], am
    assert am["Gruppe"].to_list() == ["Edeka", "Edeka", "UNKNOWN"]
    try:
        C.actor_map(mix.drop("Chain"))
        raise AssertionError("a missing Chain column must not pass silently")
    except ValueError as e:
        assert "Chain" in str(e)
    print("Chain: the panel column, UNKNOWN where null")

    # ---- a supply-side report needs no solve ------------------------------------
    from scripts.runconfig import RunConfig                

    supply = RunConfig(stages=("cube", "figures"), bases=("vkfl", "umsatz"),
                       figures=("map_hhi", "trend_hhi"), actor_levels=("Gruppe",))
    assert not supply.needs_solve()
    assert P.needed_stages(supply) == {"cube", "figures"}, P.needed_stages(supply)

    demand = RunConfig(stages=("cube", "figures"), bases=("vkfl", "pop"))
    assert demand.needs_solve()
    assert "matrix" in P.needed_stages(demand) and "cells" in P.needed_stages(demand)

    grid = RunConfig(stages=("cube",), bases=("vkfl",), figures=("grid_maps",))
    assert grid.needs_solve(), "a grid map draws the assignment itself"

    explicit = RunConfig(stages=("matrix",), bases=("vkfl",), figures=())
    assert "matrix" in P.needed_stages(explicit), "selecting a solve stage still runs it"
    print("needed_stages: solve only when the report needs it")

    # the cube builds from the panel alone
    sh2, hh2 = C.cube(None, None, panel, spec="super", years=YEARS,
                      alive_by_year={y: sorted(by_year[y]) for y in YEARS})
    assert set(sh2["basis"].unique()) == {"vkfl"}, sh2["basis"].unique().to_list()
    de = hh2.filter(pl.col("level") == "DE", pl.col("year") == 2019,
                    pl.col("actor_level") == "Gruppe")
    assert de["n_stores"][0] == 2 and de["total"][0] == 1800.0, de
    de20 = hh2.filter(pl.col("level") == "DE", pl.col("year") == 2020,
                      pl.col("actor_level") == "Gruppe")
    assert de20["n_stores"][0] == 1 and de20["hhi"][0] == 10_000.0, de20
    C.validate(sh2, hh2)
    print("supply-only cube: built from the panel, membership still per year")


    # ---- a panel-only run must not open the raster ------------------------------
    import builtins                                         
    opened = []
    _real_open = builtins.open
    def _watch(f, *a, **k):
        if str(f).endswith((".tif", ".tiff")):
            opened.append(str(f))
        return _real_open(f, *a, **k)
    builtins.open = _watch
    try:
        rc = RunConfig(stages=("cube",), bases=("vkfl", "umsatz"), figures=())
        assert not rc.needs_solve()
        assert P.needed_stages(rc) == {"cube"}
    finally:
        builtins.open = _real_open
    assert not opened, opened
    print("panel-only selection: raster never referenced")

    # the CRS fallback is GHS-POP's native Mollweide, not a guess at EPSG:3035
    assert P.CFG.raster_crs == "ESRI:54009", P.CFG.raster_crs
    print("raster_crs default:", P.CFG.raster_crs)


    # ---- a changed store frame invalidates what was built from it ---------------

    tmp = tmp_path
    s1 = pl.DataFrame({"store_id": ["A", "B", "C"]})
    s2 = pl.DataFrame({"store_id": ["A", "B", "C", "D"]})   # one switcher joins the spec
    s3 = pl.DataFrame({"store_id": ["A", "C", "B"]})        # same set, different order
    f = tmp / "fp.txt"
    fp1 = P.store_fingerprint(s1)
    assert not P._fresh(f, fp1), "no sidecar yet: never fresh"
    P._commit(f, fp1)
    assert P._fresh(f, fp1), "unchanged frame is fresh"
    assert not P._fresh(f, P.store_fingerprint(s2)), "an added store must invalidate"
    assert not P._fresh(f, P.store_fingerprint(s3)), "a reordered frame must invalidate"
    assert P.store_fingerprint(s1) != P.store_fingerprint(s1, "excl=Sonstige")
    print("fingerprint: set, order and exclusions all invalidate")


    # ---- the store frame's own columns are never the membership source ---------
    frame = stores.with_columns(pl.lit("Discount Geschäft").alias("Shop Type"),
                                pl.lit("Lidl").alias("Chain"))
    # every collapsed label says "discounter", yet A and B were supermarkets until 2019
    got = sorted(spec_stores(frame)["store_id"])
    assert got == ["A", "B"], got
    print("spec_stores ignores the frame's collapsed Shop Type/Chain")

    print("\nALL CHECKS PASSED")
