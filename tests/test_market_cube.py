"""End-to-end test of market_cube + figures on synthetic data.

No OSRM, no network share, no raster. Builds a 120x120 grid, four Kreise in two
Bundesländer, 40 stores over three years, then checks every invariant that a
published number depends on:

  * catchment population and area are conserved from cells to shares
  * shares sum to 1 in every (region, basis) cell, at all three levels
  * BL and DE totals are exact aggregations of the Kreis level
  * HHI/CR ranges and ordering
  * supply-side bases use the store's own Kreis, demand-side the cell's
  * every figure renders and writes a non-trivial file
"""
import numpy as np
import polars as pl
import geopandas as gpd
from shapely.geometry import box

import scripts.isochrones.market_cube as C
import scripts.visualizations.figures as F



def test_cube_and_figures_end_to_end(tmp_path):
    rng = np.random.default_rng(7)

    # ---------------------------------------------------------------- grid ----
    H = W = 120
    RES, X0, Y0 = 100.0, 4_200_000.0, 3_000_000.0          # EPSG:3035-ish
    rows, cols = np.divmod(np.arange(H * W), W)
    pop = np.clip(rng.gamma(1.4, 6.0, H * W), 0, None).astype(np.float32)
    cells = pl.DataFrame({
        "cell_id": np.arange(H * W, dtype=np.int64),
        "row": rows.astype(np.int32), "col": cols.astype(np.int32),
        "x": X0 + cols * RES + RES / 2, "y": Y0 - rows * RES - RES / 2,
        "pop": pop,
    })
    TOTAL_POP = float(cells["pop"].cast(pl.Float64).sum())

    # 2x2 coarsening in the south half: nodes stand for several cells (the reduction
    # the real adapt_resolution does), north half stays 1:1.
    node = np.where(rows < H // 2, np.arange(H * W),
                    H * W + (rows // 2) * W + (cols // 2))
    mapping = pl.DataFrame({"src": cells["cell_id"], "dst": node.astype(np.int64)})

    # ------------------------------------------------------------- regions ----
    # four Kreise (quadrants) in two Bundesländer: 08xxx west, 09xxx east
    quad = (rows >= H // 2).astype(int) * 2 + (cols >= W // 2).astype(int)
    AGS = ["08111", "09222", "08333", "09444"]
    lookup = pl.DataFrame({"region_code": np.arange(1, 5, dtype=np.int32),
                           "AGS5": AGS, "GEN": [f"Kreis{i}" for i in range(4)]})
    labels = pl.DataFrame({"cell_id": cells["cell_id"],
                           "region_code": (quad + 1).astype(np.int32)})

    polys, ids = [], []
    for i, a in enumerate(AGS):
        r, c = divmod(i, 2)
        polys.append(box(X0 + c * W * RES / 2, Y0 - (r + 1) * H * RES / 2,
                         X0 + (c + 1) * W * RES / 2, Y0 - r * H * RES / 2))
        ids.append(a)
    kreis_gdf = gpd.GeoDataFrame({"AGS5": ids, "GEN": [f"Kreis{i}" for i in range(4)],
                                  "LABEL": [f"Kreis{i}" for i in range(4)]},
                                 geometry=polys, crs="EPSG:3035")
    bl_gdf = kreis_gdf.assign(AGS2=kreis_gdf["AGS5"].str[:2]).dissolve("AGS2").reset_index()
    bl_gdf["LABEL"] = ["West", "East"]

    # -------------------------------------------------------------- stores ----
    N = 40
    GRUPPE = ["Edeka", "Rewe", "Schwarz", "Aldi", "dm"]
    CHAIN = {"Edeka": "Edeka", "Rewe": "Rewe", "Schwarz": "Lidl", "Aldi": "Aldi Süd",
             "dm": "dm"}
    TYPES = ["Supermarkt", "Supermarkt", "Discount Geschäft", "Discount Geschäft",
             "Drogeriemarkt"]
    YEARS = [2023, 2024, 2025]

    sx = rng.uniform(X0, X0 + W * RES, N)
    sy = rng.uniform(Y0 - H * RES, Y0, N)
    s_quad = ((sy < Y0 - H * RES / 2).astype(int) * 2 + (sx >= X0 + W * RES / 2).astype(int))
    rows_panel = []
    for i in range(N):
        g = GRUPPE[i % len(GRUPPE)]
        for y in YEARS:
            if i == 0 and y == 2025:      # exit
                continue
            rows_panel.append({
                "TDLinx": f"TD{i:05d}", "year": y, "AGS5": AGS[s_quad[i]],
                "Gruppe": g, "Chain": CHAIN[g], "Shop Type": TYPES[i % len(TYPES)],
                "VKFL": 600.0 + 30 * i + 10 * (y - 2023),
                "Verkaufsfläche Food": 400.0 + 20 * i,
                "Umsatzpotential Brutto (Mio EUR)": 4.0 + 0.1 * i,
                "Umsatzpotential Brutto Food (Mio EUR)": 3.0 + 0.08 * i,
            })
    panel = pl.DataFrame(rows_panel)
    stores = (panel.group_by("TDLinx").agg(pl.col("year").min().alias("first_year"),
                                           pl.col("year").max().alias("last_year"),
                                           pl.col("Gruppe").last(), pl.col("Shop Type").last())
              .rename({"TDLinx": "store_id"}).sort("store_id"))
    stores = stores.with_columns(
        pl.Series("x", sx[[int(s[2:]) for s in stores["store_id"]]]),
        pl.Series("y", sy[[int(s[2:]) for s in stores["store_id"]]]))

    # ---------------------------------------------- assignment (nearest store) --
    node_xy = (cells.with_columns(pl.Series("node_id", node))
               .group_by("node_id").agg(pl.col("x").mean(), pl.col("y").mean()).sort("node_id"))
    d = np.hypot(node_xy["x"].to_numpy()[:, None] - sx[None, :],
                 node_xy["y"].to_numpy()[:, None] - sy[None, :])


    def assignment_for(year, keep_types):
        ok = np.array([(t in keep_types) and
                       (stores["first_year"][i] <= year <= stores["last_year"][i])
                       for i, t in enumerate(stores["Shop Type"])])
        dd = np.where(ok[None, :], d, np.inf)
        win = dd.argmin(1)
        return pl.DataFrame({
            "cell_id": node_xy["node_id"],
            "store_id": [stores["store_id"][int(j)] for j in win],
            "rank": np.ones(len(win), dtype=np.int32),
            "duration_s": dd[np.arange(len(win)), win].astype(np.float32) / 12.0,
        })


    SPECS = {"super": ["Supermarkt"],
             "super_disc": ["Supermarkt", "Discount Geschäft"],
             "super_disc_drog": ["Supermarkt", "Discount Geschäft", "Drogeriemarkt"]}

    # ================================================================= tests ===
    node_reg = C.node_regions(mapping, cells, labels, lookup, id_col="AGS5")
    assert abs(node_reg["pop"].sum() - TOTAL_POP) / TOTAL_POP < 1e-6, "pop lost in node_regions"
    assert abs(node_reg["area_km2"].sum() - H * W * 0.01) < 1e-6

    a25 = assignment_for(2025, SPECS["super_disc"])
    lost = C.coverage(a25, node_reg)
    assert abs(lost) < 1e-9, lost

    dem = C.demand_atoms(a25, node_reg)
    got = dem.filter(pl.col("basis") == "pop")["value"].sum()
    assert abs(got - TOTAL_POP) / TOTAL_POP < 1e-6, "pop lost in demand_atoms"
    assert abs(dem.filter(pl.col("basis") == "area")["value"].sum() - H * W * 0.01) < 1e-6

    py = panel.filter(pl.col("year") == 2025)
    sup = C.supply_atoms(py, store_ids=stores["store_id"].to_list())
    assert set(sup["basis"].unique()) == {"vkfl", "vkfl_food", "umsatz", "umsatz_food"}
    # supply side keys on the store's own Kreis
    one = sup.filter((pl.col("store_id") == "TD00005") & (pl.col("basis") == "vkfl"))
    assert one["kreis"][0] == AGS[s_quad[5]]

    shares, hhi = C.shares_and_hhi(pl.concat([dem, sup], how="vertical_relaxed"),
                                   C.actor_map(py), year=2025, spec="super_disc")
    C.validate(shares, hhi)

    # level nesting: DE total == sum over Kreis totals, for every basis
    for basis in ("pop", "vkfl"):
        de = hhi.filter((pl.col("level") == "DE") & (pl.col("basis") == basis)
                        & (pl.col("actor_level") == "Gruppe"))["total"][0]
        kr = hhi.filter((pl.col("level") == "KREIS") & (pl.col("basis") == basis)
                        & (pl.col("actor_level") == "Gruppe"))["total"].sum()
        assert abs(de - kr) / de < 1e-9, (basis, de, kr)

    # a Kreis HHI is never below the national one here is NOT guaranteed, but the
    # mechanical bound is: HHI >= 10000/n_actors
    assert (hhi["hhi"] >= C.HHI_SCALE / hhi["n_actors"] - 1e-6).all()
    assert hhi.filter(pl.col("level") == "BL")["region"].unique().sort().to_list() == ["08", "09"]

    # Chain never more concentrated than Gruppe on the same slice
    g = hhi.filter((pl.col("actor_level") == "Gruppe") & (pl.col("level") == "KREIS")
                   & (pl.col("basis") == "pop")).select("region", "hhi").sort("region")
    c = hhi.filter((pl.col("actor_level") == "Chain") & (pl.col("level") == "KREIS")
                   & (pl.col("basis") == "pop")).select("region", pl.col("hhi").alias("h2")).sort("region")
    j = g.join(c, on="region")
    assert (j["hhi"] >= j["h2"] - 1e-6).all(), "Chain HHI above Gruppe HHI"

    # full cube over years and specs
    S, Hh = [], []
    for spec, types in SPECS.items():
        assigns = {y: assignment_for(y, types) for y in YEARS}
        st = stores.filter(pl.col("Shop Type").is_in(types))
        s, h = C.cube(assigns, node_reg, panel, spec=spec, stores=st)
        C.validate(s, h)
        S.append(s)
        Hh.append(h)
    shares, hhi = pl.concat(S), pl.concat(Hh)
    print(f"\ncube: {shares.height:,} share rows x {shares.width} cols, "
          f"{hhi.height:,} hhi rows")
    print(C.headline(hhi).head(9))

    # concentration must fall as the market definition widens (mechanically: more
    # actors, same demand) -- the sanity check that the spec loop did something
    wide = {r["spec"]: r["hhi"] for r in
            hhi.filter((pl.col("level") == "DE") & (pl.col("year") == 2025)
                       & (pl.col("basis") == "pop") & (pl.col("actor_level") == "Gruppe"))
            .iter_rows(named=True)}
    assert wide["super"] > wide["super_disc"] > wide["super_disc_drog"], wide
    print("spec widening lowers HHI:", {k: round(v) for k, v in wide.items()})

    assert C.top_actors(shares, spec="super_disc", year=2025, level="DE", basis="vkfl",
                        actor_level="Gruppe", n=3).height == 3

    # ================================================================ figures ==
    F.set_style()
    out = tmp_path / "fig"
    a = assignment_for(2025, SPECS["super_disc"])
    g = F.grid_slice(cells, mapping, a, stores, bbox=(X0 + 2000, Y0 - 9000,
                                                      X0 + 9000, Y0 - 2000))
    assert g.height > 1000 and g["actor"].null_count() == 0
    ax = F.plot_grid(g, stores=stores, boundaries=kreis_gdf, scale_km=1.0,
                     title="grid assignment, synthetic")
    files = F.save(ax.figure, out / "grid")

    ax = F.plot_choropleth(hhi, kreis_gdf, level="KREIS", spec="super_disc", year=2025,
                           basis="pop", actor_level="Gruppe", title="HHI by Kreis")
    files += F.save(ax.figure, out / "map")
    ax = F.plot_trend(hhi, basis="pop", actor_level="Gruppe", by="spec", title="HHI trend")
    files += F.save(ax.figure, out / "trend")
    ax = F.plot_share_trend(shares, level="DE", region="DE", spec="super_disc",
                            basis="pop", actor_level="Gruppe", top=4, title="shares")
    files += F.save(ax.figure, out / "shares")
    ax = F.plot_distribution(hhi, spec="super_disc", basis="pop", actor_level="Gruppe",
                             title="Kreis HHI distribution")
    files += F.save(ax.figure, out / "dist")
    ax = F.plot_basis_agreement(hhi, spec="super_disc", year=2025, actor_level="Gruppe",
                                title="pop vs vkfl")
    files += F.save(ax.figure, out / "scatter")
    ax = F.plot_share_map(shares, kreis_gdf, level="KREIS", spec="super_disc", year=2025,
                          basis="pop", actor_level="Gruppe", title="leader share")
    files += F.save(ax.figure, out / "share_map")
    ax = F.plot_share_map(shares, kreis_gdf, actor="Edeka", level="KREIS", spec="super_disc",
                          year=2025, basis="pop", actor_level="Gruppe", title="Edeka share")
    files += F.save(ax.figure, out / "share_map_edeka")
    ax = F.plot_leader_map(shares, kreis_gdf, level="KREIS", spec="super_disc", year=2025,
                           basis="pop", actor_level="Gruppe", title="leader")
    files += F.save(ax.figure, out / "leader_map")
    ax = F.plot_share_heatmap(shares, spec="super_disc", basis="pop", actor_level="Gruppe",
                              title="share heatmap")
    files += F.save(ax.figure, out / "heat")
    # catchment polygons: the same solve, vectorised
    polys = F.catchment_polygons(g, crs="EPSG:3035", by="actor")
    assert polys["store_id"].is_unique
    assert abs(polys["area_km2"].sum() - g.filter(pl.col("store_id").is_not_null()).height
               * 0.01) < 1e-6, "polygon area must equal the cell count"
    ax = F.plot_catchments(polys, by="actor", region=kreis_gdf, stores=stores,
                           crs="EPSG:3035", title="catchments")
    files += F.save(ax.figure, out / "catchments")
    ax = F.plot_region_bars(hhi, level="BL", spec="super_disc", basis="pop",
                            actor_level="Gruppe",
                            names=dict(zip(bl_gdf["AGS2"], bl_gdf["LABEL"])), title="by BL")
    files += F.save(ax.figure, out / "bars")
    for f in files:
        assert f.stat().st_size > 3000, f
    print(f"figures: {len(files)} files written")

    print("\nALL CHECKS PASSED")
