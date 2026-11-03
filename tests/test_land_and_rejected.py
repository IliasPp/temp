"""A Land run reports no Germany total; snap-rejected stores are out on both sides."""
import dataclasses

import polars as pl

import scripts.run_pipeline as P
import scripts.isochrones.market_cube as C
from scripts.runconfig import RunConfig


def _panel():
    rows = [dict(TDLinx=i, year=2024, AGS5=k, Gruppe=g, Chain=g, VKFL=v,
                 **{"Shop Type": "Supermarkt"})
            for i, k, g, v in (("A", "08111", "Edeka", 1000.0), ("B", "08111", "Rewe", 500.0),
                               ("C", "08222", "Schwarz", 800.0), ("D", "09162", "Aldi", 900.0))]
    return pl.DataFrame(rows)


def test_land_run_has_no_germany_total():
    cfg = dataclasses.replace(P.CFG, land="08")
    assert P.top_area(cfg) == ("BL", "08", "Baden-Württemberg")
    assert P.top_area(P.CFG)[:2] == ("DE", "DE")
    assert P.cube_levels(cfg) == ("BL", "KREIS")
    rc = RunConfig(land="08", levels=("DE", "BL", "KREIS"))
    assert P.report_levels(rc, cfg) == ["BL", "KREIS"]
    assert P.report_levels(rc, P.CFG) == ["DE", "BL", "KREIS"]
    assert any("no Germany total" in x for x in rc.validate())

    land = _panel().filter(pl.col("AGS5").str.starts_with("08"))
    s, h = C.cube(None, None, land, spec="super", years=[2024],
                  alive_by_year={2024: ["A", "B", "C"]}, levels=P.cube_levels(cfg))
    assert "DE" not in h["level"].unique().to_list()
    bl = h.filter(pl.col("level") == "BL", pl.col("actor_level") == "Gruppe")
    assert bl["region"].to_list() == ["08"] and bl["total"][0] == 2300.0


def test_rejected_stores_leave_the_supply_side(tmp_path):
    cfg = dataclasses.replace(P.CFG, work=tmp_path)
    assert P.rejected_ids(cfg) == []
    (tmp_path / "snapped").mkdir()
    (tmp_path / "snapped" / "rejected.txt").write_text("B\nD", encoding="utf-8")
    assert P.rejected_ids(cfg) == ["B", "D"]

    # the supply-only branch of run(): membership minus the rejected ids
    rej = pl.Series(P.rejected_ids(cfg), dtype=pl.Utf8)
    mem = P.membership(_panel(), ["Supermarkt"]).filter(~pl.col("store_id").is_in(rej.implode()))
    alive = {2024: mem["store_id"].to_list()}
    s, h = C.cube(None, None, _panel(), spec="super", years=[2024], alive_by_year=alive)
    de = h.filter(pl.col("level") == "DE", pl.col("actor_level") == "Gruppe")
    assert de["total"][0] == 1800.0, "B (500) and D (900) must not carry sales area"
