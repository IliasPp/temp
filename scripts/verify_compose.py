"""
Check the composition against a direct solve.

    python -m scripts.verify_compose --spec super_disc --land 01 --year 2025

Solves the spec's store set directly -- one extra matrix, on one Bundesland so it
costs minutes rather than hours -- and compares it cell for cell with the
assignment composed from the per-type matrices. The composition is exact by
construction (see scripts.isochrones.compose), so this either matches or the
reasoning is wrong somewhere; there is no tolerance to widen.

Run it once after switching to per-type solves, and again whenever
`network_voronoi_2`'s assignment logic changes. The temporary matrix is left in
place under `solve*/verify_<spec>/` so a re-run is cheap; delete it when done.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import polars as pl

import scripts.run_pipeline as P
import scripts.isochrones.compose as CO
import scripts.isochrones.network_voronoi_2 as nv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--spec", default="super_disc", choices=list(P.SPECS))
    ap.add_argument("--land", default="01", help="AGS2 prefix; '' for all of Germany")
    ap.add_argument("--year", type=int, default=2025)
    a = ap.parse_args(argv)

    cfg = dataclasses.replace(P.CFG, land=(a.land or None), years=(a.year,))
    types = P.SPECS[a.spec]

    panel = P.load_panel(cfg, shop_types=None)
    kreis_gdf, _ = P.region_layers(cfg, panel)
    raster = Path(cfg.ghsl)
    stores = P.stage_stores(panel, cfg)                 # national, like the grid
    panel = panel.filter(pl.col("Shop Type").is_in(P.UNION))
    cells, c2, mapping = P.stage_cells(stores, cfg, raster=raster)
    node_reg = P.stage_regions(cells, mapping, kreis_gdf, cfg, raster=raster)
    if cfg.land:
        panel = panel.filter(P.C.ags5("AGS5").str.starts_with(cfg.land))
        stores = stores.filter(pl.col("store_id").is_in(panel["TDLinx"].cast(pl.Utf8)))
        cells, c2, mapping, node_reg = P.land_view(cells, c2, mapping, node_reg, cfg)

    # composed: one matrix per type
    solved = {t: P.stage_matrix(t, c2, stores, P.membership(panel, [t])["store_id"], cfg)
              for t in types}
    composed, _ = P.stage_assign(a.spec, c2, solved, cfg, panel=panel, types=types)

    # direct: one matrix on the spec's whole store set
    mem = P.membership(panel, types)
    st_all, h = P.stage_matrix(f"verify_{a.spec}", c2, stores,
                               mem["store_id"].unique(), cfg)
    cand, dur = h.get()
    ids = mem.filter(pl.col("year") == a.year)["store_id"].to_list()
    direct = nv.assign(c2, P.stores_for_year(st_all, ids, a.year), cand, dur,
                       year=a.year, ranks=1)

    v = CO.verify(direct, composed[a.year])
    print()
    print(CO.report(v))
    return 0 if v["exact"] and not v["store_mismatch"] else 1


if __name__ == "__main__":
    sys.exit(main())
