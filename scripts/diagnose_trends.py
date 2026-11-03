"""
Where do the national trends come from -- market events or panel artefacts?

    python -u -m scripts.diagnose_trends runs/baseline.toml > trends.txt

Reads the cube and the panel only; no OSRM, no matrices. Three blocks:

1. National Gruppe HHI per year with its decomposition: the named groups'
   shares, the unidentified share, and the gap between `hhi` (unidentified
   stores as separate firms) and `hhi_pooled`. A dip that coincides with a jump
   in the unidentified share is a labelling effect, not competition.
2. Year-to-year contributions to the HHI change, per group: which group's
   share moved the index, and in which direction.
3. Label switches among stores present in consecutive years, weighted by
   VKFL: real ownership events (Kaiser's -> Edeka, Real -> Kaufland/Globus)
   show as a few large, dated from -> to pairs; a panel relabelling shows as
   many stores moving from a residual or unnamed label into a named one.
"""

from __future__ import annotations

import sys

import polars as pl

import scripts.run_pipeline as P
from scripts.runconfig import RunConfig

# Windows writes redirected output as cp1252, which cannot encode polars' box
# characters; UTF-8 output and plain Markdown tables work in any file and paste
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)


def hhi_block(hhi, shares, spec, basis, al="Gruppe", top=6, area=("DE", "DE")):
    h = (hhi.filter(pl.col("spec") == spec, pl.col("level") == area[0],
                    pl.col("region") == area[1],
                    pl.col("basis") == basis, pl.col("actor_level") == al)
         .select("year", "hhi", "hhi_pooled", "fringe_share", "n_actors", "n_stores")
         .sort("year"))
    s = shares.filter(pl.col("spec") == spec, pl.col("level") == area[0],
                      pl.col("region") == area[1],
                      pl.col("basis") == basis, pl.col("actor_level") == al)
    lead = (s.filter(pl.col("rank").is_not_null()).group_by("actor")
            .agg(pl.col("share").max()).sort("share", descending=True)
            .head(top)["actor"].to_list())
    wide = (s.filter(pl.col("actor").is_in(lead))
            .pivot(on="actor", index="year", values="share").sort("year"))
    print(f"\n=== {spec} | {basis} | {al}: {area[1]} HHI and its parts ===")
    print(h.join(wide, on="year", how="left")
           .with_columns(pl.col("hhi", "hhi_pooled").round(0),
                         pl.exclude("year", "hhi", "hhi_pooled", "n_actors", "n_stores")
                         .round(4)))

    # contribution of each group to the year-on-year HHI change: d(100 s)^2
    c = (s.filter(pl.col("rank").is_not_null())
         .select("year", "actor", (pl.col("share") * 100).pow(2).alias("sq"))
         .sort("actor", "year")
         .with_columns((pl.col("sq") - pl.col("sq").shift(1).over("actor"))
                       .alias("d_hhi"))
         .drop_nulls("d_hhi"))
    big = (c.group_by("actor").agg(pl.col("d_hhi").abs().max())
           .sort("d_hhi", descending=True).head(top)["actor"].to_list())
    print("\n--- contribution to the year-on-year change in HHI (points) ---")
    print(c.filter(pl.col("actor").is_in(big))
           .pivot(on="actor", index="year", values="d_hhi").sort("year")
           .with_columns(pl.exclude("year").round(0)))


def switches(panel, col, top=12):
    """VKFL-weighted label changes among stores present in consecutive years."""
    p = (panel.select(pl.col("TDLinx").cast(pl.Utf8).alias("id"), "year",
                      pl.col(col).cast(pl.Utf8).fill_null("<null>").alias("lab"),
                      pl.col("VKFL").cast(pl.Float64).fill_null(0.0).alias("vkfl"),
                      "Shop Type")
         .unique(["id", "year"], keep="first"))
    nxt = p.select("id", (pl.col("year") - 1).alias("year"),
                   pl.col("lab").alias("lab_next"), pl.col("Shop Type").alias("type_next"))
    j = p.join(nxt, on=["id", "year"], how="inner")
    sw = j.filter(pl.col("lab") != pl.col("lab_next"))
    print(f"\n=== {col}: label switches between consecutive years ===")
    print(sw.group_by("year").agg(pl.len().alias("stores"),
                                  pl.col("vkfl").sum().round(0).alias("vkfl_m2"))
            .join(j.group_by("year").agg(pl.col("vkfl").sum().alias("tot")), on="year")
            .with_columns((pl.col("vkfl_m2") / pl.col("tot")).round(4).alias("share_of_vkfl"))
            .drop("tot").sort("year")
            .with_columns((pl.col("year").cast(pl.Utf8) + "->" +
                           (pl.col("year") + 1).cast(pl.Utf8)).alias("years")))
    print(f"\n--- largest {col} moves (VKFL-weighted) ---")
    print(sw.group_by("year", "lab", "lab_next")
            .agg(pl.len().alias("stores"), pl.col("vkfl").sum().round(0).alias("vkfl_m2"))
            .sort("vkfl_m2", descending=True).head(top * 2)
            .sort("year", "vkfl_m2", descending=[False, True]))


def coverage(panel):
    """How well each year is labelled: the share of VKFL with no Gruppe / Chain."""
    print("\n=== labelling coverage per year (share of VKFL, Supermarkt) ===")
    q = panel.filter(pl.col("Shop Type") == "Supermarkt")
    cols = [c for c in ("Gruppe", "Chain", "Banner") if c in q.columns]
    tot = pl.col("VKFL").cast(pl.Float64).fill_null(0.0)
    print(q.group_by("year").agg(
        pl.len().alias("stores"),
        *[(tot.filter(pl.col(c).is_null()).sum() / tot.sum()).round(4).alias(f"{c}_missing")
          for c in cols],
        *[pl.col(c).n_unique().alias(f"n_{c}") for c in cols])
        .sort("year"))


def main(argv=None) -> int:
    path = (argv or sys.argv[1:] or ["runs/baseline.toml"])[0]
    rc = RunConfig.load(path)
    cfg = rc.pipeline_config()
    shares, hhi = P.read_cube(cfg, ["super"])
    for basis in ("pop", "area", "vkfl"):
        if hhi.filter(pl.col("basis") == basis).height:
            hhi_block(hhi, shares, "super", basis, area=P.top_area(cfg)[:2])
    panel = P.load_panel(cfg, shop_types=["Supermarkt", "Discount Geschäft", "Drogeriemarkt"])
    coverage(panel)
    sup = panel.filter(pl.col("Shop Type") == "Supermarkt")
    for col in ("Gruppe", "Chain"):
        if col in sup.columns:
            switches(sup, col)
    return 0


if __name__ == "__main__":
    sys.exit(main())
