"""
Slide tables from a run's cube: LaTeX (booktabs) for Beamer, plus the same
tables as Excel-ready CSV.

    python -m scripts.tables runs/baseline.toml
    python -m scripts.tables runs/baseline.toml --actor Chain --year 2024 --top 6
    python -m scripts.tables runs/baseline.toml --share-bases pop,area,vkfl

Writes to <run output>/tables/:

    hhi_national_<spec>_<actor>.tex     national HHI by year, one column per basis
    shares_national_<spec>_<actor>.tex  leading actors' shares, first vs last year
    hhi_laender_<spec>_<actor>.tex      HHI by Bundesland in one year, and its leader
    kreis_bands_<spec>_<actor>.tex      how concentrated local markets are, by year

Each .tex is a bare tabular: \\input it inside a frame, e.g.

    \\begin{frame}{Supermarkets: national concentration}
      \\centering\\scriptsize
      \\input{tables/hhi_national_super_Gruppe.tex}
    \\end{frame}

with \\usepackage{booktabs} in the preamble. The .csv next to it holds the same
numbers, semicolon-separated with decimal commas and a UTF-8 BOM, so German
Excel opens it correctly by double-click.

Conventions: HHI on the 0-10,000 scale, unidentified stores as separate firms
(the cube's `hhi`); shares in percent; "--" where a value does not exist.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import polars as pl

import scripts.run_pipeline as P
from scripts.isochrones.market_cube import wmedian
from scripts.runconfig import LAENDER, SPEC_LABEL, RunConfig

BASIS = {"pop": "Population", "area": "Area", "vkfl": "Sales area",
         "vkfl_food": "Sales area (food)", "umsatz": "Revenue pot.",
         "umsatz_food": "Revenue pot. (food)"}
BANDS = (1000, 2000)          # HHI bands for the Kreis table
_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}"}


def tex(s) -> str:
    return re.sub(r"[\\&%$#_{}~^]", lambda m: _TEX[m.group()], str(s))


def fmt_hhi(v) -> str:
    return "--" if v is None else f"{v:,.0f}"


def fmt_pct(v) -> str:
    return "--" if v is None else f"{100 * v:.1f}"


def write(df: pl.DataFrame, path: Path, fmts: dict, header: list[str] | None = None) -> None:
    """One table as .tex (formatted) and .csv (numbers). The first column is text
    and left-aligned, the rest right-aligned."""
    cols = df.columns
    head = header or cols
    lines = [r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(tex(h) for h in head) + r" \\", r"\midrule"]
    for row in df.iter_rows(named=True):
        cells = [tex(row[cols[0]])] + [fmts.get(c, str)(row[c]) for c in cols[1:]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    df.write_csv(path.with_suffix(".csv"), separator=";", decimal_comma=True,
                 include_bom=True)
    print(f"  {path.name}")


def national_hhi(hhi, spec, al, bases, area=("DE", "DE")) -> pl.DataFrame:
    d = hhi.filter(pl.col("spec") == spec, pl.col("level") == area[0],
                   pl.col("region") == area[1],
                   pl.col("actor_level") == al, pl.col("basis").is_in(bases))
    w = d.pivot(on="basis", index="year", values="hhi").sort("year")
    return w.select("year", *[b for b in bases if b in w.columns])


def national_shares(shares, spec, al, bases, y0, y1, top, area=("DE", "DE")) -> pl.DataFrame:
    d = shares.filter(pl.col("spec") == spec, pl.col("level") == area[0],
                      pl.col("region") == area[1],
                      pl.col("actor_level") == al, pl.col("basis").is_in(bases),
                      pl.col("year").is_in([y0, y1]))
    lead_basis = "pop" if "pop" in bases else bases[0]
    keep = (d.filter(pl.col("rank").is_not_null(), pl.col("basis") == lead_basis)
            .group_by("actor").agg(pl.col("share").max())
            .sort(["share", "actor"], descending=[True, False]).head(top)["actor"].to_list())
    lab = (pl.when(pl.col("rank").is_null()).then(pl.lit("Unidentified"))
           .when(pl.col("actor").is_in(keep)).then(pl.col("actor"))
           .otherwise(pl.lit("Other")))
    g = (d.with_columns(lab.alias("Actor"), (pl.col("basis") + " " + pl.col("year").cast(pl.Utf8))
                        .alias("col"))
         .group_by("Actor", "col").agg(pl.col("share").sum()))
    w = g.pivot(on="col", index="Actor", values="share")
    order = {a: i for i, a in enumerate(keep + ["Other", "Unidentified"])}
    cols = [f"{b} {y}" for b in bases for y in (y0, y1) if f"{b} {y}" in w.columns]
    return (w.with_columns(pl.col("Actor").replace_strict(order, default=99).alias("_o"))
            .sort("_o").select("Actor", *cols))


def laender(hhi, shares, spec, al, bases, year) -> pl.DataFrame:
    d = hhi.filter(pl.col("spec") == spec, pl.col("level") == "BL", pl.col("year") == year,
                   pl.col("actor_level") == al, pl.col("basis").is_in(bases))
    w = d.pivot(on="basis", index="region", values="hhi")
    lb = "pop" if "pop" in bases else bases[0]
    lead = (shares.filter(pl.col("spec") == spec, pl.col("level") == "BL",
                          pl.col("year") == year, pl.col("actor_level") == al,
                          pl.col("basis") == lb, pl.col("rank") == 1)
            .select("region", pl.col("actor").alias("Leader"),
                    pl.col("share").alias("Leader share")))
    return (w.join(lead, on="region", how="left")
            .with_columns(pl.col("region").replace_strict(LAENDER, default=pl.col("region"))
                          .alias("Land"))
            .sort("Land")
            .select("Land", *[b for b in bases if b in w.columns], "Leader", "Leader share"))


def kreis_bands(hhi, spec, al, basis) -> pl.DataFrame:
    k = hhi.filter(pl.col("spec") == spec, pl.col("level") == "KREIS",
                   pl.col("region") != "UNKNOWN", pl.col("actor_level") == al)
    d = k.filter(pl.col("basis") == basis).select("year", "region", "hhi")
    pop = (k.filter(pl.col("basis") == "pop").select("year", "region", pl.col("total").alias("w"))
           if k.filter(pl.col("basis") == "pop").height else None)
    if pop is not None:
        d = d.join(pop, on=["year", "region"], how="left")
    lo, hi = BANDS
    # two readings: every Kreis once (the median Kreis), and weighted by
    # catchment population (the median person's Kreis); the bands likewise
    agg = [pl.col("hhi").median().alias("Median HHI"),
           (wmedian("hhi", "w") if pop is not None else pl.lit(None, pl.Float64))
           .alias("Median HHI, people"),
           (pl.col("hhi") < lo).mean().alias(f"< {lo:,}"),
           pl.col("hhi").is_between(lo, hi).mean().alias(f"{lo:,}-{hi:,}"),
           (pl.col("hhi") > hi).mean().alias(f"> {hi:,}")]
    if pop is not None:
        agg.append(((pl.col("hhi") > hi).cast(pl.Float64) * pl.col("w")).sum()
                   .truediv(pl.col("w").sum()).alias(f"Pop. in > {hi:,}"))
    return d.group_by("year").agg(agg).sort("year")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Slide tables from a run's cube.")
    ap.add_argument("config", nargs="?", default="runs/baseline.toml")
    ap.add_argument("--actor", default="Gruppe", choices=["Gruppe", "Chain"])
    ap.add_argument("--year", type=int, help="year for single-year tables (default: latest)")
    ap.add_argument("--top", type=int, default=8, help="actors in the shares table")
    ap.add_argument("--band-basis", default="pop", help="basis of the Kreis band table")
    ap.add_argument("--share-bases", default="pop,vkfl",
                    help="bases in the shares table; two keep it slide-width")
    a = ap.parse_args(argv)

    rc = RunConfig.load(a.config)
    cfg = rc.pipeline_config()
    shares, hhi = P.read_cube(cfg, list(rc.specs), rc.exclude_actors)
    have = set(hhi["basis"].unique().to_list())
    want = set(rc.report_bases())
    bases = [b for b in BASIS if b in have and b in want]
    years = sorted(hhi["year"].unique().to_list())
    y1 = a.year or years[-1]
    y0 = years[0]
    out = cfg.outdir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    al = a.actor
    area = P.top_area(cfg)[:2]          # Germany, or the Land in a Land run
    print(f"tables -> {out}  (bases: {', '.join(bases)}; {y0} vs {y1})")

    for spec in rc.specs:
        lab = SPEC_LABEL.get(spec, spec)
        h = national_hhi(hhi, spec, al, bases, area)
        write(h, out / f"hhi_national_{spec}_{al}.tex", {b: fmt_hhi for b in bases},
              header=["Year"] + [BASIS[b] for b in h.columns[1:]])

        sb = [b for b in a.share_bases.split(",") if b in bases] or bases[:2]
        s = national_shares(shares, spec, al, sb, y0, y1, a.top, area)
        write(s, out / f"shares_national_{spec}_{al}.tex", {c: fmt_pct for c in s.columns[1:]},
              header=[al] + [f"{BASIS[c.rsplit(' ', 1)[0]]} {c.rsplit(' ', 1)[1]}"
                             for c in s.columns[1:]])

        l = laender(hhi, shares, spec, al, bases, y1)
        write(l, out / f"hhi_laender_{spec}_{al}.tex",
              {**{b: fmt_hhi for b in bases}, "Leader share": fmt_pct},
              header=["Land"] + [BASIS[b] for b in l.columns[1:-2]] + ["Leader", "Share"])

        if a.band_basis in have:
            k = kreis_bands(hhi, spec, al, a.band_basis)
            write(k, out / f"kreis_bands_{spec}_{al}.tex",
                  {"Median HHI": fmt_hhi, "Median HHI, people": fmt_hhi,
                   **{c: fmt_pct for c in k.columns[3:]}},
                  header=["Year", "Median, Kreise", "Median, people"]
                  + [f"{c} (%)" for c in k.columns[3:]])
        print(f"  ({lab})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
