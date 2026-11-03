"""
Audit of the two actor identities in the panel: Gruppe (ownership) and Chain (banner).

    python -m scripts.audit_actors runs/baseline.toml

Reads the panel parquet only. Writes summary.md and CSVs to
out/<grid>/diagnostics/actors/. Each section answers one question:

  1  field coverage   Which source fields exist, per year? The Gruppe rules read
                      Banner > Vertriebslinie > Handelszentrale > Name-juristisch
                      > Name; a field missing in some years changes what decides.
  2  how decided      Gruppe_src per year, by stores and VKFL. "supplied" absent
                      everywhere = gate_supplied() never ran.
  3  fallback groups  Gruppen that are a raw Handelszentrale string, not a rule.
  4  missed banners   Banners whose Gruppe did NOT come from the banner: rules
                      that are missing (Spar Express, E center, Real, ...).
  5  ApS             Legal names with "ApS" that are not Netto.
  6  chain nesting    Chains that map to more than one Gruppe in the same year.
  7  spelling         Chains that are the same banner spelled differently.
  8  name fallback    Chains taken from the shop name (no banner): singletons,
                      and generic names merging unrelated shops.
  9  switches         Year-on-year label changes per store: real switches vs
                      spelling flips.
"""
from __future__ import annotations

import argparse
import sys
import polars as pl

from scripts.runconfig import RunConfig

_LINES: list[str] = []
FIELDS = ["Banner", "Vertriebslinie", "Handelszentrale", "Name-juristisch", "Name",
          "Betriebsform zur Handelszentrale"]
REL = "Betriebsform zur Handelszentrale"
LEGAL = r"\b(gmbh|mbh|kg|ohg|ag|eg|e\s?k|e\s?kfm|co|ug|se|aps|gbr|inh|haftungsbeschr\w*)\b"


def say(s: str = "") -> None:
    print(s)
    _LINES.append(s)


def md(df: pl.DataFrame, digits: int = 1, plain=("year", "from", "to")) -> str:
    def f(x, c):
        if isinstance(x, float):
            return "" if x != x else f"{x:,.{digits}f}"
        if isinstance(x, int):
            return str(x) if c in plain else f"{x:,}"
        return "" if x is None else str(x)
    rows = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(f(x, c) for x, c in zip(r, df.columns)) + " |"
                             for r in df.iter_rows()])


def key(col: str) -> pl.Expr:
    """Spelling-insensitive key: case, umlauts, punctuation, legal forms."""
    return (pl.col(col).str.to_lowercase()
            .str.replace_all("ä", "ae").str.replace_all("ö", "oe")
            .str.replace_all("ü", "ue").str.replace_all("ß", "ss")
            .str.replace_all(r"&", " und ").str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(LEGAL, " ").str.replace_all(r"\s+", " ").str.strip_chars())


def wide(df: pl.DataFrame, col: str, val: str) -> pl.DataFrame:
    return df.pivot(on="year", index=col, values=val, sort_columns=True).sort(col)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("config")
    ap.add_argument("--panel", default=None, help="panel parquet (default: Config.panel)")
    ap.add_argument("--shop-types", default="Supermarkt,Discount Geschäft,Drogeriemarkt")
    a = ap.parse_args(argv)
    cfg = RunConfig.load(a.config).pipeline_config()
    out = cfg.out / cfg.tag / "diagnostics" / "actors"
    out.mkdir(parents=True, exist_ok=True)

    p = pl.read_parquet(a.panel or cfg.panel)
    types = [t.strip() for t in a.shop_types.split(",") if t.strip()]
    p = p.filter(pl.col("Shop Type").is_in(types))
    for c in FIELDS + ["Gruppe", "Gruppe_src", "Chain", "VKFL"]:
        if c not in p.columns:
            p = p.with_columns(pl.lit(None, pl.Float64 if c == "VKFL" else pl.Utf8).alias(c))
    p = p.with_columns(pl.col("TDLinx").cast(pl.Utf8), pl.col("VKFL").cast(pl.Float64),
                       pl.col("Gruppe_src").fill_null("(none)"))
    say(f"# Actor audit: {p.height:,} store-years, {p['TDLinx'].n_unique():,} stores, "
        f"{p['year'].min()}-{p['year'].max()}, types {', '.join(types)}\n")

    # 1 -------------------------------------------------------------------------
    say("## 1. Source fields filled, % of store-years")
    cov = (p.group_by("year").agg([(pl.col(c).is_not_null().mean() * 100).alias(c)
                                   for c in FIELDS]).sort("year"))
    say(md(cov))
    rel = (p.filter(pl.col(REL).is_not_null()).group_by("year", REL).len()
           .pipe(lambda d: wide(d, REL, "len")))
    say(f"\n`{REL}` values (store-years); the 2020 gate needs 'Selbständig':\n")
    say(md(rel, 0))

    # 2 -------------------------------------------------------------------------
    say("\n## 2. Which field decided Gruppe (% of VKFL per year)")
    src = (p.group_by("year", "Gruppe_src").agg(pl.col("VKFL").sum().alias("v"))
           .with_columns((pl.col("v") / pl.col("v").sum().over("year") * 100).alias("pct")))
    say(md(wide(src, "Gruppe_src", "pct")))
    n_sup = p.filter(pl.col("Gruppe_src") == "supplied").height
    say("\n" + (f"gate_supplied ran: {n_sup:,} store-years marked 'supplied'." if n_sup else
                "**gate_supplied() has not been applied to this panel** (no 'supplied' rows). "
                "Pre-2020 independents supplied by a group still count as that group."))
    hz_sel = p.filter(pl.col("Gruppe_src") == "Handelszentrale")
    if hz_sel.height:
        g = (hz_sel.group_by("year", pl.col(REL).fill_null("(null)").alias("relation"))
             .agg(pl.col("VKFL").sum()))
        say("\nGruppe from the buying centre only, VKFL by relationship (what the gate acts on):\n")
        say(md(wide(g, "relation", "VKFL"), 0))

    # 3 -------------------------------------------------------------------------
    say("\n## 3. Gruppen that are a raw Handelszentrale string")
    raw = (p.filter(pl.col("Gruppe_src") == "Handelszentrale (raw)")
           .group_by("Gruppe").agg(pl.len().alias("store-years"), pl.col("TDLinx").n_unique().alias("stores"),
                                   pl.col("VKFL").sum().alias("VKFL"),
                                   pl.col("Banner").drop_nulls().mode().first().alias("typical banner"),
                                   pl.col("year").min().alias("from"), pl.col("year").max().alias("to"))
           .sort("VKFL", descending=True))
    raw.write_csv(out / "gruppe_raw_fallback.csv")
    tot = p["VKFL"].sum()
    say(f"{raw.height:,} labels, {raw['VKFL'].sum() / tot:.1%} of all VKFL. Each is treated as one "
        f"firm. Largest:\n")
    say(md(raw.head(20), 0))

    # 4 -------------------------------------------------------------------------
    say("\n## 4. Banners whose Gruppe did not come from the banner")
    mb = (p.filter(pl.col("Banner").is_not_null() & (pl.col("Gruppe_src") != "Banner"))
          .group_by("Banner").agg(pl.len().alias("store-years"), pl.col("VKFL").sum().alias("VKFL"),
                                  pl.col("Gruppe").drop_nulls().unique().sort().str.join(" / ").alias("Gruppen assigned"),
                                  pl.col("Gruppe_src").unique().sort().str.join(" / ").alias("decided by"),
                                  pl.col("Gruppe").is_null().mean().mul(100).alias("% no Gruppe"))
          .sort("VKFL", descending=True))
    mb.write_csv(out / "banners_without_rule.csv")
    say("A banner here has no rule of its own; its Gruppe depends on weaker fields and can "
        "differ store by store. Largest by VKFL:\n")
    say(md(mb.head(25), 0))

    # 5 -------------------------------------------------------------------------
    say("\n## 5. 'ApS' legal names that are not Netto")
    aps = (p.filter(pl.col("Name-juristisch").str.contains(r"(?i)\baps\b")
                    & ~pl.col("Name-juristisch").str.contains(r"(?i)netto")
                    & ~pl.col("Banner").fill_null("").str.contains(r"(?i)netto"))
           .group_by("Name-juristisch", "Banner", "Gruppe").len().sort("len", descending=True))
    say(md(aps.head(15), 0) if aps.height else "None.")

    # 6 -------------------------------------------------------------------------
    say("\n## 6. Chains mapped to more than one Gruppe in the same year")
    nest = (p.filter(pl.col("Chain").is_not_null())
            .group_by("year", "Chain", pl.col("Gruppe").fill_null("(none)"))
            .agg(pl.len().alias("n"), pl.col("VKFL").sum().alias("v")))
    multi = (nest.group_by("year", "Chain").agg(pl.len().alias("k"), pl.col("v").sum().alias("v"),
             pl.concat_str([pl.col("Gruppe"), pl.lit(" ("), pl.col("n").cast(pl.Utf8), pl.lit(")")])
             .sort_by("n", descending=True).str.join(", ").alias("Gruppen (stores)"))
             .filter(pl.col("k") > 1))
    last = int(p["year"].max())
    say(f"{multi['Chain'].n_unique():,} chains, {multi.filter(pl.col('year') == last)['v'].sum() / p.filter(pl.col('year') == last)['VKFL'].sum():.1%} "
        f"of {last} VKFL. A chain is a firm's banner, so it should sit inside one Gruppe. {last}, largest:\n")
    say(md(multi.filter(pl.col("year") == last).sort("v", descending=True)
           .select("Chain", pl.col("v").alias("VKFL"), "Gruppen (stores)").head(20), 0))
    multi.write_csv(out / "chains_multi_gruppe.csv")

    # 7 -------------------------------------------------------------------------
    say("\n## 7. One banner, several spellings")
    sp = (p.filter(pl.col("Banner").is_not_null()).with_columns(key("Banner").alias("k"))
          .group_by(pl.col("k").alias("key")).agg(pl.col("Banner").unique().sort().alias("spellings"),
                             pl.len().alias("store-years"), pl.col("VKFL").sum().alias("VKFL"))
          .filter(pl.col("spellings").list.len() > 1).sort("VKFL", descending=True)
          .with_columns(pl.col("spellings").list.join(" / ")))
    sp.write_csv(out / "banner_spellings.csv")
    say(f"{sp.height:,} banners appear under more than one spelling ({sp['VKFL'].sum() / tot:.1%} "
        f"of VKFL); each spelling is a separate Chain today. Largest:\n")
    say(md(sp.head(20), 0))

    # 8 -------------------------------------------------------------------------
    say("\n## 8. Chains taken from the shop name (no banner)")
    nf = p.filter(pl.col("Banner").is_null() & pl.col("Chain").is_not_null())
    say(f"{nf.height / p.height:.1%} of store-years, {nf['VKFL'].sum() / tot:.1%} of VKFL. "
        f"By year, % of VKFL:\n")
    say(md(p.group_by("year").agg((pl.col("VKFL").filter(pl.col("Banner").is_null()
                                                         & pl.col("Chain").is_not_null()).sum()
                                   / pl.col("VKFL").sum() * 100).alias("% VKFL name-based"))
           .sort("year")))
    gen = (nf.group_by("Chain").agg(pl.col("TDLinx").n_unique().alias("stores"),
                                    pl.col("Name-juristisch").n_unique().alias("legal entities"),
                                    pl.col("Kreis").n_unique().alias("Kreise") if "Kreis" in nf.columns
                                    else pl.lit(None).alias("Kreise"),
                                    pl.col("Gruppe").drop_nulls().unique().sort().str.join(" / ").alias("Gruppen"))
           .filter(pl.col("stores") > 1).sort("legal entities", descending=True))
    gen.write_csv(out / "name_chains.csv")
    say("\nShop names used by several stores with different owners (unrelated shops merged into "
        "one 'chain'):\n")
    say(md(gen.filter(pl.col("legal entities") > 1).head(20), 0))

    # 9 -------------------------------------------------------------------------
    say("\n## 9. Year-on-year label changes of the same store")
    s = (p.select("TDLinx", "year", "Gruppe", "Chain", key("Chain").alias("ck"))
         .sort("TDLinx", "year"))
    nxt = s.rename({c: c + "_n" for c in ("Gruppe", "Chain", "ck")}).with_columns(pl.col("year") - 1)
    j = s.join(nxt, on=["TDLinx", "year"], how="inner")
    sw = j.group_by("year").agg(
        pl.len().alias("stores in both years"),
        (pl.col("Gruppe").fill_null("") != pl.col("Gruppe_n").fill_null("")).sum().alias("Gruppe changed"),
        ((pl.col("Chain") != pl.col("Chain_n")) & (pl.col("ck") != pl.col("ck_n"))).sum().alias("Chain changed"),
        ((pl.col("Chain") != pl.col("Chain_n")) & (pl.col("ck") == pl.col("ck_n"))).sum()
        .alias("Chain spelling only")).sort("year")
    say("A spike in 'Gruppe changed' with no matching 'Chain changed' is a coding change, "
        "not an ownership change:\n")
    say(md(sw.with_columns((pl.col("year").cast(pl.Utf8) + "→" + (pl.col("year") + 1).cast(pl.Utf8))
                           .alias("year")), 0))

    (out / "summary.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
