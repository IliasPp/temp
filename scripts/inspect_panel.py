"""inspect_panel.py — sanity checks for the TDLinx store-year panel.

Usage:  python inspect_panel.py panel.parquet [outdir]
Prints one headline line per check and writes the offending rows / tables
to outdir (default: panel_checks/) as CSV.  In a notebook, import the
functions and look at the frames directly.
"""
import sys
from pathlib import Path
import polars as pl

K = ["TDLinx", "year"]
U, UF, UN = (f"Umsatzpotential Brutto{s} (Mio EUR)" for s in ("", " Food", " Nonfood"))
V, VF, VN = "VKFL", "Verkaufsfläche Food", "Verkaufsfläche Nonfood"
NUM = [U, UF, UN, V, VF, VN, "Parkplätze", "Kassen"]
ATTR = ["Gruppe", "Chain", "Banner", "Shop Type", "Handelszentrale"]
LAND = {"01": "Schleswig-Holstein", "02": "Hamburg", "03": "Niedersachsen", "04": "Bremen",
        "05": "Nordrhein-Westfalen", "06": "Hessen", "07": "Rheinland-Pfalz",
        "08": "Baden-Württemberg", "09": "Bayern", "10": "Saarland", "11": "Berlin",
        "12": "Brandenburg", "13": "Mecklenburg-Vorpommern", "14": "Sachsen",
        "15": "Sachsen-Anhalt", "16": "Thüringen"}
BBOX = (5.8, 15.1, 47.2, 55.1)  # lon_min, lon_max, lat_min, lat_max


def load(path) -> pl.DataFrame:
    p = pl.read_parquet(path)
    return p.with_columns(
        pl.col("AGS5").cast(pl.Utf8).str.zfill(5),
        pl.col("Gemeindekennziffer").cast(pl.Utf8).str.zfill(8),
    ).sort(K)


def keys(p):
    """(TDLinx, year) must be unique."""
    return p.filter(pl.len().over(K) > 1)


def coverage(p):
    """Store count per year and Shop Type — look for level breaks between years."""
    return (p.group_by("year", "Shop Type").len()
             .pivot(on="Shop Type", index="year", values="len").sort("year"))


def missing(p):
    """Null share per column per year (columns with any nulls only)."""
    m = p.group_by("year").agg(pl.all().exclude("year").null_count() / pl.len()).sort("year")
    keep = [c for c in m.columns[1:] if m[c].sum() > 0]
    return m.select("year", *keep)


def numerics(p):
    """Per year: quantiles, zeros and negatives for every numeric field."""
    out = []
    for c in NUM:
        x = pl.col(c)
        out.append(p.group_by("year").agg(
            pl.lit(c).alias("var"), x.null_count().alias("null"),
            (x == 0).sum().alias("zero"), (x < 0).sum().alias("neg"),
            *[x.quantile(q).alias(f"p{int(q*100)}") for q in (0.01, 0.5, 0.99)],
            x.max().alias("max")))
    return pl.concat(out).sort("var", "year")


def components(p, tol=0.01):
    """Food + Nonfood must add up to the total (relative tolerance)."""
    return p.filter(
        ((pl.col(UF) + pl.col(UN) - pl.col(U)).abs() > tol * pl.col(U))
        | ((pl.col(VF) + pl.col(VN) - pl.col(V)).abs() > tol * pl.col(V))
    ).select(*K, "Shop Type", U, UF, UN, V, VF, VN)


def productivity(p, z=4.0):
    """Revenue per m² (EUR); flags stores > z robust SDs from their Shop Type x year median."""
    r = (pl.col(U) * 1e6 / pl.col(V)).alias("eur_m2")
    lr = r.log()
    med = lr.median().over("Shop Type", "year")
    mad = (lr - med).abs().median().over("Shop Type", "year") * 1.4826
    return (p.filter(pl.col(V) > 0, pl.col(U) > 0)
             .with_columns(r, ((lr - med) / mad).alias("z"))
             .filter(pl.col("z").is_finite(), pl.col("z").abs() > z)
             .select(*K, "Name", "Shop Type", "Gruppe", U, V, "eur_m2", "z"))


def _hav(lat1, lon1, lat2, lon2):
    rad = 3.141592653589793 / 180
    a = (((lat2 - lat1) * rad / 2).sin() ** 2
         + (lat1 * rad).cos() * (lat2 * rad).cos() * ((lon2 - lon1) * rad / 2).sin() ** 2)
    return 2 * 6_371_000 * a.sqrt().arcsin()


def coords(p, jump_m=500):
    """Missing/out-of-Germany coordinates, shared coordinates, and moves across years."""
    lo, hi, la, ha = BBOX
    bad = p.filter(pl.col("lat").is_null() | pl.col("lon").is_null()
                   | ~pl.col("lon").is_between(lo, hi) | ~pl.col("lat").is_between(la, ha))
    shared = (p.filter(pl.len().over("year", "lat", "lon") > 1)
               .select(*K, "lat", "lon", "Name", "Shop Type", "Gruppe")
               .sort("year", "lat", "lon"))
    moves = (p.with_columns(
                 _hav(pl.col("lat").shift().over("TDLinx"), pl.col("lon").shift().over("TDLinx"),
                      pl.col("lat"), pl.col("lon")).alias("moved_m"))
              .filter(pl.col("moved_m") > jump_m)
              .select(*K, "moved_m", "full_address"))
    return bad.select(*K, "lat", "lon", "full_address"), shared, moves


def admin(p):
    """AGS5 must map to one Kreis, its Land prefix and the Gemeinde key."""
    land = pl.col("AGS5").str.slice(0, 2).replace_strict(LAND, default=None)
    rows = p.filter(
        pl.col("AGS5").is_null()
        | (land != pl.col("Bundesland"))
        | (pl.col("Gemeindekennziffer").str.slice(0, 5) != pl.col("AGS5"))
    ).select(*K, "AGS5", "Gemeindekennziffer", "Kreis", "Bundesland")
    multi = (p.group_by("year", "AGS5").agg(pl.col("Kreis").unique().sort())
              .filter(pl.col("Kreis").list.len() > 1).sort("year", "AGS5"))
    return rows, multi


def switchers(p):
    """Per attribute: stores whose value changes, and flip-backs (A->B->...->A)."""
    out = []
    for c in ATTR:
        s = (p.select(*K, c)
              .with_columns((pl.col(c).ne_missing(pl.col(c).shift().over("TDLinx"))
                             & (pl.col("year") != pl.col("year").min().over("TDLinx")))
                            .alias("chg"))
              .group_by("TDLinx").agg(
                  pl.col("chg").sum().alias("n_changes"),
                  pl.col(c).first().alias("first"), pl.col(c).last().alias("last"),
                  pl.col(c).alias("path"), pl.col("year").alias("years"))
              .filter(pl.col("n_changes") > 0)
              .with_columns(pl.lit(c).alias("attr"),
                            ((pl.col("n_changes") > 1) & pl.col("first").eq_missing(pl.col("last")))
                            .alias("flip_back")))
        out.append(s.with_columns(pl.col("first", "last").cast(pl.Utf8),
                                  pl.col("path").cast(pl.List(pl.Utf8))))
    return pl.concat(out).sort("attr", "TDLinx")


def switch_flows(p, attr="Gruppe"):
    """Year-to-year transitions old -> new with store counts — mergers vs. coding changes."""
    return (p.with_columns(pl.col(attr).shift().over("TDLinx").alias("from"))
             .filter(pl.col("from").ne_missing(pl.col(attr)) & pl.col("from").is_not_null())
             .group_by("year", "from", attr).agg(pl.len(), pl.col(V).sum().alias("VKFL"))
             .sort("year", "len", descending=[False, True]))


def gaps(p):
    """Stores that disappear and come back (holes in the panel)."""
    return (p.group_by("TDLinx").agg(pl.col("year").min().alias("y0"),
                                     pl.col("year").max().alias("y1"), pl.len().alias("n"))
             .filter(pl.col("y1") - pl.col("y0") + 1 > pl.col("n")).sort("TDLinx"))


def totals(p, by="Gruppe", jump=0.2):
    """National VKFL and revenue by actor and year, with year-on-year jumps > `jump`."""
    t = (p.group_by("year", by).agg(pl.len().alias("stores"), pl.col(V).sum(), pl.col(U).sum())
          .sort(by, "year")
          .with_columns([(pl.col(c) / pl.col(c).shift().over(by) - 1).alias(f"d_{c[:4]}")
                         for c in ("stores", V, U)]))
    flag = pl.any_horizontal(pl.col(f"d_{c[:4]}").abs() > jump for c in ("stores", V, U))
    return t, t.filter(flag)


def main(path, outdir="panel_checks"):
    out = Path(outdir); out.mkdir(exist_ok=True)
    p = load(path)
    print(f"{p.height:,} rows, {p['TDLinx'].n_unique():,} stores, years {p['year'].min()}-{p['year'].max()}")
    bad, shared, moves = coords(p)
    adm_rows, adm_multi = admin(p)
    t_all, t_jump = totals(p)
    sw = switchers(p)
    res = {
        "keys_dup": keys(p), "coverage": coverage(p), "missing": missing(p),
        "numerics": numerics(p), "components": components(p), "productivity": productivity(p),
        "coords_bad": bad, "coords_shared": shared, "coords_moves": moves,
        "admin_rows": adm_rows, "admin_multikreis": adm_multi,
        "switchers": sw, "flows_gruppe": switch_flows(p), "flows_shoptype": switch_flows(p, "Shop Type"),
        "gaps": gaps(p), "totals_gruppe": t_all, "totals_jumps": t_jump,
    }
    for k, df in res.items():
        lists = [c for c, t in df.schema.items() if isinstance(t, pl.List)]
        df.with_columns(pl.col(c).cast(pl.List(pl.Utf8)).list.join(">") for c in lists
                        ).write_csv(out / f"{k}.csv")
        print(f"{k:18s} {df.height:>8,}")
    print("flip-backs by attr:", sw.filter("flip_back").group_by("attr").len().sort("attr").rows())
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
