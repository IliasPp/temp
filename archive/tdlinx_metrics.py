"""
metrics.py — Metric functions operating on the panel DataFrame.

Three families:
  1. Concentration    — compute_hhi, collapse_hhi, combine_hhi
  2. Market shares    — market_share, market_share_by_kreis_and_group,
                        chain_area_market_share
  3. Panel regression — fe_* specifications on Umsatzpotential Brutto

Every metric accepts a `year` argument:
  - year=<int>   → compute for that single year
  - year="ALL"   → compute for all years (output always carries `year`)

Design notes
------------
* Concentration measures are computed in ONE pass over the panel with
  `year` as a grouping key, not in a Python loop over years.
* Nothing averages HHIs across markets. An HHI is a property of a market;
  the unweighted mean of several markets' HHIs is not an HHI. Use
  `collapse_hhi` when you need one number per region.
* Regression dummies are constructed explicitly rather than via `C()`,
  so the reference category is what you asked for and not whatever sorts
  first alphabetically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl

from ..config import (
    COMPETING_IN,
    DATA_DIR,
    MARKET_ACTOR,
    MARKET_DEFINING_COLS,
    TYPEN,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _year_filter(df: pl.DataFrame, year) -> pl.DataFrame:
    """Restrict to one year, or pass through for 'ALL'."""
    if year == "ALL" or year is None:
        return df
    return df.filter(pl.col("year") == year)


def _recode_shop_types(
    df: pl.DataFrame, shop_type_groups: dict | None, col: str = "Shop Type"
) -> pl.DataFrame:
    """
    Map Shop Type onto competitive-grouping labels, dropping types that
    belong to no group.

    This must happen BEFORE any groupby, never as a post-filter: grouping
    first and filtering later computes a separate HHI per Shop Type and
    then mixes them, which is not the HHI of the combined pool.
    """
    if shop_type_groups is None:
        return df
    if shop_type_groups is Ellipsis:
        raise TypeError(
            "shop_type_groups received `...` (Ellipsis). Pass an actual dict, "
            "e.g. {'grocery': ['Supermarkt', 'Discount Geschäft']}, or None."
        )
    type_to_group = {
        t: name for name, types in shop_type_groups.items() for t in types
    }
    return (
        df.with_columns(
            pl.col(col)
            .replace_strict(type_to_group, default=None, return_dtype=pl.Utf8)
            .alias(col)
        )
        .filter(pl.col(col).is_not_null())
    )


def _attach_kreis_names(hhi_df: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """
    Attach human-readable Kreis names to an AGS5-keyed frame.

    Skips silently if Kreis is already present — otherwise the join
    produces a `Kreis_right` column that every downstream selector misses.
    """
    if "AGS5" not in hhi_df.columns or "Kreis" not in panel.columns:
        return hhi_df
    if "Kreis" in hhi_df.columns:
        return hhi_df
    lookup = (
        panel.select(["AGS5", "Kreis"])
        .drop_nulls()
        .unique(subset=["AGS5"], keep="first")
    )
    return hhi_df.join(lookup, on="AGS5", how="left")


# ══════════════════════════════════════════════════════════════════════════════
# Concentration
# ══════════════════════════════════════════════════════════════════════════════

def compute_hhi(
    df: pl.DataFrame,
    market_cols: list[str] = None,
    firm_col: str = MARKET_ACTOR,
    weight_col: str = COMPETING_IN,
    label: str = "HHI",
    year="ALL",
    shop_type_groups: dict = None,
    panel: pl.DataFrame = None,
) -> pl.DataFrame:
    """
    Herfindahl–Hirschman index per market.

    Parameters
    ----------
    market_cols : list[str]
        Columns defining a market. Defaults to MARKET_DEFINING_COLS
        (["AGS5", "Shop Type"]). `year` is always added as a key.
    firm_col : str
        Firm identifier. Stores with a null firm are treated as
        single-store independents (their own TDLinx becomes the firm).
    weight_col : str
        Size measure whose shares are squared. Default COMPETING_IN.
    shop_type_groups : dict, optional
        {group_label: [shop types]}. When given, Shop Type is recoded to
        the group label before grouping, so a Supermarkt and a Discount
        Geschäft in the same group compete in one pool.
    panel : pl.DataFrame, optional
        Source panel used only to look up Kreis names. Defaults to `df`.

    Returns
    -------
    pl.DataFrame with market_cols, year, HHI, n_firms, market_total,
    n_stores_in_market, n_stores_used, weight_coverage_pct.

    The coverage columns are new and worth reading. Stores with a null
    weight are excluded from the market, so market_total understates the
    market and every share is inflated. Because VKFL is not missing at
    random, a market at 60% coverage produces an HHI that is not
    comparable to one at 99%.
    """
    market_cols = list(market_cols or MARKET_DEFINING_COLS)
    d = _year_filter(df, year)
    d = _recode_shop_types(d, shop_type_groups)

    keys = list(dict.fromkeys(["year"] + market_cols))

    keyed = d.drop_nulls(subset=keys)
    totals = keyed.group_by(keys).agg(pl.len().alias("n_stores_in_market"))

    x = (
        keyed
        .select(keys + [firm_col, weight_col, "TDLinx"])
        .drop_nulls(subset=[weight_col])
        .with_columns(
            pl.when(pl.col(firm_col).is_null())
            .then(pl.col("TDLinx").cast(pl.Utf8))
            .otherwise(pl.col(firm_col))
            .alias(firm_col)
        )
    )

    firm_market = x.group_by(keys + [firm_col]).agg(
        pl.col(weight_col).sum().alias("firm_weight"),
        pl.len().alias("firm_n_stores"),
    )
    market_totals = firm_market.group_by(keys).agg(
        pl.col("firm_weight").sum().alias("market_total"),
        pl.col("firm_n_stores").sum().alias("n_stores_used"),
    )
    firm_market = firm_market.join(market_totals, on=keys, how="left").with_columns(
        (pl.col("firm_weight") / pl.col("market_total")).alias("share")
    )

    result = (
        firm_market.group_by(keys)
        .agg(
            (pl.col("share") ** 2).sum().alias(label),
            pl.n_unique(firm_col).alias("n_firms"),
            pl.col("market_total").first(),
            pl.col("n_stores_used").first(),
        )
        .join(totals, on=keys, how="left")
        .with_columns(
            (pl.col("n_stores_used") / pl.col("n_stores_in_market") * 100)
            .alias("weight_coverage_pct")
        )
        .sort(keys)
    )

    return _attach_kreis_names(result, panel if panel is not None else df)


def collapse_hhi(
    hhi_df: pl.DataFrame,
    hhi_col: str = "HHI",
    group_key: str | list[str] = "AGS5",
    year: int = None,
    typ=None,
    weight_col: str = "market_total",
) -> pl.DataFrame:
    """
    Reduce a multi-market HHI frame to one row per region.

    Use this instead of taking a plain `.mean()` over Shop Types. An
    unweighted mean of separately-computed HHIs answers no question: a
    Kreis with one drugstore (HHI = 1.0) and a competitive grocery market
    (HHI = 0.15) averages to 0.58, which describes neither market.

    Weighting by market_total at least produces a size-weighted average
    concentration. Better still, pass `typ` and look at one market
    definition at a time.
    """
    keys = [group_key] if isinstance(group_key, str) else list(group_key)
    df = hhi_df
    if year is not None and "year" in df.columns:
        df = df.filter(pl.col("year") == year)
    if typ is not None and "Shop Type" in df.columns:
        types = [typ] if isinstance(typ, str) else list(typ)
        df = df.filter(pl.col("Shop Type").is_in(types))

    if weight_col in df.columns:
        agg = (
            (pl.col(hhi_col) * pl.col(weight_col)).sum() / pl.col(weight_col).sum()
        ).alias(hhi_col)
    else:
        agg = pl.col(hhi_col).mean().alias(hhi_col)

    return (
        df.drop_nulls(subset=keys + [hhi_col])
        .group_by(keys)
        .agg(agg, pl.len().alias("n_markets"))
        .sort(keys)
    )


def filter_by_typ(df: pl.DataFrame, typ: str = "ALL") -> pl.DataFrame:
    if typ == "ALL":
        return df
    return df.filter(pl.col("Shop Type") == typ)


def combine_hhi(
    typ: str = "ALL",
    bundesland_rev: pl.DataFrame = None,
    bundesland_area: pl.DataFrame = None,
    kreis_rev: pl.DataFrame = None,
    kreis_area: pl.DataFrame = None,
) -> pl.DataFrame:
    """Stack four HHI variants into one long frame with a `source` column."""
    named = {
        "HHI_Bundesland_rev":  bundesland_rev,
        "HHI_Bundesland_area": bundesland_area,
        "HHI_Kreis_rev":       kreis_rev,
        "HHI_Kreis_area":      kreis_area,
    }
    missing = [k for k, v in named.items() if v is None]
    if missing:
        raise ValueError(f"All four HHI DataFrames must be passed. Missing: {missing}")
    if typ != "ALL" and typ not in TYPEN:
        raise ValueError(f"Invalid Typ: {typ!r}. Must be 'ALL' or one of:\n{TYPEN}")

    combined = pl.concat(
        [df.with_columns(pl.lit(name).alias("source")) for name, df in named.items()],
        how="diagonal_relaxed",
    )
    if typ != "ALL":
        combined = combined.filter(pl.col("Shop Type") == typ)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# Market shares
# ══════════════════════════════════════════════════════════════════════════════

def market_share(
    df: pl.DataFrame,
    group_col: str = MARKET_ACTOR,
    value_col: str = COMPETING_IN,
    share_col: str = "market_share_pct",
    sort_desc: bool = True,
    year="ALL",
) -> pl.DataFrame:
    """National market share per group, per year. Single pass."""
    d = _year_filter(df, year).filter(
        pl.col(group_col).is_not_null(), pl.col(value_col).is_not_null()
    )
    return (
        d.group_by(["year", group_col])
        .agg(pl.col(value_col).sum().alias("total_value"),
             pl.len().alias("n_stores"))
        .with_columns(
            (pl.col("total_value") / pl.col("total_value").sum().over("year") * 100)
            .alias(share_col)
        )
        .sort(["year", "total_value"], descending=[False, sort_desc])
    )


def market_share_by_kreis_and_group(
    df: pl.DataFrame,
    group_col: str = MARKET_ACTOR,
    kreis_col: str = "Kreis",
    value_col: str = COMPETING_IN,
    year="ALL",
) -> pl.DataFrame:
    """Market share per (year, Kreis, group). Single pass."""
    d = _year_filter(df, year).filter(
        pl.col(kreis_col).is_not_null(),
        pl.col(group_col).is_not_null(),
        pl.col(value_col).is_not_null(),
    )
    return (
        d.group_by(["year", kreis_col, group_col])
        .agg(pl.col(value_col).sum().alias("group_value"),
             pl.len().alias("n_stores"))
        .with_columns(
            pl.col("group_value").sum().over(["year", kreis_col]).alias("kreis_total")
        )
        .with_columns(
            (pl.col("group_value") / pl.col("kreis_total") * 100).alias("market_share_pct")
        )
        .sort(["year", kreis_col, "market_share_pct"], descending=[False, False, True])
    )


def chain_area_market_share(
    catchments: pl.DataFrame,
    panel: pl.DataFrame,
    group_col: str = MARKET_ACTOR,
) -> pl.DataFrame:
    """
    Aggregate per-store catchment areas into per-(year, group_name, chain)
    area market shares.

    Edeka's grocery area share in a year is the summed cell area of all
    Edeka grocery stores over the total grocery cell area.

    Moved here from the notebook — it belongs with the other share metrics.
    """
    joined = catchments.join(
        panel.select(["TDLinx", "year", group_col]),
        on=["TDLinx", "year"], how="left",
    )
    return (
        joined
        .group_by(["year", "group_name", group_col])
        .agg(
            pl.col("cell_area_km2").sum().alias("chain_area_km2"),
            pl.len().alias("n_stores"),
        )
        .with_columns(
            pl.col("chain_area_km2").sum().over(["year", "group_name"])
            .alias("group_total_km2")
        )
        .with_columns(
            (pl.col("chain_area_km2") / pl.col("group_total_km2") * 100)
            .alias("area_market_share_pct")
        )
        .sort(["year", "group_name", "area_market_share_pct"],
              descending=[False, False, True])
    )


# ══════════════════════════════════════════════════════════════════════════════
# Panel regressions
# ══════════════════════════════════════════════════════════════════════════════
"""
Two-way and time-fixed-effects regressions of Umsatzpotential Brutto on
store characteristics. All specifications share the same prep pipeline:

  panel → filter to non-null revenue & VKFL (+ optional Gruppe filter)
        → select needed cols → pandas
        → set (TDLinx, year) MultiIndex
        → build explicit dummies with a chosen reference level
        → fit PanelOLS with entity-clustered SEs

A caveat worth keeping in view: if TDLinx derives Umsatzpotential from
sales area and format — which the clean per-format €/m² split suggests —
then regressing revenue on VKFL is close to tautological. These specs
describe the vendor's imputation model as much as retail economics.
"""

_BASE_COLS = [
    "TDLinx", "year",
    "Umsatzpotential Brutto (Mio EUR)",
    "VKFL", "Verkaufsfläche Food", "Kassen", "Parkplätze",
    "Shop Type", "Gruppe", "AGS5",
]

_DEFAULT_SHOP_TYPE_REF = "Supermarkt"
_DEFAULT_GRUPPE_REF = "Edeka"


def _import_panel_ols():
    """linearmodels is a heavy import — only load it when actually fitting."""
    from linearmodels import PanelOLS
    return PanelOLS


def _safe_name(prefix: str, level: str) -> str:
    """Turn a category level into a valid formula identifier."""
    cleaned = re.sub(r"[^0-9A-Za-zÄÖÜäöüß]+", "_", str(level)).strip("_")
    return f"{prefix}_{cleaned}"


def _make_dummies(df_pd, col: str, reference: str, prefix: str):
    """
    Build explicit 0/1 dummy columns for `col`, omitting `reference`.

    This replaces `C(Q('Shop Type'))` in the formulas. `C()` re-derives the
    level set and orders it alphabetically, which silently overrode the
    pandas Categorical ordering that _prep_panel_data was setting — that is
    why the Shop Type reference came out as "Angeschlossener GAM" (first
    alphabetically) no matter what `shop_type_reference` was set to.

    Building the columns here makes the reference explicit, deterministic,
    and visible in the coefficient names.

    Returns (df_pd, dummy_names, reference_used).
    """
    levels = sorted(x for x in df_pd[col].dropna().unique())
    if not levels:
        return df_pd, [], None
    if reference not in levels:
        reference = levels[0]

    dummy_names = []
    for level in levels:
        if level == reference:
            continue
        name = _safe_name(prefix, level)
        df_pd[name] = (df_pd[col] == level).astype(float)
        dummy_names.append(name)
    return df_pd, dummy_names, reference


def _prep_panel_data(
    panel: pl.DataFrame,
    gruppe: str = None,
    shop_type: str = None,
    extra_filters: list = None,
    shop_type_reference: str = _DEFAULT_SHOP_TYPE_REF,
    gruppe_reference: str = _DEFAULT_GRUPPE_REF,
    build_shop_type_dummies: bool = False,
    build_gruppe_dummies: bool = False,
):
    """
    Filter, select, convert to pandas with a (TDLinx, year) MultiIndex,
    and optionally construct explicit dummies.

    Returns (df_pd, meta) where meta records the dummy column names and
    the reference level actually used for each factor.
    """
    df = panel.filter(
        pl.col("Umsatzpotential Brutto (Mio EUR)").is_not_null(),
        pl.col("VKFL").is_not_null(),
        pl.col("VKFL") > 0,
    )
    if gruppe is not None:
        df = df.filter(pl.col("Gruppe") == gruppe)
    if shop_type is not None:
        df = df.filter(pl.col("Shop Type") == shop_type)
    if extra_filters:
        df = df.filter(*extra_filters)

    cols = [c for c in _BASE_COLS if c in df.columns]
    df_pd = (
        df.select(cols)
        .to_pandas()
        .set_index(["TDLinx", "year"])
        .rename(columns={"Umsatzpotential Brutto (Mio EUR)": "revenue"})
    )

    meta = {"n_obs": len(df_pd), "shop_type": {}, "gruppe": {}}

    if build_shop_type_dummies and "Shop Type" in df_pd.columns:
        df_pd, names, ref = _make_dummies(
            df_pd, "Shop Type", shop_type_reference, "ST"
        )
        meta["shop_type"] = {"dummies": names, "reference": ref}

    if build_gruppe_dummies and "Gruppe" in df_pd.columns:
        df_pd, names, ref = _make_dummies(
            df_pd, "Gruppe", gruppe_reference, "GR"
        )
        meta["gruppe"] = {"dummies": names, "reference": ref}

    return df_pd, meta


def _fit(formula: str, data, show: bool = False, meta: dict = None):
    PanelOLS = _import_panel_ols()
    model = PanelOLS.from_formula(formula, data=data)
    results = model.fit(cov_type="clustered", cluster_entity=True)
    results.reference_levels = {
        "Shop Type": (meta or {}).get("shop_type", {}).get("reference"),
        "Gruppe": (meta or {}).get("gruppe", {}).get("reference"),
    }
    if show:
        print(results.summary)
        refs = {k: v for k, v in results.reference_levels.items() if v}
        if refs:
            print("\nReference categories:", refs)
    return results


def _terms(*groups) -> str:
    """Join non-empty dummy name lists into formula terms."""
    out = []
    for g in groups:
        out.extend(g or [])
    return (" + " + " + ".join(out)) if out else ""


# ── Specifications ───────────────────────────────────────────────────────────

def fe_basic(panel: pl.DataFrame, show: bool = False):
    """
    Two-way FE: store FE + time FE.

    Identifies VKFL / Kassen / Parkplätze elasticities from within-store
    variation across years. Chain and Shop Type effects are absorbed by
    EntityEffects, so no dummies are built.
    """
    df, meta = _prep_panel_data(panel)
    return _fit(
        "revenue ~ 1 + VKFL + Kassen + Parkplätze + EntityEffects + TimeEffects",
        df, show, meta,
    )


def fe_chain_dummies(
    panel: pl.DataFrame,
    gruppe_reference: str = _DEFAULT_GRUPPE_REF,
    show: bool = False,
):
    """
    Time FE with Gruppe dummies.

    Coefficients read as "chain X earns N Mio EUR more/less than an
    otherwise-identical store of the reference chain". The reference is
    now explicit rather than whichever chain sorted first.
    """
    df, meta = _prep_panel_data(
        panel, gruppe_reference=gruppe_reference, build_gruppe_dummies=True
    )
    formula = (
        "revenue ~ 1 + VKFL + Kassen + Parkplätze"
        + _terms(meta["gruppe"].get("dummies"))
        + " + TimeEffects"
    )
    return _fit(formula, df, show, meta)


def fe_chain_and_format(
    panel: pl.DataFrame,
    shop_type_reference: str = _DEFAULT_SHOP_TYPE_REF,
    gruppe_reference: str = _DEFAULT_GRUPPE_REF,
    show: bool = False,
):
    """
    Time FE, Gruppe + Shop Type controls.

    Gruppe coefficients now compare chains within the same format — e.g.
    dm vs Rossmann as drugstores of comparable size.
    """
    df, meta = _prep_panel_data(
        panel,
        shop_type_reference=shop_type_reference,
        gruppe_reference=gruppe_reference,
        build_shop_type_dummies=True,
        build_gruppe_dummies=True,
    )
    formula = (
        "revenue ~ 1 + VKFL + Kassen + Parkplätze"
        + _terms(meta["gruppe"].get("dummies"), meta["shop_type"].get("dummies"))
        + " + TimeEffects"
    )
    return _fit(formula, df, show, meta)


def fe_vkfl_format_interaction(
    panel: pl.DataFrame,
    shop_type_reference: str = _DEFAULT_SHOP_TYPE_REF,
    gruppe_reference: str = _DEFAULT_GRUPPE_REF,
    show: bool = False,
):
    """
    Time FE, Gruppe + Shop Type + VKFL × Shop Type.

    Lets revenue per m² vary by format. The VKFL coefficient is now the
    per-m² revenue of the *reference format*; each VKFL:ST_x term is the
    difference from it. Previously the reference was silently whichever
    format sorted first, which made these numbers hard to interpret.
    """
    df, meta = _prep_panel_data(
        panel,
        shop_type_reference=shop_type_reference,
        gruppe_reference=gruppe_reference,
        build_shop_type_dummies=True,
        build_gruppe_dummies=True,
    )
    st = meta["shop_type"].get("dummies") or []
    interactions = [f"VKFL:{d}" for d in st]
    formula = (
        "revenue ~ 1 + VKFL + Kassen + Parkplätze"
        + _terms(st, interactions, meta["gruppe"].get("dummies"))
        + " + TimeEffects"
    )
    return _fit(formula, df, show, meta)


def fe_chain_vkfl_interaction(
    panel: pl.DataFrame,
    shop_type_reference: str = _DEFAULT_SHOP_TYPE_REF,
    gruppe_reference: str = _DEFAULT_GRUPPE_REF,
    show: bool = False,
):
    """
    Time FE, Gruppe + Shop Type + VKFL × Gruppe.

    Tests whether chains differ in marginal revenue per m² — does an extra
    m² at a Schwarz store generate more than at a Rewe store?
    """
    df, meta = _prep_panel_data(
        panel,
        shop_type_reference=shop_type_reference,
        gruppe_reference=gruppe_reference,
        build_shop_type_dummies=True,
        build_gruppe_dummies=True,
    )
    gr = meta["gruppe"].get("dummies") or []
    interactions = [f"VKFL:{d}" for d in gr]
    formula = (
        "revenue ~ 1 + VKFL + Kassen + Parkplätze"
        + _terms(gr, interactions, meta["shop_type"].get("dummies"))
        + " + TimeEffects"
    )
    return _fit(formula, df, show, meta)


def fe_single_chain(
    panel: pl.DataFrame,
    gruppe: str,
    shop_type_reference: str = _DEFAULT_SHOP_TYPE_REF,
    include_entity_fe: bool = False,
    show: bool = False,
):
    """
    Restrict the sample to one chain. Time FE always included; entity FE
    optional (default off — gives cleaner cross-sectional coefficients
    within the chain).
    """
    df, meta = _prep_panel_data(
        panel, gruppe=gruppe,
        shop_type_reference=shop_type_reference,
        build_shop_type_dummies=True,
    )
    formula = (
        "revenue ~ 1 + VKFL + Kassen + Parkplätze"
        + _terms(meta["shop_type"].get("dummies"))
        + " + TimeEffects"
    )
    if include_entity_fe:
        formula += " + EntityEffects"
    return _fit(formula, df, show, meta)


def fe_compare_chains(
    panel: pl.DataFrame,
    chains: list = None,
    coef: str = "VKFL",
    min_obs: int = 100,
    verbose: bool = True,
):
    """
    Run fe_single_chain per chain and compare a focal coefficient
    (default VKFL = revenue per m²) side by side.

    Returns a pandas DataFrame, one row per chain, with estimate, SE,
    t-stat, and sample sizes. Chains below `min_obs` are skipped rather
    than producing a coefficient nobody should read.
    """
    import pandas as pd

    if chains is None:
        chains = (
            panel.filter(pl.col("Gruppe").is_not_null())
            .group_by("Gruppe").agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_obs)
            .sort("n", descending=True)["Gruppe"].to_list()
        )

    rows, skipped = [], []
    for chain in chains:
        try:
            res = fe_single_chain(panel, gruppe=chain, show=False)
            if coef not in res.params.index:
                skipped.append((chain, f"{coef} not estimated"))
                continue
            rows.append({
                "chain": chain,
                "coef": res.params[coef],
                "se": res.std_errors[coef],
                "t": res.tstats[coef],
                "n_obs": int(res.nobs),
                "n_entities": int(res.entity_info.total),
            })
        except Exception as e:
            skipped.append((chain, str(e)[:80]))

    out = pd.DataFrame(rows).sort_values("coef", ascending=False)
    if verbose:
        print(f"\n{coef} coefficient across chains:")
        print(out.to_string(index=False))
        for chain, why in skipped:
            print(f"  skipped {chain}: {why}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Catchment areas
# ══════════════════════════════════════════════════════════════════════════════

_CATCHMENT_DIR = Path(DATA_DIR) / "catchments"


def pair_catchment_areas(
    panel: pl.DataFrame,
    clip_to,
    year: int,
    shop_type_groups: dict = None,
    crs: str = "EPSG:25832",
) -> pl.DataFrame:
    """
    For each store in `year`, compute its Voronoi catchment area within the
    competitive grouping defined by `shop_type_groups`.

    Each store is tessellated only against the other stores in its own
    group. A Supermarkt grouped with Discount Geschäft competes with both,
    but not with drugstores or beverage shops.

    Returns
    -------
    pl.DataFrame with TDLinx, year, Shop Type, group_name, cell_area_km2,
    group_total_area_km2, area_market_share_pct.
    """
    # Imported lazily: voronoi pulls in scipy, shapely and geopandas, which
    # made importing metrics.py for a simple HHI cost several seconds.
    from ..isochrones.voronoi import build_voronoi

    if shop_type_groups is Ellipsis:
        raise TypeError(
            "shop_type_groups received `...` (Ellipsis). Pass a dict or None."
        )
    if shop_type_groups is None:
        all_types = (
            panel.filter(pl.col("year") == year)
            .select("Shop Type").drop_nulls().unique()["Shop Type"].to_list()
        )
        shop_type_groups = {t: [t] for t in all_types}

    results = []
    for group_name, types in shop_type_groups.items():
        vor = build_voronoi(
            panel, year=year, shop_type=types, clip_to=clip_to, crs=crs
        )
        if len(vor) == 0:
            continue
        total_area = float(vor["cell_area_km2"].sum())
        results.append(
            pl.from_pandas(vor[["TDLinx", "Shop Type", "cell_area_km2"]])
            .with_columns([
                pl.lit(year).alias("year"),
                pl.lit(group_name).alias("group_name"),
                pl.lit(total_area).alias("group_total_area_km2"),
                (pl.col("cell_area_km2") / total_area * 100)
                .alias("area_market_share_pct"),
            ])
        )

    if not results:
        return pl.DataFrame(schema={
            "TDLinx": pl.Int64, "year": pl.Int64, "Shop Type": pl.Utf8,
            "group_name": pl.Utf8, "cell_area_km2": pl.Float64,
            "group_total_area_km2": pl.Float64, "area_market_share_pct": pl.Float64,
        })

    return pl.concat(results, how="diagonal_relaxed").select([
        "TDLinx", "year", "Shop Type", "group_name",
        "cell_area_km2", "group_total_area_km2", "area_market_share_pct",
    ])


def _groups_fingerprint(shop_type_groups: dict | None) -> str:
    """Stable string identifying a grouping, for cache invalidation."""
    if shop_type_groups is None:
        return "default_per_shop_type"
    return json.dumps(
        {k: sorted(v) for k, v in sorted(shop_type_groups.items())},
        ensure_ascii=False, sort_keys=True,
    )


def pair_catchment_areas_panel(
    panel: pl.DataFrame,
    clip_to,
    shop_type_groups: dict = None,
    out_dir=None,
    crs: str = "EPSG:25832",
    verbose: bool = True,
) -> pl.DataFrame:
    """
    Run pair_catchment_areas for every year, cached and resumable.

    Two changes from the previous implementation, both correctness issues:

    1. The cache key now includes `shop_type_groups`. Before, the cache was
       keyed on year alone, so running once with a grocery/drugstore/
       beverage grouping and again with a different grouping returned the
       FIRST run's numbers for both — silently, with no error. Different
       groupings now write to different subdirectories.

    2. Results are written as one part file per year instead of
       read-concat-rewrite. The old loop read the entire accumulated
       parquet back into memory, concatenated, and rewrote it on every
       iteration — quadratic I/O, and a crash mid-write left a truncated
       file that looked complete.
    """
    fingerprint = _groups_fingerprint(shop_type_groups)
    tag = re.sub(r"[^0-9A-Za-z]+", "_", fingerprint)[:60] or "default"

    base = Path(out_dir) if out_dir is not None else _CATCHMENT_DIR
    part_dir = base / tag
    part_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = part_dir / "_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps({"shop_type_groups": fingerprint, "crs": crs},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    all_years = sorted(panel["year"].unique().to_list())
    for year in all_years:
        part = part_dir / f"year={year}.parquet"
        if part.exists():
            if verbose:
                print(f"  {year}: cached")
            continue
        new = pair_catchment_areas(
            panel, clip_to, year, shop_type_groups=shop_type_groups, crs=crs
        )
        tmp = part.with_suffix(".parquet.tmp")
        new.write_parquet(tmp)
        tmp.replace(part)
        if verbose:
            print(f"  {year}: {len(new):,} rows")

    parts = sorted(part_dir.glob("year=*.parquet"))
    if not parts:
        raise RuntimeError(f"No catchment parts written under {part_dir}")
    return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")