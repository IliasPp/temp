"""
cleaning.py — Per-sheet cleaning pipeline for the TDLinx panel.

Typical usage
-------------
sheets = load_data()                          # from data_loading.py
sheets, report = clean(sheets, report=True)   # apply all cleaning steps
sheets = attach_coords_only(sheets)           # attach cached coordinates

Design principles
-----------------
1. Null means unknown. Nothing is silently zero-filled. If a value is
   missing it stays missing, and downstream code is responsible for
   deciding whether to drop or impute. `zero_fill()` exists but is
   opt-in and records what it touched.
2. Every destructive step (row drops, outlier nulling, deduplication)
   reports a count. Nothing disappears quietly.
3. All sheets end with identical column names *and dtypes*, so
   `pl.concat(..., how="diagonal_relaxed")` in panel.py cannot silently
   up- or down-cast a column.
"""

from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path

import polars as pl

from ..config import CSV_PATH


# ══════════════════════════════════════════════════════════════════════════════
# Schema definition
# ══════════════════════════════════════════════════════════════════════════════
#
# Three schema generations observed in the data:
#   2014–2017 : divergent schema (Strasse, Postleitzahl, VKFL Gesamt, etc.)
#   2018–2024 : canonical schema
#   2025      : canonical schema with two truncated column names
#
# All sheets are normalized to the canonical 2018–2024 schema.

_TO_CANONICAL = {
    # 2014–2017 mappings
    "Name postalisch":              "Name",
    "Name juristisch":              "Name-juristisch",
    "Strasse":                      "Straße",
    "Postleitzahl":                 "Plz",
    "Einwohner/Gemeinde":           "Einwohner pro Gemeinde",
    "Eröffnungs-/Übernahmedatum":   "Eröffnung / Übernahme",
    "VKFL Gesamt":                  "VKFL",
    "VKFL Food":                    "Verkaufsfläche Food",
    "VKFL Nonfood":                 "Verkaufsfläche Nonfood",
    "Gesamtumsatz":                 "Umsatzpotential Brutto (Mio EUR)",
    "Foodumsatz":                   "Umsatzpotential Brutto Food (Mio EUR)",
    "Nonfoodumsatz":                "Umsatzpotential Brutto Nonfood (Mio EUR)",
    "Anzahl Kassen":                "Kassen",
    "VL TDLinx":                    "Vertriebslinie TDLinx",
    "HZ TDLinx":                    "Handelszentrale TDLinx",
    "Telefon mit Vorwahl":          "Telefon",
    # 2025 truncations
    "Einwohner":                    "Einwohner pro Gemeinde",
    "Eröffnung":                    "Eröffnung / Übernahme",
}

# Present only in 2014–2017, no canonical equivalent
_SCHEMA_ONLY_COLS = {"Land", "Strasse 2", "Subtyp", "Betriebsform", "GLN (ILN)", "Status"}

# Year-specific stray columns, not used downstream
_STRAY_COLS = {"rechnung", "KKZ", "BKartA-relevant", "Telefax", "...39"}

# Canonical schema: column order AND dtype.
#
# The dtype half is what fixes the "missing column becomes all-zero" bug.
# Previously absent columns were added via `pl.lit(None)`, which produces
# dtype Null; `fill_null(0)` on a Null column yields an Int32 column of
# zeros, and `.str` accessors on it raise. Every column now gets an
# explicit dtype whether it was present in the source or not.
_CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "TDLinx":                                    pl.Int64,
    "Banner":                                    pl.Utf8,
    "Name":                                      pl.Utf8,
    "Name-juristisch":                           pl.Utf8,
    "Plz":                                       pl.Utf8,
    "Ort":                                       pl.Utf8,
    "Ortsteil":                                  pl.Utf8,
    "Straße":                                    pl.Utf8,
    "Typ":                                       pl.Utf8,
    "Nielsentyp":                                pl.Utf8,
    "Betriebsform zur Handelszentrale":          pl.Utf8,
    "Telefon":                                   pl.Utf8,
    "Ballungsraum":                              pl.Utf8,
    "Nielsengebiet":                             pl.Utf8,
    "Bundesland":                                pl.Utf8,
    "Regierungsbezirk":                          pl.Utf8,
    "Kreis":                                     pl.Utf8,
    "Gemeindekennziffer":                        pl.Utf8,
    "Einwohner pro Gemeinde":                    pl.Float64,
    "Eröffnung / Übernahme":                     pl.Utf8,
    "Ersteröffnungsdatum":                       pl.Utf8,
    "Wiedereröffnungsdatum":                     pl.Utf8,
    "Vertriebsregion Name":                      pl.Utf8,
    "Vertriebslinie TDLinx":                     pl.Utf8,
    "Vertriebslinie":                            pl.Utf8,
    "Handelszentrale TDLinx":                    pl.Utf8,
    "Handelszentrale":                           pl.Utf8,
    "Umsatzpotential Brutto (Mio EUR)":          pl.Float64,
    "Umsatzpotential Brutto Food (Mio EUR)":     pl.Float64,
    "Umsatzpotential Brutto Nonfood (Mio EUR)":  pl.Float64,
    "VKFL":                                      pl.Float64,
    "Verkaufsfläche Food":                       pl.Float64,
    "Verkaufsfläche Nonfood":                    pl.Float64,
    "Parkplätze":                                pl.Float64,
    "Kassen":                                    pl.Float64,
    "AGS5":                                      pl.Utf8,
}

_RAW_CANONICAL_COLS = list(_CANONICAL_SCHEMA)

# Derived columns added by this module, after the raw schema is fixed.
_DERIVED_COLS = ["Shop Type", "Gruppe", "Chain", "Gruppe_src"]

# Diagnostic flags added by this module. Kept out of CANONICAL_COLS so the
# panel column order is unchanged, but carried through so you can audit.
QUALITY_FLAGS = ["revenue_outlier", "area_mismatch"]

# Full canonical col order for the panel — includes derived cols
CANONICAL_COLS = _RAW_CANONICAL_COLS + _DERIVED_COLS

# Column groups used by the numeric parser and the area reconciler
_REVENUE_COLS = [
    "Umsatzpotential Brutto (Mio EUR)",
    "Umsatzpotential Brutto Food (Mio EUR)",
    "Umsatzpotential Brutto Nonfood (Mio EUR)",
]
_AREA_COLS = ["VKFL", "Verkaufsfläche Food", "Verkaufsfläche Nonfood"]
_COUNT_COLS = ["Parkplätze", "Kassen", "Einwohner pro Gemeinde"]


# ══════════════════════════════════════════════════════════════════════════════
# German number parsing
# ══════════════════════════════════════════════════════════════════════════════

def _german_number_expr(col: str, ambiguous_dot: str = "thousands") -> pl.Expr:
    """
    Parse a German-formatted numeric string column to Float64.

    Handles: "1.234.567,89", "1234,56", "1 234,56", "1.234", "1234.56".

    The previous implementation used `.str.replace(",", ".")`, which
    replaces only the FIRST match and ignores thousands separators, so
    "1.234,56" became "1.234.56" and cast to null. Silent data loss.

    A lone dot is genuinely ambiguous ("1.234" is 1234 or 1.234), and the
    right reading differs by column:
      ambiguous_dot="thousands" → areas, counts (VKFL is an integer m²)
      ambiguous_dot="decimal"   → revenue in Mio EUR (0.1–100 range)
    """
    s = (
        pl.col(col).cast(pl.Utf8)
        .str.strip_chars()
        # currency symbols, spaces, non-breaking spaces, apostrophe grouping
        .str.replace_all(r"[\s\u00a0'`´€]", "")
    )

    # German reading: strip dots (thousands), comma becomes the decimal point
    as_german = s.str.replace_all(r"\.", "").str.replace_all(",", ".")

    has_comma = s.str.contains(",", literal=True)
    # A dot used as a thousands separator always precedes exactly 3 digits,
    # in one or more groups: 1.234 / 12.345.678
    dot_is_thousands = s.str.contains(r"^-?\d{1,3}(?:\.\d{3})+$")

    if ambiguous_dot == "thousands":
        use_german = has_comma | dot_is_thousands
    elif ambiguous_dot == "decimal":
        use_german = has_comma
    else:
        raise ValueError(
            f"ambiguous_dot must be 'thousands' or 'decimal', got {ambiguous_dot!r}"
        )

    return (
        pl.when(use_german).then(as_german).otherwise(s)
        .cast(pl.Float64, strict=False)
        .alias(col)
    )


def parse_numerics(df: pl.DataFrame) -> pl.DataFrame:
    """Cast all numeric canonical columns from strings to Float64."""
    exprs = []
    for col in _REVENUE_COLS:
        if col in df.columns and df.schema[col] == pl.Utf8:
            exprs.append(_german_number_expr(col, ambiguous_dot="decimal"))
    for col in _AREA_COLS + _COUNT_COLS:
        if col in df.columns and df.schema[col] == pl.Utf8:
            exprs.append(_german_number_expr(col, ambiguous_dot="thousands"))
    return df.with_columns(exprs) if exprs else df


# ══════════════════════════════════════════════════════════════════════════════
# Schema normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Rename to canonical names, drop strays, add missing cols with dtype."""
    rename_map = {k: v for k, v in _TO_CANONICAL.items() if k in df.columns}
    if rename_map:
        # Guard against a rename colliding with an existing canonical column
        # (e.g. a sheet carrying both "Einwohner" and "Einwohner pro Gemeinde").
        collisions = {k: v for k, v in rename_map.items() if v in df.columns}
        for old in collisions:
            rename_map.pop(old)
        if collisions:
            df = df.drop(list(collisions))
        df = df.rename(rename_map)

    drop_cols = (_SCHEMA_ONLY_COLS | _STRAY_COLS) & set(df.columns)
    if drop_cols:
        df = df.drop(sorted(drop_cols))

    # Add absent canonical columns with their proper dtype, not Null dtype.
    missing = [c for c in _RAW_CANONICAL_COLS if c not in df.columns]
    if missing:
        df = df.with_columns([
            pl.lit(None, dtype=_CANONICAL_SCHEMA[c]).alias(c) for c in missing
        ])

    return df.select(_RAW_CANONICAL_COLS)


def enforce_dtypes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Cast every canonical column to its declared dtype.

    Run last, so all sheets are dtype-identical before concatenation.
    Without this, `diagonal_relaxed` concat can silently promote or
    demote a column when one year read as Int64 and another as Float64.
    """
    casts = []
    for col, dtype in _CANONICAL_SCHEMA.items():
        if col in df.columns and df.schema[col] != dtype:
            casts.append(pl.col(col).cast(dtype, strict=False).alias(col))
    return df.with_columns(casts) if casts else df


# ══════════════════════════════════════════════════════════════════════════════
# Null / whitespace normalization
# ══════════════════════════════════════════════════════════════════════════════

# Values that mean "no value" but arrive as literal text. Matched exactly
# (after stripping), not as substrings — the old code used replace_all with
# a substring match, so "Kein Banner Nord" would have been mangled.
_NULL_LIKE_GLOBAL = {"", "-", "--", "n/a", "N/A", "NA", "k.A.", "k. A.", "keine Angabe"}
_NULL_LIKE_BY_COL = {
    "Banner":     {"Kein Banner"},
    "Nielsentyp": {"ohne Nielsendefinition"},
}


def normalize_nulls(df: pl.DataFrame) -> pl.DataFrame:
    """
    Strip whitespace from string columns and convert null-like literals
    to proper nulls.
    """
    df = df.with_columns(pl.col(pl.Utf8).str.strip_chars())

    exprs = []
    for col in df.columns:
        if df.schema[col] != pl.Utf8:
            continue
        sentinels = _NULL_LIKE_GLOBAL | _NULL_LIKE_BY_COL.get(col, set())
        exprs.append(
            pl.when(pl.col(col).is_in(list(sentinels)))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    return df.with_columns(exprs) if exprs else df


# ══════════════════════════════════════════════════════════════════════════════
# Row filtering
# ══════════════════════════════════════════════════════════════════════════════

def drop_junk_rows(df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """
    Drop rows whose TDLinx is null or non-numeric (summary/header artefacts).

    The old regex `^\\d+$` was applied to `cast(pl.Utf8)`, which renders a
    Float64 TDLinx as "12345.0" and therefore matched nothing — silently
    dropping every row in any sheet where Excel inferred a float. This
    casts to Int64 instead, which is dtype-agnostic.
    """
    if "TDLinx" not in df.columns:
        return df, 0
    before = len(df)
    out = df.with_columns(
        pl.col("TDLinx").cast(pl.Int64, strict=False).alias("_tdlinx_num")
    ).filter(
        pl.col("_tdlinx_num").is_not_null() & (pl.col("_tdlinx_num") > 0)
    ).with_columns(
        pl.col("_tdlinx_num").alias("TDLinx")
    ).drop("_tdlinx_num")
    return out, before - len(out)


def drop_duplicates(
    df: pl.DataFrame, strategy: str = "drop_all"
) -> tuple[pl.DataFrame, int]:
    """
    Resolve TDLinx values appearing more than once within a sheet.

    strategy="drop_all"  : remove every copy (previous behaviour)
    strategy="keep_first": keep one row per TDLinx

    "drop_all" removes the store from that year entirely, which punches a
    hole in the panel and makes it unbalanced. Whichever you choose, the
    count is now reported instead of vanishing.
    """
    if "TDLinx" not in df.columns:
        return df, 0
    dup_mask = pl.col("TDLinx").count().over("TDLinx") > 1
    n_dup_rows = df.filter(dup_mask).height
    if n_dup_rows == 0:
        return df, 0
    if strategy == "drop_all":
        return df.filter(~dup_mask), n_dup_rows
    if strategy == "keep_first":
        return df.unique(subset=["TDLinx"], keep="first", maintain_order=True), n_dup_rows
    raise ValueError(f"strategy must be 'drop_all' or 'keep_first', got {strategy!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Revenue: units and outliers
# ══════════════════════════════════════════════════════════════════════════════
#
# The Umsatzpotential columns are not on a consistent scale across years.
# The previous code capped values above 200 Mio EUR to null, which does not
# fix a unit mismatch — it truncates the right tail while leaving small
# values in the wrong unit, producing a scale break across years that
# TimeEffects absorbs in mean but not in slope.
#
# Fill REVENUE_YEAR_SCALE after running audit_revenue_scale() below. An
# empty dict means "no rescaling applied", which is the honest default
# until the per-year factors have actually been established.

REVENUE_YEAR_SCALE: dict[str, float] = {
    # e.g. "2014": 1 / 1000,
}

_REVENUE_MAX_PLAUSIBLE = 200.0


def audit_revenue_scale(sheets: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Per-year distribution of the revenue columns, to determine the unit
    of each year before deciding on REVENUE_YEAR_SCALE.

    Run this on *parsed but unscaled* sheets. A year whose median is
    ~1000× another year's median is on a different unit, not full of
    outliers.
    """
    rows = []
    for year, df in sorted(sheets.items()):
        for col in _REVENUE_COLS:
            if col not in df.columns:
                continue
            s = df[col].cast(pl.Float64, strict=False).drop_nulls()
            if s.len() == 0:
                continue
            rows.append({
                "year": year,
                "column": col,
                "n": s.len(),
                "median": float(s.median()),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
            })
    return pl.DataFrame(rows)


def apply_revenue_scale(df: pl.DataFrame, year: str) -> pl.DataFrame:
    """Multiply revenue columns by the year's unit-correction factor."""
    factor = REVENUE_YEAR_SCALE.get(str(year))
    if factor is None:
        return df
    return df.with_columns([
        (pl.col(c) * factor).alias(c) for c in _REVENUE_COLS if c in df.columns
    ])


def flag_revenue_outliers(
    df: pl.DataFrame, max_plausible: float = _REVENUE_MAX_PLAUSIBLE
) -> tuple[pl.DataFrame, int]:
    """
    Null implausible total-revenue values and record which rows were hit.

    Crucially, the value stays null. In the previous pipeline these nulls
    were immediately zero-filled by fill_null_numerics, so every outlier
    re-entered the data as a hard zero and passed every `is_not_null()`
    guard downstream. That single interaction was the largest source of
    corrupted revenue in the panel.
    """
    total = "Umsatzpotential Brutto (Mio EUR)"
    if total not in df.columns:
        return df.with_columns(pl.lit(False).alias("revenue_outlier")), 0

    df = df.with_columns(
        (pl.col(total) > max_plausible).fill_null(False).alias("revenue_outlier")
    )
    n = int(df["revenue_outlier"].sum())
    if n:
        # Null the whole revenue row: the components are inconsistent with a
        # rejected total, so keeping Food/Nonfood would be incoherent.
        df = df.with_columns([
            pl.when(pl.col("revenue_outlier")).then(None).otherwise(pl.col(c)).alias(c)
            for c in _REVENUE_COLS if c in df.columns
        ])
    return df, n


# ══════════════════════════════════════════════════════════════════════════════
# Sales area reconciliation
# ══════════════════════════════════════════════════════════════════════════════

def reconcile_areas(df: pl.DataFrame, tolerance: float = 0.05) -> pl.DataFrame:
    """
    Reconcile VKFL against its Food / Nonfood components.

    1. Zero means unknown for all three columns. A store with 0 m² of
       sales floor is not a store — and VKFL is COMPETING_IN, so a zero
       there becomes a zero HHI weight and a zero regressor.
       (The old code nulled zeros in Food/Nonfood but left VKFL == 0
       intact, which is the one that actually matters.)
    2. If exactly one component is known, the other is a true structural
       zero — a pure-food store has no nonfood area. Fill it.
    3. Flag rows where Food + Nonfood disagrees with VKFL by more than
       `tolerance`, rather than silently trusting either.
    """
    food, non, total = "Verkaufsfläche Food", "Verkaufsfläche Nonfood", "VKFL"
    present = [c for c in (food, non, total) if c in df.columns]
    if not present:
        return df.with_columns(pl.lit(False).alias("area_mismatch"))

    # 1. zero → null
    df = df.with_columns([
        pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
        for c in present
    ])

    # 2. complement fill, only when the sibling is known
    if food in df.columns and non in df.columns:
        df = df.with_columns([
            pl.when(pl.col(food).is_null() & pl.col(non).is_not_null())
              .then(pl.lit(0.0)).otherwise(pl.col(food)).alias(food),
            pl.when(pl.col(non).is_null() & pl.col(food).is_not_null())
              .then(pl.lit(0.0)).otherwise(pl.col(non)).alias(non),
        ])

    # 3. consistency flag
    if all(c in df.columns for c in (food, non, total)):
        components = pl.col(food) + pl.col(non)
        df = df.with_columns(
            (
                components.is_not_null()
                & pl.col(total).is_not_null()
                & (pl.col(total) > 0)
                & ((components - pl.col(total)).abs() / pl.col(total) > tolerance)
            ).fill_null(False).alias("area_mismatch")
        )
    else:
        df = df.with_columns(pl.lit(False).alias("area_mismatch"))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Opt-in zero filling
# ══════════════════════════════════════════════════════════════════════════════

def zero_fill(df: pl.DataFrame, cols: list[str], verbose: bool = True) -> pl.DataFrame:
    """
    Explicitly zero-fill named columns. NOT part of the default pipeline.

    This replaces the old `fill_null_numerics`, which zero-filled revenue,
    population and counts unconditionally — after outliers had been nulled,
    so rejected values came back as zeros.

    Only call this for columns where you can defend "null means zero" for
    that specific column in that specific analysis, and never for revenue.
    """
    banned = set(_REVENUE_COLS) & set(cols)
    if banned:
        raise ValueError(
            f"Refusing to zero-fill revenue columns: {sorted(banned)}. "
            "A missing Umsatzpotential is unknown, not zero — filling it "
            "puts fabricated zeros into every regression sample."
        )
    exprs, filled = [], {}
    for col in cols:
        if col not in df.columns:
            continue
        n = int(df[col].null_count())
        if n:
            filled[col] = n
        exprs.append(pl.col(col).fill_null(0).alias(col))
    if verbose and filled:
        for col, n in filled.items():
            print(f"  zero_fill: {col} — {n:,} nulls filled")
    return df.with_columns(exprs) if exprs else df


# ══════════════════════════════════════════════════════════════════════════════
# Derived columns
# ══════════════════════════════════════════════════════════════════════════════

_SHOP_TYPE_MAP = {
    "Kleiner Supermarkt":  "Supermarkt",
    "Großer Supermarkt":   "Supermarkt",
    "Supermarkt":          "Supermarkt",
    "SB-Warenhaus":        "Supermarkt",
    "Verbrauchermarkt":    "Supermarkt",
    "Angeschlossener GAM": "Getränkemarkt",
    "Getränkemarkt":       "Getränkemarkt",
    "Discount Geschäft":   "Discount Geschäft",
    "Drogeriemarkt":       "Drogeriemarkt",
}


def derive_shop_type(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preserve raw `Typ` and derive the aggregated `Shop Type`.

    Now an explicit exact-match map rather than `str.contains("Supermarkt")`.
    The substring test also caught anything else containing the word, and
    silently depended on rule order.

    Note: "Angeschlossener GAM" is folded into "Getränkemarkt" here, so it
    should NOT survive into `Shop Type`. If your regression output shows it
    as the reference category, that model ran on a panel built before this
    mapping — worth re-checking.
    """
    if "Typ" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("Shop Type"))
    return df.with_columns(
        pl.col("Typ")
        .replace_strict(_SHOP_TYPE_MAP, default=pl.col("Typ"), return_dtype=pl.Utf8)
        .alias("Shop Type")
    )


# ── Gruppe assignment ─────────────────────────────────────────────────────────
#
# Netto is resolved by LEGAL ENTITY, and the rule is a single test:
#
#     Name-juristisch contains "ApS"  →  Netto (Salling)
#     any other Netto                 →  Edeka
#
# "ApS" (Anpartsselskab) is the Danish legal form and appears only in
# Salling Group's entities. Everything else trading as Netto is Edeka's
# Netto Marken-Discount — including Handelszentrale "Netto" paired with
# Name-juristisch "Netto II AG & Co. oHG".
#
# Do NOT infer this from Handelszentrale. In the real panel:
#     "Netto"                 → Edeka   (Marken-Discount AG, Netto II AG)
#     "Netto Marken-Discount" → Edeka
#     "Netto Nord"            → Salling (Netto ApS & Co. KG)
# Two earlier versions of this function got the split wrong in opposite
# directions by reasoning from the trading names — first assigning all
# bare "Netto" to Edeka, then all of it to Salling. The legal form is the
# only field that decides it. Getting this wrong moves stores between two
# firms that genuinely compete, shifting both Edeka's share and the HHI.

# Danish legal form — the sole discriminator, tested on Name-juristisch only.
_NETTO_SALLING_LEGAL = r"\baps\b"
# Any store trading under the Netto name, either owner.
_NETTO_ANY = r"\bnetto\b"
# Fallback for the rare row where Name-juristisch is null: "Netto Nord" is
# Salling's regional Handelszentrale label.
_NETTO_SALLING_HZ = {"Netto Nord"}

_HANDELSZENTRALE_TO_GRUPPE: dict[str, str] = {
    # Exact Handelszentrale → Gruppe entries for chains where that field is
    # unambiguous. Netto is deliberately absent — it is resolved by legal
    # entity below. Extend this from audit_handelszentrale(df) rather than
    # from outside knowledge of who owns what; TDLinx's labels do not
    # always follow the retail brand structure.
}

#: fields searched by rules built with _any_of(). "Vertriebslinie" is not in
#: it, so Vertriebslinie is consulted only by rules that list it explicitly
#: (dm, Müller), even though it sits second in _FIELD_ORDER.
_NAME_FIELDS = ["Handelszentrale", "Name", "Name-juristisch", "Banner"]

#: Evidence order: the first field that names a group decides. The banner is
#: what the store trades as and is re-recorded every year; the buying centre is
#: next (and is gated for independents, see gate_supplied); the legal name is
#: often stale after a merchant switches groups ("Edeka Meier GmbH" trading as
#: Rewe); the shop's own name is the weakest evidence of all.
_FIELD_ORDER = ["Banner", "Vertriebslinie", "Handelszentrale", "Name-juristisch", "Name"]


def _any_of(columns: list[str], pattern: str) -> list[tuple[str, str]]:
    return [(c, pattern) for c in columns]


# Ordered, first-match-wins within a field. Patterns carry word boundaries:
# without them "aldi" matched "Geraldine" and "Laldin", "norma" matched
# "Norman", "rewe" would match "Brewer". Hoffmann is read from the buying
# centre only -- as a legal or shop name it is a surname, and 43 unrelated
# village shops called Hoffmann were being merged into one competitor.
_GRUPPE_FALLBACK_RULES: list[dict] = [
    {"label": "Edeka",     "any": _any_of(_NAME_FIELDS, _NETTO_ANY)},

    {"label": "Edeka",     "any": _any_of(_NAME_FIELDS,
                                          r"edeka|\bnp\b|marktkauf|tante enso|nah\s*&\s*gut")},
    {"label": "Schwarz",   "any": _any_of(_NAME_FIELDS, r"\blidl\b|\bkaufland\b")},
    {"label": "Aldi",      "any": _any_of(_NAME_FIELDS, r"\baldi\b")},
    {"label": "Rewe",      "any": _any_of(_NAME_FIELDS, r"\brewe\b|\bpenny\b|\bnahkauf\b|\btoom\b")},
    {"label": "Rossmann",  "any": _any_of(_NAME_FIELDS, r"\brossmann\b")},
    {"label": "dm",        "any": [("Handelszentrale", r"^dm\b|dm-drogerie"),
                                   ("Name-juristisch", r"^dm\b|dm-drogerie"),
                                   ("Vertriebslinie",  r"^dm\b|dm-drogerie")]},
    {"label": "Müller",    "any": [("Handelszentrale", r"müller"),
                                   ("Vertriebslinie",  r"müller drogerie")]},
    {"label": "Kaiser's Tengelmann",
                           "any": _any_of(_NAME_FIELDS, r"\bkaiser'?s\b|\btengelmann\b")},
    {"label": "Klaas + Kock",
                           "any": _any_of(_NAME_FIELDS, r"\bklaas\s*\+\s*kock\b|\bk\s*\+\s*k\b")},
    {"label": "Globus",    "any": _any_of(_NAME_FIELDS, r"\bglobus\b")},
    {"label": "Norma",     "any": _any_of(_NAME_FIELDS, r"\bnorma\b")},
    {"label": "Tegut",     "any": _any_of(_NAME_FIELDS, r"\btegut\b")},
    {"label": "Fristo",    "any": [("Handelszentrale", r"\bfristo\b"),
                                   ("Name-juristisch", r"\bfristo\b")]},
    {"label": "Hoffmann",  "any": [("Handelszentrale", r"hoffmann")]},
    {"label": "Dennree",   "any": _any_of(_NAME_FIELDS, r"\bdennree\b|\bdenn[´'’`]?s\b")},
    {"label": "Bünting",   "any": _any_of(_NAME_FIELDS, r"\bbünting\b|\bcombi\b|\bfamila\b")},
    {"label": "Bartels-Langness",
                           "any": _any_of(_NAME_FIELDS, r"\bbartels[- ]?langness\b|\bbela\b")},
]


def _contains_ci(col: str, pattern: str) -> pl.Expr:
    """Null-safe, case-insensitive regex contains."""
    return pl.col(col).is_not_null() & pl.col(col).str.contains(f"(?i){pattern}")


def assign_gruppe(df: pl.DataFrame, fallback_to_handelszentrale: bool = True) -> pl.DataFrame:
    """
    Assign Gruppe and record which field decided it (Gruppe_src).

    Order of resolution per row:
      1. Netto by legal entity: "ApS" in Name-juristisch is Salling; a Netto
         with no legal entity is Salling if its buying centre is "Netto Nord",
         otherwise Edeka.
      2. Handelszentrale exact-matches _HANDELSZENTRALE_TO_GRUPPE.
      3. The rules, field by field in _FIELD_ORDER: the first field in which
         any rule matches decides, and within a field the first rule wins.
      4. If fallback_to_handelszentrale=True (default), anything still
         unmatched carries its raw Handelszentrale through as its own group
         (Gruppe_src "Handelszentrale (raw)"). Set False to leave it null.

    Gruppe_src is what gate_supplied() needs: a group that rests only on the
    buying centre of an independently run store is a supply relationship
    until shown otherwise. Verify with audit_handelszentrale(df).
    """
    available = {c for c in df.columns if df.schema[c] == pl.Utf8}
    lab = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
    src = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))

    def add(cond, label, source):
        nonlocal lab, src
        lab = lab.when(cond).then(pl.lit(label))
        src = src.when(cond).then(pl.lit(source))

    # Step 1: Netto, by legal entity.
    if "Name-juristisch" in available:
        add(_contains_ci("Name-juristisch", _NETTO_SALLING_LEGAL), "Netto (Salling)",
            "Name-juristisch")
    if "Handelszentrale" in available and "Name-juristisch" in available:
        netto_no_legal = (_contains_ci("Handelszentrale", _NETTO_ANY)
                          & pl.col("Name-juristisch").is_null())
        add(netto_no_legal & pl.col("Handelszentrale").is_in(list(_NETTO_SALLING_HZ)),
            "Netto (Salling)", "Handelszentrale")
        add(netto_no_legal, "Edeka", "Handelszentrale")

    # Step 2: exact Handelszentrale lookup.
    if "Handelszentrale" in available:
        for key, label in _HANDELSZENTRALE_TO_GRUPPE.items():
            add(pl.col("Handelszentrale") == key, label, "Handelszentrale")

    # Step 3: rules, strongest field first.
    for field in _FIELD_ORDER:
        if field not in available:
            continue
        for rule in _GRUPPE_FALLBACK_RULES:
            pats = [p for c, p in rule["any"] if c == field]
            if pats:
                add(_contains_ci(field, "|".join(f"(?:{p})" for p in pats)),
                    rule["label"], field)

    # Step 4: raw Handelszentrale as a last resort, or null.
    if fallback_to_handelszentrale and "Handelszentrale" in available:
        lab = lab.otherwise(pl.col("Handelszentrale"))
        src = (src.when(pl.col("Handelszentrale").is_not_null())
               .then(pl.lit("Handelszentrale (raw)"))
               .otherwise(pl.lit(None, dtype=pl.Utf8)))
    else:
        lab = lab.otherwise(pl.lit(None, dtype=pl.Utf8))
        src = src.otherwise(pl.lit(None, dtype=pl.Utf8))

    return df.with_columns(lab.alias("Gruppe"), src.alias("Gruppe_src"))


# gate_supplied() is NOT part of clean(): clean() works sheet by sheet, and the
# gate needs the later years. panel.build_panel() calls it on the concatenated
# panel, after the actor table and before finalize_chain(). It is idempotent.
def gate_supplied(panel: pl.DataFrame, since: int = 2020, *,
                  relation: str = "Betriebsform zur Handelszentrale",
                  verbose: bool = True) -> pl.DataFrame:
    """
    Remove group attribution that rests only on a supply relationship.

    Runs on the concatenated panel, not per sheet: it needs the later years.

    Problem. "Selbständig" in the relationship field covers two different
    things: merchants who are members of the Edeka or Rewe cooperative, and
    independent shops that merely buy from a group's wholesale arm. Up to 2019
    TDLinx filled the buying centre for both; from 2020 it leaves it blank for
    shops without a group affiliation. Read year by year, those shops are
    "Edeka" until 2019 and unidentified afterwards -- a trend made by a coding
    change, which showed as a dip in Edeka's catchment share.

    Rule. From `since` on, TDLinx's coding is taken as the reference for which
    independently run stores belong to a group. Before `since`, a Selbständig
    store whose group came only from the buying centre (Gruppe_src
    "Handelszentrale") keeps it if either
      * its Banner is one that independently run stores of that group carry
        in the reference years (Feneberg, Kupsch, ... -- read from the data,
        not listed by hand), or
      * the same store is attached to that group in any reference year.
    Otherwise it is a supplied customer: Gruppe null, Gruppe_src "supplied".
    Stores whose group comes from their banner, or which are run by the group
    (Regie), are never touched, and neither is any row from `since` on.
    """
    need = {"Gruppe", "Gruppe_src", relation, "Banner", "TDLinx", "year"}
    if missing := need - set(panel.columns):
        raise ValueError(f"gate_supplied: panel lacks {sorted(missing)}")
    via_hz = (pl.col("Gruppe_src") == "Handelszentrale") & (pl.col(relation) == "Selbständig")
    ref = panel.filter(pl.col("year") >= since, via_hz)
    allow = (ref.filter(pl.col("Banner").is_not_null()).select("Gruppe", "Banner").unique()
             .with_columns(pl.lit(True).alias("_ab")))
    attached = (ref.select("Gruppe", pl.col("TDLinx").cast(pl.Utf8).alias("_id")).unique()
                .with_columns(pl.lit(True).alias("_at")))
    p = (panel.with_row_index("_r")
         .with_columns(pl.col("TDLinx").cast(pl.Utf8).alias("_id"))
         .join(allow, on=["Gruppe", "Banner"], how="left")
         .join(attached, on=["Gruppe", "_id"], how="left")
         .sort("_r"))
    drop = (via_hz & (pl.col("year") < since)
            & pl.col("_ab").is_null() & pl.col("_at").is_null())
    p = p.with_columns(drop.alias("_drop"))
    if verbose:
        v = "VKFL" if "VKFL" in p.columns else None
        agg = [pl.len().alias("stores")] + ([pl.col(v).sum().round(0).alias("vkfl")] if v else [])
        out = p.filter(pl.col("_drop")).group_by("year", "Gruppe").agg(agg).sort("year", "Gruppe")
        print(f"gate_supplied: {out['stores'].sum() if out.height else 0:,} store-years before "
              f"{since} lose a group that rested only on their supplier "
              f"(reference: {allow.height} banners, {attached.height:,} stores)")
        if out.height:
            print(out)
    return (p.with_columns(
                pl.when(pl.col("_drop")).then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(pl.col("Gruppe")).alias("Gruppe"),
                pl.when(pl.col("_drop")).then(pl.lit("supplied"))
                .otherwise(pl.col("Gruppe_src")).alias("Gruppe_src"))
            .drop("_r", "_id", "_ab", "_at", "_drop"))


# ── Chain assignment ──────────────────────────────────────────────────────────
#
# Chain is the banner the shopper sees (Lidl, Kaufland, E-center), one level
# below Gruppe. Per sheet it is just the banner. Its final form is set on the
# whole panel by finalize_chain(): one spelling per banner, a table override
# where actors.csv gives one, and "<Gruppe> (other)" for stores without a
# banner. The shop's own name is never used -- it made every independent its
# own chain and merged unrelated shops called "Dorfladen".

def assign_chain(df: pl.DataFrame) -> pl.DataFrame:
    """Chain = Banner as TDLinx records it (null where there is none).
    finalize_chain() completes it on the panel."""
    if "Banner" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("Chain"))
    return df.with_columns(pl.col("Banner").alias("Chain"))


# ── Actor table (actors.csv) ──────────────────────────────────────────────────
#
# The rules above find most groups, but some decisions are facts about firms
# that no pattern can know: Edeka banners without the word "Edeka" (E-center),
# two labels for one firm (Mein Real / Real), a wholesaler's independent
# customers (not the wholesaler's own stores). They live in one reviewed table,
# applied to the whole panel after the per-sheet rules. docs/actors.md
# describes the format; scripts/audit_actors.py finds the cases.

ACTORS_CSV = Path(__file__).with_name("actors.csv")
_TABLE_FIELDS = ("Banner", "Handelszentrale", "Gruppe")
_LEGAL_FORMS = (r"\b(gmbh|mbh|kg|ohg|ag|eg|e\s?k|e\s?kfm|co|ug|se|aps|gbr|inh"
                r"|haftungsbeschr\w*)\b")
#: Gruppe_src values meaning "the group came from the buying centre"
_FROM_HZ = ("Handelszentrale", "Handelszentrale (raw)")


def name_key(e: pl.Expr) -> pl.Expr:
    """Spelling-insensitive key of a name: case, umlauts, punctuation (also
    the three apostrophes TDLinx uses) and legal forms removed.
    "Denn´s Biomarkt" and "Denn's Biomarkt" share a key, so do "E-center" and
    "E center"."""
    return (e.str.to_lowercase()
            .str.replace_all("ä", "ae").str.replace_all("ö", "oe")
            .str.replace_all("ü", "ue").str.replace_all("ß", "ss")
            .str.replace_all("&", " und ").str.replace_all(r"[^a-z0-9 ]", " ")
            .str.replace_all(_LEGAL_FORMS, " ").str.replace_all(r"\s+", " ")
            .str.strip_chars())


def load_actor_table(path: str | Path = ACTORS_CSV,
                     statuses: tuple[str, ...] = ("confirmed", "proposed")) -> pl.DataFrame:
    """actors.csv with only the rows to apply (status), in file order, plus
    the match key of `value`. Rows marked `open` are listed but not applied."""
    t = pl.read_csv(path, comment_prefix="#", infer_schema_length=0)
    t = t.with_columns(pl.all().fill_null(""))
    bad = set(t["field"].unique()) - set(_TABLE_FIELDS)
    if bad:
        raise ValueError(f"actors.csv: unknown field(s) {sorted(bad)}; use {_TABLE_FIELDS}")
    if (dup := t["id"].is_duplicated()).any():
        raise ValueError(f"actors.csv: duplicate id(s) {t.filter(dup)['id'].unique().to_list()}")
    return (t.filter(pl.col("status").is_in(list(statuses)))
            .with_columns(name_key(pl.col("value")).alias("_k"),
                          *[pl.col(c).replace("", None).cast(pl.Int32, strict=True).alias(c)
                            for c in ("year_from", "year_to")]))


def apply_actor_table(panel: pl.DataFrame, table: pl.DataFrame | None = None) -> pl.DataFrame:
    """Override Gruppe (and record a Chain) where actors.csv has a matching row.

    Runs on the concatenated panel: rows can be limited to years. For every
    store-year the FIRST matching row decides, in file order. A row matches
    when the store's `field`, compared by name_key(), equals `value`, and its
    optional conditions hold: `relation` (Betriebsform zur Handelszentrale),
    `src` = "hz" (the group came from the buying centre, not from a banner),
    `year_from` / `year_to` (inclusive).

    gruppe "independent": Gruppe null, Gruppe_src "independent" -- the store is
    no firm's; the cube counts it as its own small competitor.
    gruppe empty: Gruppe unchanged (a row that only sets the chain).
    Otherwise Gruppe = gruppe, Gruppe_src = "table". `actor_rule` keeps the id.
    """
    t = load_actor_table() if table is None else table
    if t.is_empty():
        return panel.with_columns(pl.lit(None, dtype=pl.Utf8).alias("actor_rule"),
                                  pl.lit(None, dtype=pl.Utf8).alias("Chain_table"))
    rel = "Betriebsform zur Handelszentrale"
    p = panel.with_columns([name_key(pl.col(f)).alias(f"_k_{f}")
                            for f in _TABLE_FIELDS if f in panel.columns])
    rule = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.Utf8))
    for r in t.iter_rows(named=True):
        if f"_k_{r['field']}" not in p.columns:
            continue
        c = pl.col(f"_k_{r['field']}") == r["_k"]
        if r["relation"]:
            c &= pl.col(rel) == r["relation"]
        if r["src"] == "hz":
            c &= pl.col("Gruppe_src").is_in(list(_FROM_HZ))
        elif r["src"]:
            raise ValueError(f"actors.csv row {r['id']}: src must be empty or 'hz'")
        if r["year_from"] is not None:
            c &= pl.col("year") >= r["year_from"]
        if r["year_to"] is not None:
            c &= pl.col("year") <= r["year_to"]
        rule = rule.when(c).then(pl.lit(r["id"]))
    p = (p.with_columns(rule.otherwise(None).alias("actor_rule")).with_row_index("_r")
         .join(t.select(pl.col("id").alias("actor_rule"), pl.col("gruppe").alias("_g"),
                        pl.col("chain").alias("Chain_table")),
               on="actor_rule", how="left")
         .sort("_r").drop("_r"))
    hit, g = pl.col("actor_rule").is_not_null(), pl.col("_g")
    return (p.with_columns(
                pl.when(hit & (g == "independent")).then(pl.lit(None, dtype=pl.Utf8))
                .when(hit & (g != "")).then(g).otherwise(pl.col("Gruppe")).alias("Gruppe"),
                pl.when(hit & (g == "independent")).then(pl.lit("independent"))
                .when(hit & (g != "")).then(pl.lit("table"))
                .otherwise(pl.col("Gruppe_src")).alias("Gruppe_src"),
                pl.when(pl.col("Chain_table") == "").then(None)
                .otherwise(pl.col("Chain_table")).alias("Chain_table"))
            .drop("_g", *[c for c in p.columns if c.startswith("_k_")]))


def finalize_chain(panel: pl.DataFrame) -> pl.DataFrame:
    """Chain, decided on the whole panel, in this order:

    1. the table's `chain` where an actors.csv row set one;
    2. the banner, in its most frequent spelling across the panel (ties:
       alphabetical) -- "Denn´s Biomarkt" and "Denn's Biomarkt" become one;
    3. no banner but a Gruppe: "<Gruppe> (other)" -- an Edeka merchant trading
       under their own name is Edeka's, not a chain of their own;
    4. neither: null. The store is unidentified at both levels, as it should be.
    """
    canon = (panel.filter(pl.col("Banner").is_not_null())
             .group_by(name_key(pl.col("Banner")).alias("_kb"), "Banner").len()
             .sort(["_kb", "len", "Banner"], descending=[False, True, False])
             .unique("_kb", keep="first", maintain_order=True)
             .select("_kb", pl.col("Banner").alias("_cb")))
    table = pl.col("Chain_table") if "Chain_table" in panel.columns else pl.lit(None, pl.Utf8)
    return (panel.with_columns(name_key(pl.col("Banner")).alias("_kb")).with_row_index("_r")
            .join(canon, on="_kb", how="left").sort("_r").drop("_r")
            .with_columns(pl.coalesce(
                table, pl.col("_cb"),
                pl.when(pl.col("Gruppe").is_not_null())
                .then(pl.col("Gruppe") + pl.lit(" (other)"))).alias("Chain"))
            .drop("_kb", "_cb", *(["Chain_table"] if "Chain_table" in panel.columns else [])))


def audit_handelszentrale(df: pl.DataFrame) -> pl.DataFrame:
    """
    Every distinct (Handelszentrale, Name-juristisch) pair with its assigned
    Gruppe and store count — the table used to verify the Netto split,
    generalised to every chain in the panel.

    A Handelszentrale mapping to more than one Gruppe is expected for Netto
    (the legal entity decides, not the label). Anywhere else it usually
    signals an inconsistent fallback regex.
    """
    cols = [c for c in ("Handelszentrale", "Name-juristisch", "Gruppe") if c in df.columns]
    return (
        df.group_by(cols)
        .agg(pl.len().alias("n_stores"))
        .sort(["Handelszentrale", "n_stores"], descending=[False, True])
    )


def audit_gruppe(df: pl.DataFrame, top_n: int = 40) -> pl.DataFrame:
    """
    List Handelszentrale values that matched no rule at all, with store
    counts. These are the ones currently falling through to
    fallback_to_handelszentrale (raw Handelszentrale as its own group) or
    to null.

    Use this to decide which fall-through values deserve their own
    _HANDELSZENTRALE_TO_GRUPPE entry. Anything with a large count is a
    chain currently being treated as an unnamed singleton group.
    """
    matched = assign_gruppe(df, fallback_to_handelszentrale=False)
    return (
        matched.filter(pl.col("Gruppe").is_null())
        .group_by("Handelszentrale")
        .agg(pl.len().alias("n_stores"))
        .sort("n_stores", descending=True)
        .head(top_n)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _clean_sheet(
    df: pl.DataFrame,
    year: str,
    dedupe_strategy: str = "drop_all",
) -> tuple[pl.DataFrame, dict]:
    """
    Clean one year's sheet. Returns (cleaned_df, report).

    Step order matters and differs from the previous version:
      * normalize_nulls runs BEFORE parse_numerics, so "" and " 12,5 "
        become null / parseable rather than failing the cast.
      * drop_junk_rows runs BEFORE numeric work, so header artefacts do
        not contribute to outlier statistics.
      * There is no zero-filling step at all. Outlier nulls stay null.
    """
    n_in = len(df)

    # Order: schema/nulls/junk first (so numerics parse), then numerics and
    # their checks, then the derived columns Shop Type -> Gruppe (+Gruppe_src)
    # -> Chain, then deduplication. gate_supplied() runs later, in panel.py.
    df = normalize_schema(df)
    df = normalize_nulls(df)
    df, n_junk = drop_junk_rows(df)
    df = parse_numerics(df)
    df = enforce_dtypes(df)
    df = apply_revenue_scale(df, year)
    df, n_outliers = flag_revenue_outliers(df)
    df = reconcile_areas(df)
    df = derive_shop_type(df)
    df = assign_gruppe(df)
    df = assign_chain(df)
    df, n_dupes = drop_duplicates(df, strategy=dedupe_strategy)
    df = df.sort("TDLinx")

    total = "Umsatzpotential Brutto (Mio EUR)"
    report = {
        "year": year,
        "rows_in": n_in,
        "rows_out": len(df),
        "junk_rows_dropped": n_junk,
        "duplicate_rows_dropped": n_dupes,
        "revenue_outliers_nulled": n_outliers,
        "revenue_null_pct": round(
            df[total].null_count() / max(len(df), 1) * 100, 1
        ) if total in df.columns else None,
        "vkfl_null_pct": round(
            df["VKFL"].null_count() / max(len(df), 1) * 100, 1
        ) if "VKFL" in df.columns else None,
        "gruppe_null_pct": round(
            df["Gruppe"].null_count() / max(len(df), 1) * 100, 1
        ) if "Gruppe" in df.columns else None,
        "chain_null_pct": round(
            df["Chain"].null_count() / max(len(df), 1) * 100, 1
        ) if "Chain" in df.columns else None,
        "area_mismatches": int(df["area_mismatch"].sum()) if "area_mismatch" in df.columns else 0,
    }
    return df, report


def clean(
    sheets: dict[str, pl.DataFrame],
    dedupe_strategy: str = "drop_all",
    report: bool = False,
) -> dict[str, pl.DataFrame] | tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """
    Apply all cleaning steps to every sheet returned by load_data().

    Returns just the cleaned dict by default — same call signature as the
    original clean(sheets). Pass report=True to additionally get back a
    per-year diagnostic table (rows dropped, outliers nulled, null rates).
    Read it when you use it — a year with a revenue_null_pct far from its
    neighbours is a parsing or unit problem, not a data problem.

    Because the return type depends on `report`, `edeka = clean(edeka)`
    keeps working unchanged; only `edeka, rep = clean(edeka, report=True)`
    gets the diagnostics.
    """
    cleaned, reports = {}, []
    for year, df in sheets.items():
        cleaned[year], rep = _clean_sheet(df, year, dedupe_strategy=dedupe_strategy)
        reports.append(rep)

    if not report:
        return cleaned

    report_df = pl.DataFrame(reports).sort("year")
    return cleaned, report_df


# ══════════════════════════════════════════════════════════════════════════════
# Geocoding
# ══════════════════════════════════════════════════════════════════════════════

def build_full_address(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    """
    Add a `full_address` column from Straße / Plz / Ort / Ortsteil.

    Rows missing Straße, Plz or Ort get a null address rather than a
    partial one. With `ignore_nulls=True` a missing street silently
    produced "68159 Mannheim, Deutschland", which geocodes cleanly to the
    city centroid — a plausible-looking coordinate that is simply wrong,
    and indistinguishable downstream from a real match.
    """
    if "full_address" in df.columns:
        return df

    required = [c for c in ("Straße", "Plz", "Ort") if c in df.columns]
    complete = pl.all_horizontal([pl.col(c).is_not_null() for c in required])

    ortsteil = (
        pl.when(
            pl.col("Ortsteil").is_not_null()
            & (pl.col("Ortsteil").str.strip_chars() != "")
        )
        .then(pl.lit(", ") + pl.col("Ortsteil"))
        .otherwise(pl.lit(""))
        if "Ortsteil" in df.columns else pl.lit("")
    )

    address = pl.concat_str(
        [
            pl.col("Straße"), pl.lit(", "),
            pl.col("Plz").cast(pl.Utf8), pl.lit(" "),
            pl.col("Ort"),
            ortsteil,
            pl.lit(", Deutschland"),
        ],
        separator="",
        ignore_nulls=False,
    )

    out = df.with_columns(
        pl.when(complete).then(address).otherwise(None).alias("full_address")
    )
    if verbose:
        n_bad = out["full_address"].null_count()
        if n_bad:
            print(f"  build_full_address: {n_bad:,} rows lack Straße/Plz/Ort — not geocodable")
    return out


def _ensure_csv_header(path: str) -> None:
    if not os.path.exists(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("full_address,lat,lon\n")


def _append_rows(path: str, rows: list[tuple]) -> None:
    """Append many rows with a single file handle."""
    if not rows:
        return
    _ensure_csv_header(path)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        for address, lat, lon in rows:
            writer.writerow([
                address,
                "" if lat is None else lat,
                "" if lon is None else lon,
            ])


def read_coord_cache(csv_path: str = CSV_PATH) -> pl.DataFrame:
    """
    Read the geocoding cache once, deduplicated, with real Float64 coords.

    Two fixes over the previous inline version:
      * lat/lon are explicitly cast. `attach_coords_only` never called
        `cleanup_csv`, so an empty string in the cache came through as a
        Utf8 column: `is_not_null()` returned True for "" and the failure
        only surfaced later inside points_from_xy.
      * Deduplication uses group_by rather than sort + unique(keep="last").
        `unique` gives no ordering guarantee without maintain_order, so
        the "sort nulls first, keep last" trick could keep the null row.
    """
    if not os.path.exists(csv_path):
        return pl.DataFrame(
            schema={"full_address": pl.Utf8, "lat": pl.Float64, "lon": pl.Float64}
        )

    coords = pl.read_csv(
        csv_path,
        encoding="utf8-lossy",
        schema_overrides={"full_address": pl.Utf8, "lat": pl.Utf8, "lon": pl.Utf8},
        ignore_errors=True,
        truncate_ragged_lines=True,
    )
    return (
        coords
        .with_columns([
            pl.col("lat").str.strip_chars().cast(pl.Float64, strict=False),
            pl.col("lon").str.strip_chars().cast(pl.Float64, strict=False),
        ])
        .filter(pl.col("full_address").is_not_null())
        .group_by("full_address")
        .agg(
            pl.col("lat").drop_nulls().first().alias("lat"),
            pl.col("lon").drop_nulls().first().alias("lon"),
        )
    )


def cleanup_csv(csv_path: str = CSV_PATH, verbose: bool = True) -> pl.DataFrame:
    """Rewrite the cache deduplicated, keeping the resolved result per address."""
    before = 0
    if os.path.exists(csv_path):
        before = pl.read_csv(
            csv_path, encoding="utf8-lossy", ignore_errors=True,
            truncate_ragged_lines=True,
        ).height
    coords = read_coord_cache(csv_path)
    coords.write_csv(csv_path)
    if verbose:
        print(
            f"CSV cleanup: {before:,} → {len(coords):,} rows "
            f"({coords['lat'].null_count():,} still unresolved)"
        )
    return coords


def load_done_addresses(csv_path: str = CSV_PATH) -> set[str]:
    """Addresses that already have real (non-null) coordinates."""
    coords = read_coord_cache(csv_path)
    return set(
        coords.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
        ["full_address"].to_list()
    )


def attach_coords(
    df: pl.DataFrame,
    coords: pl.DataFrame | str = CSV_PATH,
) -> pl.DataFrame:
    """
    Left-join cached coordinates onto df by full_address.

    Accepts either a path or an already-loaded cache frame. Pass the frame
    when looping over sheets — the old signature forced a full CSV re-read
    and re-parse per sheet per phase (~36 reads for a 12-year panel).
    """
    if isinstance(coords, (str, os.PathLike)):
        coords = read_coord_cache(str(coords))
    df = df.drop([c for c in ("lat", "lon") if c in df.columns])
    return df.join(coords, on="full_address", how="left")


def attach_coords_only(
    sheets: dict[str, pl.DataFrame], csv_path: str = CSV_PATH
) -> dict[str, pl.DataFrame]:
    """Attach cached coordinates without geocoding anything new."""
    coords = read_coord_cache(csv_path)
    return {
        year: attach_coords(build_full_address(df, verbose=False), coords)
        for year, df in sheets.items()
    }


def geocode_batch_photon(
    addresses: list[str],
    csv_path: str = CSV_PATH,
    batch_size: int = 100,
    verbose: bool = True,
) -> None:
    """
    Phase 1 — Photon batch geocoding.

    Each batch is appended in one write so progress survives a kernel
    restart. Failures are written with empty coordinates, which marks them
    as attempted-but-unresolved; `load_done_addresses` still excludes them,
    so Phase 2 (Nominatim) picks them up.
    """
    import time
    from geopandas.tools import geocode as gpd_geocode

    _ensure_csv_header(csv_path)

    for i in range(0, len(addresses), batch_size):
        batch = addresses[i: i + batch_size]
        batch_num = i // batch_size + 1
        rows = []
        try:
            gdf = gpd_geocode(batch, provider="photon", user_agent="edeka_geocoder")
            gdf = gdf.reset_index(drop=True)
            for idx, addr in enumerate(batch):
                pt = gdf.geometry.iloc[idx]
                if pt is not None and not pt.is_empty:
                    rows.append((addr, pt.y, pt.x))
                else:
                    rows.append((addr, None, None))
            if verbose:
                done = min(i + batch_size, len(addresses))
                print(f"Batch {batch_num} done — {done:,}/{len(addresses):,}")
        except Exception as e:
            print(f"Batch {batch_num} failed ({e}) — {len(batch)} addresses unresolved")
            rows = [(addr, None, None) for addr in batch]

        _append_rows(csv_path, rows)
        time.sleep(1)


def nominatim_retry(
    df_failed: pl.DataFrame, csv_path: str = CSV_PATH, verbose: bool = True
) -> None:
    """
    Phase 2 — Nominatim retry for addresses still missing after Photon.
    Falls back to a simplified address (Straße + PLZ + Ort) if the full
    address returns nothing.
    """
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter

    _ensure_csv_header(csv_path)
    done = load_done_addresses(csv_path)
    to_retry = df_failed.filter(~pl.col("full_address").is_in(list(done)))
    if verbose:
        print(f"Nominatim retry: {len(to_retry):,} address(es) remaining")

    geolocator = Nominatim(user_agent="edeka_geocoder_retry")
    geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    buffer, n_ok = [], 0
    for row in to_retry.iter_rows(named=True):
        address = row["full_address"]
        lat = lon = None
        try:
            loc = geocode_fn(address)
            if loc is None and all(row.get(c) for c in ("Straße", "Plz", "Ort")):
                loc = geocode_fn(f"{row['Straße']}, {row['Plz']} {row['Ort']}, Deutschland")
            if loc:
                lat, lon, n_ok = loc.latitude, loc.longitude, n_ok + 1
        except Exception as e:
            print(f"Error [{address[:50]}]: {e}")

        buffer.append((address, lat, lon))
        if len(buffer) >= 25:
            _append_rows(csv_path, buffer)
            buffer = []

    _append_rows(csv_path, buffer)
    if verbose:
        print(f"Nominatim resolved {n_ok:,} / {len(to_retry):,}")


def geocode_sheets(
    sheets: dict[str, pl.DataFrame],
    csv_path: str = CSV_PATH,
    batch_size: int = 100,
    verbose: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Full two-phase geocoding pipeline for all sheets.

    Phase 1 — Photon batch geocoding (fast, bulk).
    Phase 2 — Nominatim retry for any addresses still missing.
    """
    sheets = {year: build_full_address(df, verbose=verbose) for year, df in sheets.items()}

    all_addresses: set[str] = set()
    for df in sheets.values():
        all_addresses.update(df["full_address"].drop_nulls().unique().to_list())

    cleanup_csv(csv_path, verbose=verbose)
    done = load_done_addresses(csv_path)
    remaining = sorted(all_addresses - done)

    if verbose:
        print(f"Total unique addresses : {len(all_addresses):,}")
        print(f"Already geocoded       : {len(done):,}")
        print(f"Remaining (Phase 1)    : {len(remaining):,}")

    if remaining:
        geocode_batch_photon(remaining, csv_path, batch_size=batch_size, verbose=verbose)

    coords = read_coord_cache(csv_path)
    sheets = {year: attach_coords(df, coords) for year, df in sheets.items()}

    address_cols = ["Straße", "Plz", "Ort", "Ortsteil", "full_address"]
    failed = pl.concat(
        [
            df.filter(pl.col("lat").is_null(), pl.col("full_address").is_not_null())
              .select([c for c in address_cols if c in df.columns])
            for df in sheets.values()
        ],
        how="diagonal_relaxed",
    ).unique(subset=["full_address"])

    if verbose:
        print(f"Remaining after Phase 1 (Nominatim): {len(failed):,}")
    if len(failed) > 0:
        nominatim_retry(failed, csv_path, verbose=verbose)

    coords = read_coord_cache(csv_path)
    sheets = {year: attach_coords(df, coords) for year, df in sheets.items()}

    if verbose:
        total = sum(len(df) for df in sheets.values())
        geocoded = sum(df["lat"].drop_nulls().len() for df in sheets.values())
        print(f"Final: {geocoded:,} / {total:,} rows geocoded ({geocoded/total*100:.1f}%)")

    return sheets
