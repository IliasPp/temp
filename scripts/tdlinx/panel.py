"""
panel.py — Panel data construction from the cleaned per-year sheets.

Usage
-----
from edeka_project.panel import build_panel

panel = build_panel(edeka)
"""

import polars as pl
from .cleaning import (CANONICAL_COLS, apply_actor_table, finalize_chain, gate_supplied,
                       load_actor_table)


PANEL_COLS = CANONICAL_COLS + ["lat", "lon"]

def build_panel(sheets: dict[str, pl.DataFrame], *, actors: bool = True,
                gate: bool = True) -> pl.DataFrame:
    """Concatenate the cleaned sheets and settle the two actor identities.

    Steps that need the whole panel, in this order (docs/actors.md):
      1. apply_actor_table -- actors.csv overrides (needs `year`);
      2. gate_supplied     -- pre-2020 supplied independents lose their group;
      3. finalize_chain    -- one spelling per banner, "<Gruppe> (other)".
    `actors=False` / `gate=False` skip steps 1 / 2, for comparisons.
    """
    panel = pl.concat(
        [
            df.with_columns(pl.lit(int(year)).alias("year"))
            for year, df in sorted(sheets.items())
        ],
        how="diagonal_relaxed",
    )

    panel = apply_actor_table(panel, load_actor_table() if actors else pl.DataFrame())
    if gate:
        panel = gate_supplied(panel)
    panel = finalize_chain(panel)

    desired = ["TDLinx", "year"] + CANONICAL_COLS + ["actor_rule", "lat", "lon", "full_address"]
    # deduplicate preserving order, then filter to only existing columns
    col_order = list(dict.fromkeys(
        c for c in desired if c in panel.columns
    ))

    return panel.select(col_order).sort(["TDLinx", "year"])

def panel_summary(panel: pl.DataFrame) -> None:
    """Print basic panel diagnostics."""
    n_ids = panel["TDLinx"].n_unique()
    n_years = panel["year"].n_unique()
    years = sorted(panel["year"].unique().to_list())
    obs_per_id = panel.group_by("TDLinx").len()["len"]
    balanced = obs_per_id.n_unique() == 1

    print(f"Observations : {len(panel):,}")
    print(f"Unique TDLinx: {n_ids:,}")
    print(f"Years        : {years}")
    print(f"Balanced     : {balanced}")
    print(f"Obs per ID   : min={obs_per_id.min()}, "
          f"mean={obs_per_id.mean():.1f}, "
          f"max={obs_per_id.max()}")


def get_entrants(panel: pl.DataFrame, year: int) -> pl.DataFrame:
    """TDLinx values appearing for the first time in the given year."""
    return panel.filter(
        pl.col("year") == year,
        ~pl.col("TDLinx").is_in(
            panel.filter(pl.col("year") < year)["TDLinx"]
        ),
    )


def get_exits(panel: pl.DataFrame, year: int) -> pl.DataFrame:
    """TDLinx values present before the given year but absent from it onward."""
    return panel.filter(
        pl.col("year") < year,
        ~pl.col("TDLinx").is_in(
            panel.filter(pl.col("year") >= year)["TDLinx"]
        ),
    )["TDLinx"].unique()