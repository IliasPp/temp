"""Population-weighted median across Kreise: helper, Kreis band table, figure."""
import polars as pl

import scripts.tables as T
import scripts.visualizations.figures as F
from scripts.isochrones.market_cube import wmedian


def test_wmedian_against_hand_calculation():
    # values 1000, 2000, 6000 with weights 1, 1, 8: half the weight (5) is reached
    # only at 6000; the plain median is 2000
    d = pl.DataFrame({"g": [1, 1, 1], "v": [6000.0, 1000.0, 2000.0], "w": [8.0, 1.0, 1.0]})
    r = d.group_by("g").agg(wmedian("v", "w").alias("m"), pl.col("v").median().alias("med"))
    assert r["m"][0] == 6000.0 and r["med"][0] == 2000.0
    # exactly half: the smallest value reaching it
    d2 = pl.DataFrame({"g": [1, 1], "v": [1000.0, 3000.0], "w": [1.0, 1.0]})
    assert d2.group_by("g").agg(wmedian("v", "w"))["v"][0] == 1000.0


def cube():
    rows = []
    for y in (2024, 2025):
        for r, h, pop in (("01001", 1500.0, 1e6), ("01002", 4000.0, 3e4), ("01003", 5000.0, 2e4)):
            for basis, hh in (("pop", h), ("vkfl", h + 100)):
                rows.append(dict(spec="super", year=y, level="KREIS", region=r, basis=basis,
                                 actor_level="Gruppe", hhi=hh, total=pop if basis == "pop" else 5e3))
    return pl.DataFrame(rows)


def test_kreis_bands_reports_both_medians():
    k = T.kreis_bands(cube(), "super", "Gruppe", "vkfl")
    row = k.filter(pl.col("year") == 2025)
    assert row["Median HHI"][0] == 4100.0                  # the median Kreis
    assert row["Median HHI, people"][0] == 1600.0          # the big Kreis holds most people
    assert row["Pop. in > 2,000"][0] < 0.05


def test_distribution_figure_draws_weighted_median():
    import matplotlib; matplotlib.use("Agg")
    ax = F.plot_distribution(cube(), spec="super", basis="vkfl", actor_level="Gruppe")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "population-weighted median" in labels
    ys = [l for l in ax.get_lines() if l.get_label() == "population-weighted median"][0].get_ydata()
    assert list(ys) == [1600.0, 1600.0]
    supply_only = cube().filter(pl.col("basis") == "vkfl")
    ax2 = F.plot_distribution(supply_only, spec="super", basis="vkfl", actor_level="Gruppe")
    assert ax2.get_legend() is None, "no population rows: plain boxes only"
