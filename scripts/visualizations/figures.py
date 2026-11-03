"""
Publication-ready figures for the market structure cube.

Every function takes the tidy `shares` / `hhi` frames from `market_cube`, or the
raw grid, and returns a matplotlib Axes. `save()` writes PDF (vector, for the
paper) and PNG (300 dpi, for slides and the wiki) from one call, so no figure
exists in only one format.

Colour is categorical and fixed: the same Gruppe gets the same colour in every
figure of the run, otherwise a reader comparing two maps is comparing palettes.
The palette is Okabe-Ito extended -- distinguishable in greyscale print and
under the common colour-vision deficiencies.

Nothing here imports network_voronoi or market_metrics: figures are a pure
function of the tables, so they can be rebuilt without the solve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
           "#F0E442", "#7F3C8D", "#3B7C70", "#A5AA99", "#B07AA1", "#8C564B"]
GREY = "#D9D9D9"

BASIS_LABEL = {
    "pop": "catchment population",
    "area": "catchment area",
    "vkfl": "sales area (VKFL)",
    "vkfl_food": "sales area, food",
    "umsatz": "revenue potential",
    "umsatz_food": "revenue potential, food",
}
from scripts.runconfig import SPEC_LABEL          # noqa: E402  one definition

#: Fixed colours for the operators that recur across figures, so Edeka is the
#: same blue on every map. Matched case-insensitively; "aldi" also catches
#: "Aldi Nord"/"Aldi Süd". Each pair that can share a figure is distinguishable
#: (Lidl/Kaufland at chain level, Rossmann/dm/Müller in the drugstore spec), and
#: none is too light to read on white. Everything else takes the next FREE
#: colour (below) in the order it appears.
ACTOR_COLOURS = {
    "edeka": "#1F4E9C", "rewe": "#C8102E", "schwarz": "#E6A100", "lidl": "#E6A100",
    "kaufland": "#7F2704", "aldi": "#56B4E9", "penny": "#E7298A", "netto": "#009E73",
    "norma": "#6A3D9A", "globus": "#A6761D", "tegut": "#66A61E",
    "rossmann": "#D55E00", "dm": "#999933", "müller": "#882255",
}


def actor_colour(name: str) -> str | None:
    k = str(name).strip().casefold()
    return ACTOR_COLOURS.get(k) or (ACTOR_COLOURS["aldi"] if k.startswith("aldi") else None)


# ==========================================================================
# style / IO
# ==========================================================================

def set_style(base: float = 10.0) -> None:
    """House style. Call once per session, before any figure."""
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans Pro", "Arial", "DejaVu Sans"],
        "font.size": base, "axes.titlesize": base + 1, "axes.labelsize": base,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E8E8E8", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem: str | Path, *, formats: Sequence[str] = ("pdf", "png"),
         close: bool = True) -> list[Path]:
    """Write one figure to every format. Returns the paths written."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for f in formats:
        p = stem.with_suffix(f".{f}")
        fig.savefig(p, format=f)
        out.append(p)
    if close:
        plt.close(fig)
    return out


NEUTRAL = ("other", "unassigned")


#: Colours for operators without a fixed one. Chosen to stay clear of every
#: ACTOR_COLOURS entry and of the greys, so an unlisted operator can never pass
#: for Schwarz or Edeka. Exact-match exclusion against PALETTE was not enough:
#: PALETTE's orange and blue are near-copies of the fixed Schwarz and Edeka.
FREE = ["#FB9A99", "#CAB2D6", "#FFD92F", "#8DD3C7", "#BC80BD", "#D9B38C",
        "#B2DF8A", "#F781BF", "#E5C494"]


def colours(cats: Sequence[str], *, other: str = "other") -> dict[str, str]:
    """Stable category -> colour map. Known operators keep their fixed colour
    (ACTOR_COLOURS), residual categories go to grey, the rest take FREE colours
    in order of appearance."""
    c, i = {}, 0
    for k in cats:
        if k == other or k in NEUTRAL:
            c[k] = GREY
        elif k == "unidentified" or str(k).upper() == "UNKNOWN":
            c[k] = "#9E9E9E"
        elif (fixed := actor_colour(k)):
            c[k] = fixed
        else:
            c[k] = FREE[i % len(FREE)]
            i += 1
    return c


# ==========================================================================
# grid assignment maps  (Mannheim, Hamburg)
# ==========================================================================

def region_bbox(gdf, id_col: str, value: str, *, pad_km: float = 2.0,
                crs=3035) -> tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of one region in metres, padded."""
    sel = gdf[gdf[id_col].astype(str) == str(value)]
    if sel.empty:
        raise ValueError(f"region_bbox: no {value!r} in {id_col!r}")
    x0, y0, x1, y1 = sel.to_crs(crs).total_bounds
    p = pad_km * 1000
    return x0 - p, y0 - p, x1 + p, y1 + p


def centre_bbox(gdf, id_col: str, value: str, *, half_km: float = 3.0,
                crs=3035) -> tuple[float, float, float, float]:
    """Square window of side 2*half_km around a region's centroid."""
    sel = gdf[gdf[id_col].astype(str) == str(value)].to_crs(crs)
    if sel.empty:
        raise ValueError(f"centre_bbox: no {value!r} in {id_col!r}")
    c = sel.geometry.union_all().centroid if hasattr(sel.geometry, "union_all") \
        else sel.geometry.unary_union.centroid
    h = half_km * 1000
    return c.x - h, c.y - h, c.x + h, c.y + h


def grid_slice(
    cells: pl.DataFrame, mapping: pl.DataFrame, assignment: pl.DataFrame,
    stores: pl.DataFrame, *, bbox: tuple[float, float, float, float],
    by: str = "Gruppe",
) -> pl.DataFrame:
    """Original 100 m cells inside `bbox`, each carrying its winning store/actor.

    This is the grid the solve actually produced, pushed back down from solve
    nodes to the 100 m cells they stand for -- so a coarsened area shows as a
    block of identical squares rather than as missing data. That visible
    blockiness is the honest picture of the reduction.
    """
    # `cells` must already be in the bbox's CRS. In the pipeline they come from
    # run_pipeline.grid_view(), which has moved them to the raster CRS; passing
    # cells_raw directly (EPSG:3035 x/y) with a raster-CRS bbox selects nothing.
    x0, y0, x1, y1 = bbox
    sub = cells.filter(pl.col("x").is_between(x0, x1) & pl.col("y").is_between(y0, y1))
    if sub.is_empty():
        raise ValueError("grid_slice: no cells in bbox -- the bbox and the cells "
                         "must be in the same CRS (the raster's; pass crs= from "
                         "run_pipeline.grid_crs)")
    a = assignment.filter(pl.col("rank") == 1).select(
        pl.col("cell_id").alias("_node"), pl.col("store_id").cast(pl.Utf8))
    st = stores.select(pl.col("store_id").cast(pl.Utf8), by)
    return (sub.select("cell_id", "row", "col", "x", "y", "pop")
            .join(mapping.rename({"src": "cell_id", "dst": "_node"}), on="cell_id",
                  how="left")
            .join(a, on="_node", how="left")
            .join(st, on="store_id", how="left")
            # no winner and an unnamed winner are different facts: the first is
            # a cell no store reached, the second a store the panel cannot place
            .with_columns(pl.when(pl.col("store_id").is_null()).then(pl.lit("unassigned"))
                          .otherwise(pl.col(by).fill_null("unidentified")).alias("actor")))


def plot_grid(
    g: pl.DataFrame, *, res_m: float = 100.0, top: int = 8, stores=None,
    boundaries=None, title: str = "", ax=None, edges: bool = True,
    scale_km: float | None = 1.0, legend: bool = True,
    cell_grid: bool | None = None, crs=3035,
):
    """Draw the assignment grid as squares: one 100 m raster cell, one square.

    `edges=True` outlines the boundary between *stores* (not actors), which is
    the catchment boundary the solve produced -- two adjacent Edeka stores show
    a seam, and that seam is real.
    """
    rows, cols = g["row"].to_numpy(), g["col"].to_numpy()
    r0, c0 = rows.min(), cols.min()
    H, W = rows.max() - r0 + 1, cols.max() - c0 + 1

    counts = (g.filter(pl.col("actor") != "unassigned")
              .group_by("actor").agg(pl.len().alias("n")).sort("n", descending=True))
    keep = counts["actor"].to_list()[:top]
    cat = keep + (["other"] if counts.height > top else [])
    if (g["actor"] == "unassigned").any():
        cat = cat + ["unassigned"]
    cmap = colours(cat)
    idx = {k: i for i, k in enumerate(cat)}

    default = idx.get("other", idx.get("unassigned", 0))
    act = np.array([idx.get(a, default) for a in g["actor"].to_list()], dtype=np.int16)
    arr = np.full((H, W), -1, dtype=np.int16)
    arr[rows - r0, cols - c0] = act

    x, y = g["x"].to_numpy(), g["y"].to_numpy()
    extent = (x.min() - res_m / 2, x.max() + res_m / 2,
              y.min() - res_m / 2, y.max() + res_m / 2)
    cm = ListedColormap(["#FFFFFF"] + [cmap[k] for k in cat])   # -1 = no raster cell
    norm = BoundaryNorm(np.arange(-1.5, len(cat) + 0.5), cm.N)

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.imshow(arr, cmap=cm, norm=norm, extent=extent, origin="upper",
              interpolation="nearest")

    if edges:
        sid = np.full((H, W), -1, dtype=np.int64)
        ids = [s for s in g["store_id"].unique().to_list() if s is not None]
        codes = {s: i for i, s in enumerate(ids)}
        sid[rows - r0, cols - c0] = [codes.get(s, -1) for s in g["store_id"].to_list()]
        segs = []
        vy = np.nonzero((sid[:, 1:] != sid[:, :-1]) & (sid[:, 1:] >= 0) & (sid[:, :-1] >= 0))
        for r, c in zip(*vy):
            X = extent[0] + (c + 1) * res_m
            Y = extent[3] - r * res_m
            segs.append([(X, Y), (X, Y - res_m)])
        hz = np.nonzero((sid[1:] != sid[:-1]) & (sid[1:] >= 0) & (sid[:-1] >= 0))
        for r, c in zip(*hz):
            X = extent[0] + c * res_m
            Y = extent[3] - (r + 1) * res_m
            segs.append([(X, Y), (X + res_m, Y)])
        ax.add_collection(LineCollection(segs, colors="white", linewidths=0.35,
                                         alpha=0.9))

    if stores is not None and stores.height:
        sx = "x_orig" if "x_orig" in stores.columns else "x"
        sy = "y_orig" if "y_orig" in stores.columns else "y"
        s = stores.filter(pl.col(sx).is_between(extent[0], extent[1])
                          & pl.col(sy).is_between(extent[2], extent[3]))
        if s.height:
            ax.scatter(s[sx], s[sy], s=26, facecolor="white", edgecolor="black",
                       linewidth=0.8, zorder=4)

    if boundaries is not None:
        boundaries.to_crs(crs).boundary.plot(ax=ax, color="black", linewidth=0.9,
                                              alpha=0.8, zorder=3)

    if cell_grid is None:                 # on a tight zoom the squares are the point
        cell_grid = (extent[1] - extent[0]) <= 4_000
    if cell_grid:
        ax.set_xticks(np.arange(extent[0], extent[1] + res_m, res_m), minor=True)
        ax.set_yticks(np.arange(extent[2], extent[3] + res_m, res_m), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.6)
        ax.tick_params(which="minor", length=0)

    if scale_km:
        L = scale_km * 1000
        x0 = extent[0] + 0.06 * (extent[1] - extent[0])
        yb = extent[2] + 0.05 * (extent[3] - extent[2])
        ax.plot([x0, x0 + L], [yb, yb], color="black", lw=2.5, zorder=5)
        ax.text(x0 + L / 2, yb + 0.012 * (extent[3] - extent[2]),
                f"{scale_km:g} km", ha="center", va="bottom", fontsize=8, zorder=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_frame_on(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.grid(which="major", visible=False)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(handles=[Line2D([], [], marker="s", ls="", markersize=7,
                                  markerfacecolor=cmap[k], markeredgecolor="none",
                                  label=k) for k in cat],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  title="assigned to")
    return ax


def catchment_polygons(g: pl.DataFrame, *, crs, res_m: float = 100.0,
                       by: str | None = "actor", simplify_m: float | None = None):
    """Vectorise an assignment slice into one polygon per store.

    `g` is `grid_slice()` output. The catchment is already there -- it is the set
    of 100 m cells a store wins -- so this only draws the outline around each
    set; no re-solve, no OSRM, nothing recomputed.

    `area_km2` is counted from cells (exact, and identical to the cube's `area`
    basis) rather than measured off the polygon, so the two can never disagree.
    Cells with no winner are dropped rather than becoming holes in someone's
    catchment.
    """
    import geopandas as gpd
    from rasterio.features import shapes
    from rasterio.transform import from_origin

    g = g.filter(pl.col("store_id").is_not_null())
    if g.is_empty():
        raise ValueError("catchment_polygons: nothing assigned in this slice")
    rows, cols = g["row"].to_numpy(), g["col"].to_numpy()
    r0, c0 = rows.min(), cols.min()
    H, W = rows.max() - r0 + 1, cols.max() - c0 + 1

    ids = g["store_id"].unique().to_list()
    code = {s: i for i, s in enumerate(ids)}
    arr = np.full((H, W), -1, dtype=np.int32)
    arr[rows - r0, cols - c0] = [code[s] for s in g["store_id"].to_list()]

    x, y = g["x"].to_numpy(), g["y"].to_numpy()
    tf = from_origin(x.min() - res_m / 2, y.max() + res_m / 2, res_m, res_m)
    geoms, codes = [], []
    for geom, val in shapes(arr, mask=arr >= 0, transform=tf, connectivity=4):
        geoms.append(geom)
        codes.append(int(val))

    from shapely.geometry import shape
    gdf = gpd.GeoDataFrame({"store_id": [ids[c] for c in codes]},
                           geometry=[shape(gm) for gm in geoms], crs=crs)
    gdf = gdf.dissolve(by="store_id").reset_index()
    if simplify_m:
        gdf["geometry"] = gdf.geometry.simplify(simplify_m)

    aggs = [pl.len().alias("n_cells")]
    if by and by in g.columns:
        aggs.append(pl.col(by).first().alias(by))
    cnt = {r["store_id"]: r for r in g.group_by("store_id").agg(aggs).iter_rows(named=True)}
    # plain dicts, not a pandas merge: polars -> pandas needs pyarrow, which is
    # one more thing to install on a machine that does not otherwise need it
    sid = list(gdf["store_id"])
    gdf["n_cells"] = [cnt[x]["n_cells"] for x in sid]
    gdf["area_km2"] = [cnt[x]["n_cells"] * (res_m / 1000) ** 2 for x in sid]
    if by and by in g.columns:
        gdf[by] = [cnt[x][by] for x in sid]
    return gdf


def plot_catchments(polys, *, by: str = "actor", region=None, top: int = 8,
                    stores=None, title: str = "", ax=None, crs=None,
                    labels: bool = False, legend: bool = True):
    """Catchment polygons, coloured by operator.

    The polygon view and the cell view (`plot_grid`) show the same solve at two
    zooms: use this for a whole Kreis, where 100 m squares would be invisible,
    and the grid for a few kilometres, where the squares are the point.
    """
    import matplotlib.pyplot as _plt
    crs = crs or polys.crs
    lab = polys[by].fillna("unassigned") if by in polys.columns else None
    if lab is None:
        raise ValueError(f"plot_catchments: {by!r} not in the polygons")
    counts = lab.value_counts()
    keep = list(counts.index[:top])
    cats = keep + (["other"] if len(counts) > top else [])
    cmap = colours(cats)
    g = polys.copy()
    g["_c"] = [x if x in keep else "other" for x in lab]

    if ax is None:
        _, ax = _plt.subplots(figsize=(8.5, 8.5))
    if region is not None:
        region.to_crs(crs).boundary.plot(ax=ax, color="black", linewidth=1.2, alpha=0.8,
                                         zorder=3)
    g.plot(ax=ax, color=[cmap[c] for c in g["_c"]], edgecolor="white", linewidth=0.35,
           alpha=0.9)
    if stores is not None and len(stores):
        sx = "x_orig" if "x_orig" in stores.columns else "x"
        sy = "y_orig" if "y_orig" in stores.columns else "y"
        b = g.total_bounds
        st = stores.filter(pl.col(sx).is_between(b[0], b[2])
                           & pl.col(sy).is_between(b[1], b[3]))
        if st.height:
            ax.scatter(st[sx], st[sy], s=18, color="black", marker="x", linewidths=0.9,
                       zorder=5)
    if labels and "area_km2" in g.columns:
        for _, r in g.iterrows():
            c = r.geometry.representative_point()
            ax.annotate(f"{r['area_km2']:.0f}", (c.x, c.y), fontsize=6, ha="center",
                        va="center")
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(handles=[Line2D([], [], marker="s", ls="", markersize=8,
                                  markerfacecolor=cmap[c], markeredgecolor="none",
                                  label=c) for c in cats],
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  title=by)
    return ax


# ==========================================================================
# cube figures
# ==========================================================================

def _slice(df: pl.DataFrame, **flt) -> pl.DataFrame:
    for k, v in flt.items():
        df = df.filter(pl.col(k).is_in(v) if isinstance(v, (list, tuple)) else pl.col(k) == v)
    return df


#: Colour-scale cap for HHI maps. Above roughly this level a choropleth stops
#: informing: a handful of rural Kreise run to 8,000-10,000 and stretch the ramp
#: so that the whole populated country reads as one pale shade. Capping fixes the
#: ramp across maps too, so years and specs are comparable. Pass cap=None for
#: the raw range, or cap=<value> for another level.
HHI_CAP = 2500.0


def plot_choropleth(hhi: pl.DataFrame, gdf, *, value_col: str = "hhi",
                    id_col: str = "AGS5", cmap: str = "OrRd", title: str = "",
                    vmin=None, vmax=None, cap: float | None = HHI_CAP,
                    ax=None, label: str | None = None, **flt):
    """Kreis or Bundesland choropleth of any cube column. Pass the slice as kwargs:
    plot_choropleth(hhi, kreis_gdf, spec='super_disc', year=2025, basis='pop',
                    actor_level='Gruppe', level='KREIS').

    HHI columns are capped at `cap` (default HHI_CAP = 2,500): regions above it
    all take the top colour and the colour bar is marked with a "≥". The values
    themselves are untouched -- this is the colour scale, not the data. An
    explicit `vmax` wins, and `cap=None` turns it off. Other columns are not
    capped unless `vmax` says so.
    """
    d = _slice(hhi, **flt)
    if d.is_empty():
        raise ValueError(f"plot_choropleth: empty slice {flt}")
    vals = dict(zip(d["region"].to_list(), d[value_col].to_list()))
    g = gdf.copy()
    g[value_col] = [vals.get(str(v)) for v in g[id_col].astype(str)]
    capped = (vmax is None and cap is not None and value_col.startswith("hhi"))
    if capped:
        vmax = float(cap)
        if vmin is None:
            # a slice lying entirely above the cap (a small Land, all Kreise
            # concentrated) would otherwise get vmin > vmax and fail to draw
            lo = g[value_col].min()
            vmin = min(float(lo), vmax) if lo == lo and lo is not None else None
    over = int((g[value_col] > vmax).sum()) if vmax is not None else 0
    lab = label or value_col.upper()
    if capped:
        lab = f"{lab} (≥ {vmax:,.0f})"
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 7.6))
    g.plot(column=value_col, cmap=cmap, ax=ax, legend=True, edgecolor="white",
           linewidth=0.25, vmin=vmin, vmax=vmax,
           missing_kwds={"color": GREY, "label": "no data"},
           legend_kwds={"shrink": 0.55, "label": lab,
                        **({"extend": "max"} if capped and over else {})})
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(title)
    if capped and over:
        print(f"  plot_choropleth: {over} region(s) above the {vmax:,.0f} cap, "
              f"drawn in the top colour")
    return ax


def plot_share_map(shares: pl.DataFrame, gdf, *, actor: str | None = None,
                   id_col: str = "AGS5", cmap: str = "PuBu", title: str = "",
                   vmin=None, vmax=None, ax=None, **flt):
    """Market share as a plain percentage, by region.

    With `actor` it maps that operator's share; without it, the share of whoever
    leads each region (CR1). The leader map is the more honest summary of a
    concentrated market -- a single national colour scale for one chain says as
    much about where that chain trades as about concentration.
    """
    d = _slice(shares, **flt).filter(pl.col("region") != "UNKNOWN")
    d = d.filter(pl.col("actor") == actor) if actor else d.filter(pl.col("rank") == 1)
    if d.is_empty():
        raise ValueError(f"plot_share_map: empty slice {flt} actor={actor!r}")
    vals = dict(zip(d["region"].to_list(), d["share"].to_list()))
    g = gdf.copy()
    g["share"] = [vals.get(str(v)) for v in g[id_col].astype(str)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 7.6))
    g.plot(column="share", cmap=cmap, ax=ax, legend=True, edgecolor="white",
           linewidth=0.25, vmin=vmin, vmax=vmax,
           missing_kwds={"color": GREY, "label": "no data"},
           legend_kwds={"shrink": 0.55,
                        "label": f"{actor} share" if actor else "share of the largest"})
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(title)
    return ax


def plot_leader_map(shares: pl.DataFrame, gdf, *, id_col: str = "AGS5", top: int = 7,
                    title: str = "", ax=None, **flt):
    """Who leads each region. Categorical, not a scale.

    Answers a question an HHI map cannot: concentration of similar height can be
    Edeka country or Schwarz country, and the boundary between them is a fact
    about the German market worth seeing.
    """
    d = (_slice(shares, **flt).filter(pl.col("region") != "UNKNOWN",
                                      pl.col("rank") == 1))
    if d.is_empty():
        raise ValueError(f"plot_leader_map: empty slice {flt}")
    counts = d.group_by("actor").agg(pl.len().alias("n")).sort("n", descending=True)
    keep = counts["actor"].to_list()[:top]
    lead = {r: (a if a in keep else "other")
            for r, a in zip(d["region"].to_list(), d["actor"].to_list())}
    cats = keep + (["other"] if counts.height > top else [])
    cmap = colours(cats)
    g = gdf.copy()
    g["_lead"] = [lead.get(str(v)) for v in g[id_col].astype(str)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 7.6))
    blank = g[g["_lead"].isna()]
    if len(blank):
        blank.plot(ax=ax, color=GREY, edgecolor="white", linewidth=0.25)
    for k in cats:
        sub = g[g["_lead"] == k]
        if len(sub):
            sub.plot(ax=ax, color=cmap[k], edgecolor="white", linewidth=0.25)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", markersize=8,
                              markerfacecolor=cmap[k], markeredgecolor="none", label=k)
                       for k in cats],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="leader")
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(title)
    return ax


def plot_share_heatmap(shares: pl.DataFrame, *, top: int = 10, region: str = "DE",
                       level: str = "DE", annotate: bool = True, cmap: str = "PuBu",
                       title: str = "", ax=None, **flt):
    """Actor x year grid of market shares.

    The plain-percentage view of the same numbers the trend lines show: easier to
    read off a single cell, and it makes a level shift between two vintages
    visible as a seam down the column.

    Rows are the actors with the largest share in ANY year, ordered by it, so a
    chain that exited mid-panel (Real, Kaiser's Tengelmann) keeps its row and
    its empty cells show when it left. Unidentified stores are not a firm: they
    get the bottom row, labelled "unidentified", whatever their size.
    """
    d = _slice(shares, level=level, region=region, **flt)
    if d.is_empty():
        raise ValueError(f"plot_share_heatmap: empty slice {flt}")
    # Only the literal label UNKNOWN is treated as unidentified here. Other
    # labels in runconfig.UNIDENTIFIED ("Sonstige", "Other", ...) are fringe in
    # the cube (rank null) but appear as ordinary rows in this figure.
    d = d.with_columns(pl.when(pl.col("actor").str.to_uppercase() == "UNKNOWN")
                       .then(pl.lit("unidentified")).otherwise(pl.col("actor")).alias("actor"))
    years = sorted(d["year"].unique().to_list())
    order = (d.filter(pl.col("actor") != "unidentified").group_by("actor")
             .agg(pl.col("share").max()).sort(["share", "actor"], descending=[True, False])
             ["actor"].to_list()[:top])
    if d.filter(pl.col("actor") == "unidentified").height:
        order = order + ["unidentified"]
    M = np.full((len(order), len(years)), np.nan)
    look = {(r["actor"], r["year"]): r["share"] for r in d.iter_rows(named=True)}
    for i, a in enumerate(order):
        for j, y in enumerate(years):
            v = look.get((a, y))
            if v is not None:
                M[i, j] = v
    if ax is None:
        _, ax = plt.subplots(figsize=(1.0 + 0.62 * len(years), 1.2 + 0.42 * len(order)))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0,
                   vmax=float(np.nanmax(M)) if np.isfinite(M).any() else 1)
    ax.set_xticks(range(len(years)), years)
    ax.set_yticks(range(len(order)), order)
    ax.grid(False)
    if annotate:
        hi = np.nanmax(M) if np.isfinite(M).any() else 1
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if M[i, j] > 0.6 * hi else "#222")
    cb = ax.get_figure().colorbar(im, ax=ax, shrink=0.8)
    cb.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    cb.set_label("market share")
    if title:
        ax.set_title(title)
    return ax


def plot_trend(hhi: pl.DataFrame, *, value_col: str = "hhi", by: str = "spec",
               region: str = "DE", level: str = "DE", title: str = "",
               ax=None, **flt):
    """National (or one region's) concentration over the panel, one line per `by`."""
    d = _slice(hhi, level=level, region=region, **flt).sort("year")
    if d.is_empty():
        raise ValueError(f"plot_trend: empty slice {flt}")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    for i, (k, grp) in enumerate(sorted(d.group_by(by), key=lambda t: str(t[0]))):
        k = k[0] if isinstance(k, tuple) else k
        grp = grp.sort("year")
        ax.plot(grp["year"], grp[value_col], marker="o", ms=3.5, lw=1.6,
                color=PALETTE[i % len(PALETTE)],
                label=SPEC_LABEL.get(k, BASIS_LABEL.get(k, str(k))))
    ax.set_xlabel("")
    ax.set_ylabel("HHI (0-10,000)" if value_col == "hhi" else value_col)
    if value_col == "hhi":
        for y, lab in ((1500, "moderate"), (2500, "high")):
            if ax.get_ylim()[1] > y > ax.get_ylim()[0]:
                ax.axhline(y, color="#999", lw=0.8, ls=":")
                ax.text(ax.get_xlim()[1], y, f" {lab}", va="center", fontsize=7,
                        color="#777")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    return ax


def plot_share_trend(shares: pl.DataFrame, *, top: int = 6, stacked: bool = False,
                     title: str = "", ax=None, **flt):
    """Actor shares over the panel.

    Actors are chosen by their LARGEST share in any year, not by the last year:
    a chain that exits mid-panel (Real, Kaiser's Tengelmann) would otherwise be
    folded into "other" for every year, and "other" would appear to shrink as
    the buyers grow -- consolidation drawn as a trend of the residual.
    Unidentified stores are drawn as their own line, never inside "other", so a
    change in how well the panel identifies operators shows as what it is.
    """
    d = _slice(shares, **flt)
    if d.is_empty():
        raise ValueError(f"plot_share_trend: empty slice {flt}")
    # as in plot_share_heatmap: only the label UNKNOWN becomes "unidentified";
    # an actor literally called "Other" would merge into the "other" line
    unk = d["actor"].str.to_uppercase() == "UNKNOWN"
    keep = (d.filter(~unk).group_by("actor").agg(pl.col("share").max())
            .sort(["share", "actor"], descending=[True, False])
            .head(top)["actor"].to_list())
    lab = (pl.when(pl.col("actor").str.to_uppercase() == "UNKNOWN").then(pl.lit("unidentified"))
           .when(pl.col("actor").is_in(keep)).then(pl.col("actor"))
           .otherwise(pl.lit("other")))
    d = (d.with_columns(lab.alias("a"))
         .group_by("year", "a").agg(pl.col("share").sum()).sort("year"))
    if d.filter(pl.col("a") == "unidentified").height:
        keep = keep + ["unidentified"]
    cmap = colours(keep + ["other"])
    years = sorted(d["year"].unique().to_list())
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.2))
    if stacked:
        Y = [[d.filter((pl.col("year") == y) & (pl.col("a") == k))["share"].sum()
              for y in years] for k in keep + ["other"]]
        ax.stackplot(years, *Y, labels=keep + ["other"],
                     colors=[cmap[k] for k in keep + ["other"]], alpha=0.95)
    else:
        for k in keep + ["other"]:
            g = d.filter(pl.col("a") == k).sort("year")
            ax.plot(g["year"], g["share"], marker="o", ms=3.5, lw=1.6,
                    color=cmap[k], label=k)
    ax.set_ylabel("market share")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    if title:
        ax.set_title(title)
    return ax


def plot_distribution(hhi: pl.DataFrame, *, value_col: str = "hhi",
                      level: str = "KREIS", weight_basis: str | None = "pop",
                      title: str = "", ax=None, **flt):
    """Distribution of a Kreis-level metric across years -- boxplot per year.

    The national line hides this: an unchanged national HHI is compatible with
    half the Kreise concentrating and half loosening.

    Boxes count every Kreis once (the median Kreis). The diamonds are the
    population-weighted median (the median person's Kreis), weighting each Kreis
    by its catchment population: the `total` of the `weight_basis` rows of the
    same slice. Without such rows (a supply-side-only run) they are left out.
    """
    from scripts.isochrones.market_cube import wmedian
    d = _slice(hhi, level=level, **flt).filter(pl.col("region") != "UNKNOWN")
    if d.is_empty():
        raise ValueError(f"plot_distribution: empty slice {flt}")
    years = sorted(d["year"].unique().to_list())
    data = [d.filter(pl.col("year") == y)[value_col].drop_nulls().to_numpy()
            for y in years]
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.2))
    bp = ax.boxplot(data, positions=range(len(years)), widths=0.62, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.4))
    for p in bp["boxes"]:
        p.set(facecolor=PALETTE[0], alpha=0.45, edgecolor=PALETTE[0])
    if weight_basis:
        wf = {k: v for k, v in flt.items() if k != "basis"}
        w = (_slice(hhi, level=level, basis=weight_basis, **wf)
             .filter(pl.col("region") != "UNKNOWN")
             .select("year", "region", pl.col("total").alias("_w")))
        wm = (d.join(w, on=["year", "region"], how="inner")
              .group_by("year").agg(wmedian(value_col, "_w").alias("m")))
        pos = {y: i for i, y in enumerate(years)}
        if wm.height:
            ax.plot([pos[y] for y in wm["year"]], wm["m"], "D", color="black", ms=4.5,
                    zorder=4, label="population-weighted median")
            ax.plot([], [], color="black", lw=1.4, label="median Kreis (box: middle 50 %)")
            ax.legend(loc="best", fontsize="small")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=0)
    ax.set_ylabel("HHI (0-10,000)" if value_col == "hhi" else value_col)
    if title:
        ax.set_title(title)
    return ax


def plot_basis_agreement(hhi: pl.DataFrame, *, x_basis: str = "pop",
                         y_basis: str = "vkfl", level: str = "KREIS",
                         title: str = "", ax=None, **flt):
    """Same regions, two share bases. Points off the 45-degree line are the ones
    where the demand-side and supply-side readings disagree -- typically Kreise
    whose catchments cross the border, and the first thing a referee asks about.
    """
    d = _slice(hhi, level=level, **flt).filter(pl.col("region") != "UNKNOWN")
    x = d.filter(pl.col("basis") == x_basis).select("region", "hhi", "total")
    y = d.filter(pl.col("basis") == y_basis).select("region", pl.col("hhi").alias("hhi_y"))
    j = x.join(y, on="region", how="inner")
    if j.is_empty():
        raise ValueError(f"plot_basis_agreement: no overlap for {x_basis}/{y_basis}")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.4, 5.2))
    s = 6 + 40 * (j["total"].to_numpy() / max(j["total"].max(), 1)) ** 0.5
    ax.scatter(j["hhi"], j["hhi_y"], s=s, color=PALETTE[0], alpha=0.55,
               linewidths=0)
    lim = [0, max(j["hhi"].max(), j["hhi_y"].max()) * 1.05]
    ax.plot(lim, lim, color="#666", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"HHI, {BASIS_LABEL.get(x_basis, x_basis)}")
    ax.set_ylabel(f"HHI, {BASIS_LABEL.get(y_basis, y_basis)}")
    r = np.corrcoef(j["hhi"].to_numpy(), j["hhi_y"].to_numpy())[0, 1]
    ax.annotate(f"r = {r:.2f}   n = {j.height}", (0.04, 0.94), xycoords="axes fraction",
                fontsize=8, color="#555")
    if title:
        ax.set_title(title)
    return ax


def plot_region_bars(hhi: pl.DataFrame, *, level: str = "BL", years: Sequence[int] | None = None,
                     names: Mapping[str, str] | None = None, value_col: str = "hhi",
                     title: str = "", ax=None, **flt):
    """Regions ranked, first vs last year -- a dumbbell when two years are given."""
    d = _slice(hhi, level=level, **flt).filter(pl.col("region") != "UNKNOWN")
    years = years or [d["year"].min(), d["year"].max()]
    a = d.filter(pl.col("year") == years[0]).select("region", pl.col(value_col).alias("a"))
    b = d.filter(pl.col("year") == years[-1]).select("region", pl.col(value_col).alias("b"))
    j = a.join(b, on="region", how="inner").sort("b")
    lab = [names.get(r, r) if names else r for r in j["region"].to_list()]
    ypos = np.arange(j.height)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 0.28 * j.height + 1.4))
    ax.hlines(ypos, j["a"], j["b"], color="#BBB", lw=1.6, zorder=1)
    ax.scatter(j["a"], ypos, s=28, color=GREY, edgecolor="#888", zorder=2,
               label=str(years[0]))
    ax.scatter(j["b"], ypos, s=28, color=PALETTE[0], zorder=3, label=str(years[-1]))
    ax.set_yticks(ypos)
    ax.set_yticklabels(lab)
    ax.set_xlabel("HHI (0-10,000)" if value_col == "hhi" else value_col)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    return ax