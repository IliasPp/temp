"""
Market structure metrics from network-Voronoi catchments.

Consumes `network_voronoi.assign()` output plus a store frame carrying Gruppe,
and produces concentration measures, per-Gruppe shares, and maps.

Regions are passed as GeoDataFrames (kreis_gdf, bundesland_gdf from
load_geodata()), not paths.


ALLOCATION IS THE PIVOT
-----------------------
Everything reads from one table: (cell_id, store_id, share), summing to 1 per node.

    allocate(a2024, stores)                    -> node-level shares (winner-take-all)
      +-- expand_to_cells(...)                 -> 100 m cells x region x pop
            +-- market_shares() -> hhi()       concentration by Kreis
            +-- store_totals()                 catchment pop per store/Gruppe

    local_market(a_iso, stores)                concentration of the choice set
                                               inside each store's own catchment

This module is winner-take-all only: the rank-1 store takes the cell, which is
the literal network-Voronoi reading and assumes nothing about demand. Any
allocation that splits a cell across stores needs a demand model, and a demand
model needs its own parameters, its own diagnostics and its own defence. That
lives in `huff.py`, which takes exactly the same (cell_id, store_id, share)
contract and is therefore a drop-in for `allocate` when it is ready.

Keeping them apart is not tidiness. A Huff share is a modelling claim with a
free elasticity in it; a WTA share is an arithmetic consequence of the drive
times. They should not be swappable by accident, and results from the two should
never be reported without saying which is which.


SUPPLY-SIDE vs DEMAND-SIDE SHARES
---------------------------------
market_shares() reports each Gruppe's share of allocated POPULATION. That is a
deliberate choice, and it is not what competition practice usually means by
market share -- it treats every consumer as spending the same and, under WTA,
weights a 2,000 m2 Edeka and a 700 m2 Netto identically.

size_shares() gives the supply-side counterpart: each Gruppe's share of sales
area (VKFL) in the region. Report both. Where they agree the finding is robust
to the choice; where they diverge, the divergence is itself the result.


WHY REGIONS ARE LABELLED ON THE ORIGINAL GRID
---------------------------------------------
A coarsened/deduped solve node can straddle a Kreis border, so labelling nodes
would dump their whole population on one side. Shares are computed per node,
then pushed down to original 100 m cells, each carrying exactly one region and
its own population. Exact regardless of how hard the solve was coarsened.

Labels come from rasterizing polygons onto the GHSL grid and indexing by
(row, col) -- O(n) with a C inner loop. A 35M-point sjoin is not.


HHI is on the 0-10,000 scale (squared percentage shares), the convention in
German/EU competition practice. Group by Gruppe, not store_id: two Edeka stores
in one Kreis are one competitor, and outlet-level HHI understates concentration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

HHI_SCALE = 10_000.0


# ==========================================================================
# Allocation
# ==========================================================================

def allocate(
    assignment: pl.DataFrame, stores: pl.DataFrame | None = None, *, rule: str = "wta",
    **kw,
) -> pl.DataFrame:
    """Returns (cell_id, store_id, share); share sums to 1 per node.

    Winner-take-all: the rank-1 store takes the cell. No demand model, no free
    parameters -- the share is a direct consequence of the drive-time matrix.

    Probabilistic allocation (Huff) has moved to `huff.py`. It produces the same
    three-column contract, so downstream code is unchanged:

        import huff
        alloc = huff.allocate(a, stores, attract_col="VKFL", beta=2.0)
    """
    if rule != "wta":
        raise ValueError(
            f"allocate: rule={rule!r} is no longer handled here. This module is "
            f"winner-take-all only; probabilistic allocation lives in huff.py, "
            f"which returns the same (cell_id, store_id, share) frame:\n"
            f"    import huff; alloc = huff.allocate(assignment, stores, ...)\n"
            f"It is parked pending a defensible beta and a truncation diagnostic, "
            f"so do not report from it without reading its module docstring."
        )
    if kw:
        raise ValueError(f"allocate: winner-take-all takes no parameters; got {list(kw)}. "
                         f"These look like Huff arguments -- see huff.py.")
    return assignment.filter(pl.col("rank") == 1).select(
        "cell_id", "store_id", pl.lit(1.0).alias("share"))


# ==========================================================================
# Regions
# ==========================================================================

def rasterize_regions(
    regions, template: str | Path, cells: pl.DataFrame, *, id_col: str,
    name_col: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Label every original cell with its region. Returns (labels, lookup).

    `regions` is a GeoDataFrame (kreis_gdf / bundesland_gdf) or a path.
    `cells` is the original load_cells() frame (needs row/col).
    `template` is the raster load_cells() was built from.

    Cells outside every polygon (coastline mismatch, enclaves) get code 0.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    gdf = regions if hasattr(regions, "geometry") else gpd.read_file(regions)
    if id_col not in gdf.columns:
        raise ValueError(f"rasterize_regions: {id_col!r} not in {list(gdf.columns)}")

    with rasterio.open(template) as src:
        transform, shape, crs = src.transform, (src.height, src.width), src.crs
    gdf = gdf.to_crs(crs)

    cols = [id_col] + ([name_col] if name_col else [])
    lookup = pl.DataFrame(
        {c: [None if v is None else str(v) for v in gdf[c].tolist()] for c in cols}
    ).with_row_index("region_code", offset=1).with_columns(
        pl.col("region_code").cast(pl.Int32))

    grid = rasterize(zip(gdf.geometry, lookup["region_code"].to_list()),
                     out_shape=shape, transform=transform, fill=0, dtype="int32")
    code = grid[cells["row"].to_numpy(), cells["col"].to_numpy()]

    n_out = int((code == 0).sum())
    if n_out:
        pop_out = cells.filter(pl.Series(code == 0))["pop"].sum()
        print(f"rasterize_regions: {n_out:,} cell(s) outside all polygons "
              f"({pop_out / cells['pop'].sum():.2%} of pop) -> code 0")
    print(f"rasterize_regions: {lookup.height:,} regions")
    return cells.select("cell_id").with_columns(pl.Series("region_code", code)), lookup


def expand_to_cells(
    alloc: pl.DataFrame, cells: pl.DataFrame, mapping: pl.DataFrame, *,
    labels: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Push node shares down to original cells. Returns (cell_id, store_id,
    pop_alloc[, region_code]).

    `cells` is the original load_cells() frame; `mapping` is compose() of every
    reduction (src = original cell_id, dst = solve node). This undoes coarsening:
    a 1 km node's shares apply to each constituent 100 m cell weighted by its own
    population. Total population conserved exactly.
    """
    out = (mapping.rename({"src": "cell_id", "dst": "_node"})
           .join(cells.select("cell_id", "pop"), on="cell_id", how="inner")
           .join(alloc.rename({"cell_id": "_node"}), on="_node", how="inner")
           .with_columns((pl.col("pop").cast(pl.Float64)
                          * pl.col("share").cast(pl.Float64)).alias("pop_alloc"))
           .select("cell_id", "store_id", "pop_alloc"))
    if labels is not None:
        out = out.join(labels, on="cell_id", how="left").with_columns(
            pl.col("region_code").fill_null(0))
    return out


# ==========================================================================
# Concentration
# ==========================================================================

def market_shares(
    cell_alloc: pl.DataFrame, stores: pl.DataFrame, *, by: str = "Gruppe",
    region_col: str = "region_code",
) -> pl.DataFrame:
    """Share of each `by`-unit within each region. Returns region, by, value, share."""
    df = cell_alloc
    if by != "store_id":
        if by not in stores.columns:
            raise ValueError(f"market_shares: {by!r} not in stores. Pass it through "
                             f"prepare_stores(..., keep=({by!r},)). Got {list(stores.columns)}")
        df = df.join(stores.select("store_id", by), on="store_id", how="left") \
               .with_columns(pl.col(by).fill_null("UNKNOWN"))
    return (df.group_by(region_col, by)
              .agg(pl.col("pop_alloc").cast(pl.Float64).sum().alias("value"))
              .with_columns((pl.col("value") / pl.col("value").sum().over(region_col))
                            .alias("share"))
              .sort([region_col, "share", by], descending=[False, True, False]))


def hhi(shares: pl.DataFrame, *, region_col: str = "region_code",
        scale: float = HHI_SCALE) -> pl.DataFrame:
    """HHI per region, plus n_players and CR1/CR3.

    CR alongside HHI because HHI alone can't separate "one dominant player" from
    "several mid-sized ones", and both readings matter for a merger argument.
    """
    return (shares.group_by(region_col).agg(
                (pl.col("share").pow(2).sum() * scale).alias("hhi"),
                pl.len().alias("n_players"),
                pl.col("share").max().alias("cr1"),
                pl.col("share").sort(descending=True).head(3).sum().alias("cr3"),
                pl.col("value").sum().alias("pop"))
            .sort(region_col))


def local_market(
    assignment: pl.DataFrame, stores: pl.DataFrame, *, by: str = "Gruppe",
    size_col: str | None = None, scale: float = HHI_SCALE,
    catchment: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Concentration of the choice set each store's own customers face.

    Returns store_id, local_hhi, n_groups, n_stores, catchment_pop -- the last
    four population-weighted over the store's rank-1 catchment.

    The measure that doesn't inherit administrative boundaries. A store on a
    Kreis border faces rivals on both sides; a Kreis-level HHI splits that in two
    and reports each half as more concentrated than it is. Present the Kreis
    figure for comparability with official statistics, defend with this one.


    THE CHOICE SET MUST BE AN ISOCHRONE, NOT A TOP-N
    ------------------------------------------------
    `assignment` must be built as the set of stores each cell can REACH:

        a_iso = nv.assign(c2, stores, cand, dur, year=y,
                          ranks=cand.k, max_seconds=20*60)

    not as a fixed top-N. This is not a preference. An HHI computed over a choice
    set capped at N alternatives is bounded below by scale/N whatever the market
    looks like: cap it at 3 and every store in Germany reports at least 3,333,
    the median of that distribution is an artefact of the cap, and the number is
    not comparable to the Kreis HHI printed beside it, which has no such floor.
    A rank cap silently sets the answer; a time cap states a market definition.

    20 minutes is the usual grocery isochrone; report 15/20/25 and say which.

    TRUNCATION. The isochrone is only honest where it is not cut off by k. If a
    cell has all k candidates inside `max_seconds`, its true choice set is larger
    than what was solved, and its HHI is biased UP. That share is printed; if it
    is not near zero, raise k (via nv.refine_until) rather than reading the
    numbers.


    WEIGHTING. Groups are weighted by store count by default, or by `size_col`
    (VKFL) if given -- two stores of very different size are not two equal
    competitors. Neither weighting is a demand model: this measures what is
    available, not what is chosen. That is the point. It costs no free parameter,
    so it cannot be attacked on one.
    """
    max_rank = int(assignment["rank"].max())
    if max_rank < 2:
        raise ValueError(
            "local_market: the choice set has only rank 1, so every catchment is "
            "trivially a monopoly. Rebuild with nv.assign(..., ranks=cand.k, "
            "max_seconds=<isochrone>)."
        )

    a = assignment.select("cell_id", "rank", "store_id", "pop")
    cols = [c for c in ({by} | ({size_col} if size_col else set())) if c != "store_id"]
    if cols:
        miss = [c for c in cols if c not in stores.columns]
        if miss:
            raise ValueError(f"local_market: {miss} not in stores. Pass through "
                             f"prepare_stores(..., keep=...) or join them on.")
        a = a.join(stores.select("store_id", *cols), on="store_id", how="left")
    if by != "store_id":
        a = a.with_columns(pl.col(by).fill_null("UNKNOWN"))

    if size_col is None:
        w = pl.lit(1.0)
    else:
        # A recorded size of 0 is an unrecorded store, not a store with no floor
        # space -- but it is still a competitor, so it must not be dropped or
        # given weight 0 (a cell where every alternative is 0 would then divide
        # 0/0 and return NaN, which propagates into the HHI without raising).
        # Impute at the median of the observed positive sizes.
        pos = stores.filter(pl.col(size_col) > 0)[size_col]
        if pos.len() == 0:
            raise ValueError(f"local_market: no store has a positive {size_col!r}")
        fill = float(pos.median())
        n_imp = stores.height - pos.len()
        if n_imp:
            print(f"local_market: {n_imp:,}/{stores.height:,} store(s) have no usable "
                  f"{size_col!r}; weighted at the median {fill:,.0f}.")
        w = pl.when(pl.col(size_col) > 0).then(pl.col(size_col).cast(pl.Float64)) \
              .otherwise(pl.lit(fill))

    per_cell = (
        a.with_columns(w.alias("_w"))
         .group_by("cell_id", by)
         .agg(pl.col("_w").sum().alias("w"), pl.len().alias("n"))
         .group_by("cell_id")
         .agg(((pl.col("w") / pl.col("w").sum()).pow(2).sum() * scale).alias("h"),
              pl.len().alias("n_groups"),
              pl.col("n").sum().alias("n_stores"))
    )
    sat = per_cell.filter(pl.col("n_stores") >= max_rank).height
    if sat:
        print(f"local_market: {sat:,}/{per_cell.height:,} cell(s) "
              f"({sat / per_cell.height:.1%}) have all {max_rank} solved candidates "
              f"inside the isochrone -- their choice set is truncated by k and their "
              f"HHI is biased UP. Widen k (nv.refine_until) before reporting.")

    base = catchment if catchment is not None else assignment.filter(pl.col("rank") == 1)
    p = pl.col("pop").cast(pl.Float64)
    return (base.select("cell_id", "store_id", "pop")
            .join(per_cell, on="cell_id", how="inner")
            .group_by("store_id").agg(
                ((pl.col("h") * p).sum() / p.sum()).alias("local_hhi"),
                ((pl.col("n_groups") * p).sum() / p.sum()).alias("n_groups"),
                ((pl.col("n_stores") * p).sum() / p.sum()).alias("n_stores"),
                p.sum().alias("catchment_pop"))
            .sort("store_id"))


def label_stores(stores: pl.DataFrame, regions, *, id_col: str, epsg: int = 3035
                 ) -> pl.DataFrame:
    """Region of each store, by point-in-polygon. Returns store_id, <id_col>.

    A plain sjoin is fine here -- this is thousands of points, not millions, which
    is why cells go through rasterize_regions() and stores do not.
    """
    import geopandas as gpd

    gdf = (regions if hasattr(regions, "geometry") else gpd.read_file(regions))
    pts = gpd.GeoDataFrame(
        {"store_id": stores["store_id"].to_list()},
        geometry=gpd.points_from_xy(stores["lon"].to_numpy(), stores["lat"].to_numpy()),
        crs="EPSG:4326").to_crs(epsg)
    j = gpd.sjoin(pts, gdf[[id_col, "geometry"]].to_crs(epsg), how="left",
                  predicate="within")
    j = j[~j.index.duplicated(keep="first")]          # a point on a shared edge
    n_out = int(j[id_col].isna().sum())
    if n_out:
        print(f"label_stores: {n_out:,} store(s) fell outside every polygon")
    return pl.DataFrame({"store_id": j["store_id"].tolist(),
                         id_col: [None if v is None or v != v else str(v)
                                  for v in j[id_col].tolist()]})


def size_shares(stores: pl.DataFrame, *, by: str = "Gruppe", size_col: str = "VKFL",
                region_col: str | None = None) -> pl.DataFrame:
    """Share of total sales area (or any size measure) per `by`-unit.

    The supply-side counterpart to market_shares(). Competition practice usually
    reads local market share off sales area or turnover, not headcount, so this
    is the version an opposing expert will reach for. Feed it to hhi() exactly
    like market_shares() output -- same column names.

    `region_col` must already be on `stores` (see label_stores); omit it for a
    single national market.
    """
    for c in (by, size_col):
        if c not in stores.columns:
            raise ValueError(f"size_shares: {c!r} not in stores")
    keys = ([region_col] if region_col else []) + [by]
    df = stores.filter(pl.col(size_col).is_not_null() & (pl.col(size_col) > 0))
    if df.height < stores.height:
        print(f"size_shares: {stores.height - df.height:,} store(s) dropped for "
              f"missing or zero {size_col!r}")
    if region_col:
        df = df.with_columns(pl.col(region_col).fill_null("UNKNOWN"))
    out = (df.group_by(keys)
             .agg(pl.col(size_col).cast(pl.Float64).sum().alias("value"),
                  pl.len().alias("n_stores")))
    denom = pl.col("value").sum().over(region_col) if region_col else pl.col("value").sum()
    return out.with_columns((pl.col("value") / denom).alias("share")).sort(keys)


def store_totals(cell_alloc: pl.DataFrame, stores: pl.DataFrame | None = None,
                 *, by: str | None = None) -> pl.DataFrame:
    """Allocated population per store or per `by`-unit. The Huff-aware analogue of
    network_voronoi.catchments(), which is rank-1 only."""
    df, key = cell_alloc, "store_id"
    if by and by != "store_id":
        df = df.join(stores.select("store_id", by), on="store_id", how="left")
        key = by
    return (df.group_by(key).agg(pl.col("pop_alloc").cast(pl.Float64).sum().alias("pop"),
                                 pl.len().alias("n_cells"))
              .with_columns((pl.col("pop") / pl.col("pop").sum()).alias("share"))
              .sort(["pop", key], descending=[True, False]))


# ==========================================================================
# Maps
# ==========================================================================

def _xy(cells: pl.DataFrame) -> list[str]:
    """Display coordinates, aliased to x/y.

    Prefers the pre-snap position when `snap` kept it. Plotting snapped
    coordinates draws population strung along road centrelines with holes between
    them -- an artefact of routing, not a fact about where people live. Routing,
    dedupe and the certification bound still use the snapped x/y; only maps use
    these.
    """
    if "x_orig" in cells.columns:
        return [pl.col("x_orig").alias("x"), pl.col("y_orig").alias("y")]
    return ["x", "y"]


def plot_choropleth(metric: pl.DataFrame, regions, lookup: pl.DataFrame, *,
                    id_col: str, value_col: str = "hhi", title: str = "",
                    cmap: str = "OrRd", ax=None, save_path: str | Path | None = None,
                    dpi: int = 200, **kw):
    """Region choropleth of any metric column (hhi, cr1, cr3, share).

    `save_path` writes the figure (png/pdf/svg by extension) after plotting and
    prints the path. If `ax` was passed in as part of a larger figure, this saves
    that whole figure, not just this subplot.
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt

    gdf = (regions if hasattr(regions, "geometry") else gpd.read_file(regions)).copy()
    j = lookup.join(metric, on="region_code", how="left")
    vals = dict(zip(j[id_col].to_list(), j[value_col].to_list()))
    gdf[value_col] = [vals.get(str(v)) for v in gdf[id_col].tolist()]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 9))
    gdf.plot(column=value_col, cmap=cmap, ax=ax, legend=True, edgecolor="white",
             linewidth=0.2, missing_kwds={"color": "#eee"}, **kw)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"saved {save_path}")
    return ax


def plot_catchments(assignment: pl.DataFrame, cells: pl.DataFrame, stores: pl.DataFrame,
                    *, by: str = "Gruppe", region=None, region_col: str | None = None,
                    region_name: str | None = None, top: int = 12, s: float | None = None,
                    title: str = "", ax=None, boundaries=None,
                    save_path: str | Path | None = None, dpi: int = 200):
    """Solve nodes coloured by the Gruppe of their rank-1 store.

    `save_path` writes the figure (png/pdf/svg by extension) after plotting and
    prints the path. Use a raster format (png) for the national view -- millions
    of scatter points in a vector pdf/svg is a very large, slow file.

    `cells` must be the frame the assignment is keyed on (post-dedupe) -- it
    supplies x/y. Pass `region`/`region_col`/`region_name` to zoom to one Kreis,
    e.g. region=kreis_gdf, region_col="GEN", region_name="Mannheim".

    Drawn as a dense scatter, not polygons: that is what was actually computed,
    and it shows genuine ragged catchment edges rather than implying clean
    boundaries that were never estimated. Groups outside the top-N collapse to
    'other' -- a 40-chain legend is unreadable and the tail invisible anyway.

    Marker size defaults to scaling with the number of nodes actually in view, so
    a zoomed Kreis doesn't render as a near-empty scatter of national-sized dots.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    r1 = (assignment.filter(pl.col("rank") == 1)
          .join(stores.select("store_id", by), on="store_id", how="left")
          .join(cells.select("cell_id", *_xy(cells)), on="cell_id", how="inner")
          .with_columns(pl.col(by).fill_null("UNKNOWN")))

    bounds = None
    if region is not None and region_name is not None:
        sel = region[region[region_col].astype(str) == region_name].to_crs(3035)
        if sel.empty:
            raise ValueError(f"plot_catchments: no {region_name!r} in {region_col!r}")
        x0, y0, x1, y1 = sel.total_bounds
        pad = 0.05 * max(x1 - x0, y1 - y0)
        bounds = (x0 - pad, x1 + pad, y0 - pad, y1 + pad)
        r1 = r1.filter(pl.col("x").is_between(bounds[0], bounds[1])
                       & pl.col("y").is_between(bounds[2], bounds[3]))
        if boundaries is None:
            boundaries = sel

    keep = (r1.group_by(by).agg(pl.len().alias("n"))
              .sort(["n", by], descending=[True, False]).head(top)[by].to_list())
    r1 = r1.with_columns(pl.when(pl.col(by).is_in(keep)).then(pl.col(by))
                           .otherwise(pl.lit("other")).alias("_g"))
    cats = keep + (["other"] if (r1["_g"] == "other").any() else [])
    cmap = plt.get_cmap("tab20")
    colour = {g: cmap(i % 20) for i, g in enumerate(cats)}

    if s is None:      # dense national view needs tiny points, a zoomed Kreis needs
        s = float(np.clip(2.5e5 / max(r1.height, 1), 0.3, 12.0))   # visible ones
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 9))
    ax.scatter(r1["x"].to_numpy(), r1["y"].to_numpy(),
               c=[colour[g] for g in r1["_g"].to_list()], s=s, linewidths=0,
               rasterized=True)
    if boundaries is not None:
        import geopandas as gpd
        b = boundaries if hasattr(boundaries, "geometry") else gpd.read_file(boundaries)
        b.to_crs(3035).boundary.plot(ax=ax, color="black", linewidth=0.4, alpha=0.5)
    if bounds:
        ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal"); ax.set_axis_off()
    if title:
        ax.set_title(title)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=colour[g], label=g)
                       for g in cats], loc="upper left", fontsize=7, framealpha=0.9)
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"saved {save_path}")
    return ax