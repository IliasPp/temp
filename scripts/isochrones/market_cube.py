"""
Market structure cube -- shares and HHI over every reporting dimension.

Consumes the catchment solve (`network_voronoi`) plus the TDLinx panel and emits
two tidy tables. Everything downstream (tables, figures, exports) reads these and
nothing else, so a number in a figure can always be traced to one row.

    shares   one row per (spec, year, level, region, basis, actor_level, actor)
    hhi      one row per (spec, year, level, region, basis, actor_level)

Dimensions
----------
    spec         super | super_disc | super_disc_drog                     (3)
    year         2014 .. 2025                                            (12)
    level        DE | BL | KREIS                                          (3)
    basis        pop | area | vkfl | vkfl_food | umsatz | umsatz_food      (6)
    actor_level  Gruppe (ownership) | Chain (banner)                       (2)

`basis` folds two of the instruction's dimensions together, because they are not
orthogonal: food/non-food only exists for the supply-side measures (a person
allocated to a store is not "food population"), so 2 measures x 2 food variants
+ 2 demand-side measures = 6, not 8.

Demand vs supply side
---------------------
    pop, area           the store's *catchment*: population and inhabited area
                        won from the network-Voronoi solve. Region = the region
                        the cell sits in, so a store serves several Kreise and
                        its weight splits across them.
    vkfl, umsatz, ...   the store's own attributes. Region = the store's own
                        AGS5. This is the supply-side reading used in
                        competition practice and it is what makes the two
                        columns worth reporting side by side: where they
                        disagree, the catchment crosses a border.

Both describe the same set of stores in the same year (the spec's members, as
the assignment used them; stores that failed to snap are out on both sides), so the comparison is internal, not a comparison of
two datasets. Only the demand side needs the solve: a VKFL/Umsatz-only report
is built from the panel alone (`cube(assignments=None, ...)`).

Data flow for one (spec, year)
------------------------------
    assignment (node -> store)  x  node_regions (node -> Kreis, pop, area)
        -> demand_atoms    (store, kreis, basis in {pop, area}, value)
    panel rows of the year's member stores
        -> supply_atoms    (store, kreis, basis in {vkfl, umsatz, ...}, value)
    atoms  x  actor_map (store -> Gruppe, Chain, that year)
        -> shares_and_hhi  -> shares rows + hhi rows for DE, BL, KREIS
An "atom" is the smallest unit summed: one store's value in one Kreis.

Cost
----
The expensive object is `node_regions`, built once per (grid, region layer):
it collapses the cell mapping (7.27M populated cells) to one row per (solve
node, Kreis). After
that every (spec, year) cube is a join on a few million nodes and a group_by on
a few tens of thousands of rows -- seconds, not minutes. Do not expand back to
100m cells per year; the node table already carries the exact per-region split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import polars as pl

HHI_SCALE = 10_000.0
UNKNOWN = "UNKNOWN"

#: basis name -> (side, source column). Supply columns are panel columns; missing
#: ones are skipped with a warning rather than failing the run.
BASES: dict[str, tuple[str, str]] = {
    "pop": ("demand", "pop"),
    "area": ("demand", "area_km2"),
    "vkfl": ("supply", "VKFL"),
    "vkfl_food": ("supply", "Verkaufsfläche Food"),
    "umsatz": ("supply", "Umsatzpotential Brutto (Mio EUR)"),
    "umsatz_food": ("supply", "Umsatzpotential Brutto Food (Mio EUR)"),
}

LEVELS = ("DE", "BL", "KREIS")

#: Kreis reforms inside 2014-2025: the panel carries the AGS5 of its own year,
#: the region layer the current one. Old code -> current code. Extend it when
#: `unknown_kreise()` reports a code; it assumes a region layer from 2021 or later.
KREIS_XWALK = {
    "03152": "03159", "03156": "03159",    # Göttingen + Osterode am Harz, 2016-11
    "16056": "16063",                      # Eisenach -> Wartburgkreis, 2021-07
}


# ==========================================================================
# helpers
# ==========================================================================

def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    """Streaming collect, tolerant of the polars version."""
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def ags5(col: str) -> pl.Expr:
    """Normalise any AGS5-ish column to a 5-character string, nulls -> UNKNOWN."""
    return (pl.col(col).cast(pl.Utf8).str.replace_all(r"\D", "").str.zfill(5)
            .str.slice(0, 5).replace("", None).replace(KREIS_XWALK)
            .fill_null(UNKNOWN).alias("kreis"))


def unknown_kreise(panel: pl.DataFrame, known, *, region_col: str = "AGS5") -> list[str]:
    """Panel Kreis codes (after KREIS_XWALK) that the region layer does not have.

    A code here is a Kreis reform the crosswalk misses: its stores still count
    in DE and BL, but vanish from every Kreis map and the Kreis tables.
    """
    got = panel.select(ags5(region_col)).unique()["kreis"].to_list()
    return sorted(set(got) - set(map(str, known)) - {UNKNOWN})


def _region_expr(level: str) -> pl.Expr:
    if level == "DE":
        return pl.lit("DE")
    if level == "BL":
        return (pl.when(pl.col("kreis") == UNKNOWN).then(pl.lit(UNKNOWN))
                  .otherwise(pl.col("kreis").str.slice(0, 2)))
    if level == "KREIS":
        return pl.col("kreis")
    raise ValueError(f"level must be one of {LEVELS}, got {level!r}")


# ==========================================================================
# 1. node -> region weights  (built once)
# ==========================================================================

def rasterize_regions(regions, template, cells: pl.DataFrame, *, id_col: str,
                      name_col: str | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Label every original cell with its region. Returns (labels, lookup).

    Burns the polygons onto the population raster's own grid and indexes it by
    (row, col): O(n) with a C inner loop, where a 7.27M-point sjoin is not.
    Cells outside every polygon get region_code 0.
    """
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(template) as src:
        transform, shape, crs = src.transform, (src.height, src.width), src.crs
    # region_layers() already delivers the polygons in the raster CRS, so this is
    # a no-op in the pipeline; it matters only when called with another layer
    gdf = regions.to_crs(crs)
    cols = [id_col] + ([name_col] if name_col else [])
    lookup = (pl.DataFrame({c: [None if v is None else str(v) for v in gdf[c].tolist()]
                            for c in cols})
              .with_row_index("region_code", offset=1)
              .with_columns(pl.col("region_code").cast(pl.Int32)))
    grid = rasterize(zip(gdf.geometry, lookup["region_code"].to_list()),
                     out_shape=shape, transform=transform, fill=0, dtype="int32")
    code = grid[cells["row"].to_numpy(), cells["col"].to_numpy()]
    if (n_out := int((code == 0).sum())):
        pop_out = cells.filter(pl.Series(code == 0))["pop"].sum()
        print(f"rasterize_regions: {n_out:,} cell(s) outside all polygons "
              f"({pop_out / cells['pop'].sum():.2%} of pop) -> code 0")
    return cells.select("cell_id").with_columns(pl.Series("region_code", code)), lookup


def node_regions(
    mapping: pl.DataFrame, cells: pl.DataFrame, labels: pl.DataFrame,
    lookup: pl.DataFrame, *, id_col: str = "AGS5", cell_km2: float = 0.01,
) -> pl.DataFrame:
    """Collapse the 100 m grid to (node_id, kreis, pop, n_cells, area_km2).

    `mapping` is compose() of every reduction (src = original cell_id, dst =
    solve node), `cells` is load_cells() output, `labels`/`lookup` come from
    rasterize_regions() on the *original* grid.

    A solve node that straddles a Kreis border appears once per Kreis with its
    population split by the constituent cells -- which is why regions are
    labelled on the original grid and never on the node.

    `area_km2` counts the cells the raster actually carries. GHS-POP cells with
    no population are not in the solve, so this is inhabited area, not land
    area; that is the defensible denominator for a retail catchment but it must
    be called by its name in the text.
    """
    lk = lookup.select(pl.col("region_code"), ags5(id_col))
    lf = (mapping.lazy().rename({"src": "cell_id", "dst": "node_id"})
          .join(cells.lazy().select("cell_id", "pop"), on="cell_id", how="inner")
          .join(labels.lazy(), on="cell_id", how="left")
          .with_columns(pl.col("region_code").fill_null(0))
          .group_by("node_id", "region_code")
          .agg(pl.col("pop").cast(pl.Float64).sum().alias("pop"),
               pl.len().alias("n_cells"))
          .join(lk.lazy(), on="region_code", how="left")
          .with_columns(pl.col("kreis").fill_null(UNKNOWN),
                        (pl.col("n_cells") * cell_km2).alias("area_km2"))
          .select("node_id", "kreis", "pop", "n_cells", "area_km2")
          .sort("node_id", "kreis"))
    out = _collect(lf)
    unk = out.filter(pl.col("kreis") == UNKNOWN)["pop"].sum()
    tot = out["pop"].sum()
    if unk:
        print(f"node_regions: {unk:,.0f} people ({unk / tot:.2%}) outside every "
              f"polygon -> kreis={UNKNOWN!r}; they count in DE, not in BL/KREIS")
    print(f"node_regions: {out.height:,} (node, Kreis) rows | "
          f"{out['area_km2'].sum():,.0f} km2 inhabited | {tot:,.0f} people")
    return out


# ==========================================================================
# 2. atoms:  (store_id, kreis, basis, value)
# ==========================================================================

def demand_atoms(assignment: pl.DataFrame, node_reg: pl.DataFrame) -> pl.DataFrame:
    """Catchment population and area per (store, Kreis) from a rank-1 assignment."""
    a = assignment.filter(pl.col("rank") == 1).select(
        pl.col("cell_id").alias("node_id"),
        pl.col("store_id").cast(pl.Utf8))      # ids are Utf8 throughout
    w = (a.lazy().join(node_reg.lazy(), on="node_id", how="inner")
         .group_by("store_id", "kreis")
         .agg(pl.col("pop").sum(), pl.col("area_km2").sum())
         .sort("store_id", "kreis"))
    return (_collect(w)
            .unpivot(index=["store_id", "kreis"], on=["pop", "area_km2"],
                     variable_name="basis", value_name="value")
            .with_columns(pl.col("basis").replace({"area_km2": "area"})))


def supply_atoms(
    panel_year: pl.DataFrame, *, bases: Mapping[str, tuple[str, str]] = BASES,
    id_col: str = "TDLinx", region_col: str = "AGS5",
    store_ids: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Store-level size/revenue measures, keyed on the store's own Kreis.

    `store_ids` restricts to the stores that actually entered the solve, so the
    supply-side and demand-side columns describe the same set of stores. Without
    it a store dropped for failing to snap would still carry sales area.
    """
    cols = {b: c for b, (side, c) in bases.items() if side == "supply"}
    have = {b: c for b, c in cols.items() if c in panel_year.columns}
    for b, c in cols.items():
        if b not in have:
            print(f"supply_atoms: column {c!r} not in panel -> basis {b!r} skipped")
    if not have:
        return pl.DataFrame(schema={"store_id": pl.Utf8, "kreis": pl.Utf8,
                                    "basis": pl.Utf8, "value": pl.Float64})
    df = panel_year.select(pl.col(id_col).cast(pl.Utf8).alias("store_id"),
                           ags5(region_col),
                           *[pl.col(c).cast(pl.Float64).alias(b) for b, c in have.items()])
    if store_ids is not None:
        df = df.filter(pl.col("store_id").is_in(
            pl.Series(list(store_ids), dtype=pl.Utf8).implode()))
    # one row per store even if the panel repeats it within a year
    df = df.group_by("store_id").agg(pl.col("kreis").first(),
                                     *[pl.col(b).sum() for b in have])
    return df.unpivot(index=["store_id", "kreis"], on=list(have),
                      variable_name="basis", value_name="value") \
             .filter(pl.col("value").is_not_null())


def actor_map(
    panel_year: pl.DataFrame, *, id_col: str = "TDLinx",
    gruppe_col: str = "Gruppe", chain_col: str = "Chain",
) -> pl.DataFrame:
    """(store_id, Gruppe, Chain) for one year, read straight from the panel.

    Gruppe is ownership/buying group (Schwarz, Edeka), Chain is the banner the
    shopper sees (Lidl, Kaufland). Both are reported because a merger case turns
    on the first and a consumer-facing market description on the second. Both are
    per store-year: banners switch, and the year's value is the one that counts.

    Resolving what a chain *is* -- banner, fascia, name of an unbranded store --
    is the panel's job, not this function's. It reads one column per actor level
    and nothing else.
    """
    missing = [c for c in (id_col, gruppe_col, chain_col) if c not in panel_year.columns]
    if missing:
        raise ValueError(f"actor_map: {', '.join(missing)} not in the panel "
                         f"({', '.join(panel_year.columns[:12])}...)")
    return (panel_year.select(pl.col(id_col).cast(pl.Utf8).alias("store_id"),
                              pl.col(gruppe_col).cast(pl.Utf8).alias("Gruppe"),
                              pl.col(chain_col).cast(pl.Utf8).alias("Chain"))
            .group_by("store_id").agg(pl.col("Gruppe").last(), pl.col("Chain").last())
            .with_columns(pl.col("Gruppe").fill_null(UNKNOWN),
                          pl.col("Chain").fill_null(UNKNOWN)))


# ==========================================================================
# 3. the cube
# ==========================================================================

def shares_and_hhi(
    atoms: pl.DataFrame, actors: pl.DataFrame, *, year: int, spec: str,
    levels: Sequence[str] = LEVELS, actor_levels: Sequence[str] = ("Gruppe", "Chain"),
    top: int = 5, exclude: Sequence[str] = (), fringe: Sequence[str] = (UNKNOWN,),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Shares and concentration for one (spec, year) over every level/basis/actor.

    HHI is on the 0-10,000 scale (squared percentage shares), the German/EU
    convention. Shares are computed over the actors present in that region: a
    Kreis with two Gruppen has an HHI of at least 5,000 by construction, which
    is the point -- it is a claim about that Kreis, not about Germany.

    Fringe. Stores whose actor is unidentified (`fringe`, UNKNOWN by default)
    are not one firm. Pooling them into a single actor makes the residual look
    like a large competitor and inflates the HHI; so `hhi` counts each fringe
    store as its own firm, and `hhi_pooled` keeps the pooled reading for
    comparison. The shares table keeps one pooled fringe row per region (its
    share is real), with `rank` null: CR1/CR3 and "the leader" are named actors.
    """
    if atoms.is_empty():
        raise ValueError(f"shares_and_hhi: no atoms for {spec} {year}")
    # Labels are compared case-insensitively. An atom with no actor label gets
    # UNKNOWN, which is in the default fringe list (runconfig.UNIDENTIFIED).
    drop = {x.strip().casefold() for x in exclude if x and x.strip()}
    fr = [x.strip().casefold() for x in fringe]
    by = ["region", "basis"]
    S, H = [], []
    for al in actor_levels:
        a = (atoms.join(actors.select("store_id", pl.col(al).alias("actor")),
                        on="store_id", how="left")
             .with_columns(pl.col("actor").fill_null(UNKNOWN)))
        key = pl.col("actor").str.strip_chars().str.to_lowercase()
        if drop:
            # removed from the market, not reallocated: the shares that follow are
            # conditional on the identified operators and are all higher for it
            a = a.filter(~key.is_in(list(drop)))
            if a.is_empty():
                raise ValueError(f"shares_and_hhi: every actor excluded at "
                                 f"{al} level ({spec} {year})")
        a = a.with_columns(key.is_in(fr).alias("_fr"))
        for lev in levels:
            # In the formulas below s_j is actor j's share of the (region, basis)
            # total, F the fringe stores, s_f one fringe store's own share:
            #   hhi        = 10^4 * ( sum_{named j} s_j^2 + sum_{f in F} s_f^2 )
            #   hhi_pooled = 10^4 * ( sum over ALL actor labels of s_label^2 ),
            #                i.e. each fringe label (UNKNOWN, "Sonstige", ...) as one actor
            #   crN        = sum of the N largest NAMED shares
            al_ = a.with_columns(_region_expr(lev).alias("region"))
            tot = al_.group_by(by).agg(pl.col("value").sum().alias("_tot"))
            s = (al_.group_by(*by, "actor")
                 .agg(pl.col("value").sum(),
                      # distinct, not pl.len(): a catchment straddling two Kreise
                      # contributes one atom per Kreis
                      pl.col("store_id").n_unique().alias("n_stores"),
                      pl.col("_fr").first())
                 .join(tot, on=by).filter(pl.col("_tot") > 0)
                 .with_columns((pl.col("value") / pl.col("_tot")).alias("share"))
                 # explicit tie-break on actor: group_by order is not stable
                 # rank: 1..n over NAMED actors by descending share (fringe rows
                 # sort last and get rank null)
                 .sort(*by, "_fr", "share", "actor",
                       descending=[False, False, False, True, False])
                 .with_columns(pl.when(~pl.col("_fr"))
                               .then((~pl.col("_fr")).cum_sum().over(by))
                               .cast(pl.Int32).alias("rank")))
            # fringe stores' squared shares, each store treated as its own firm
            fsq = (al_.filter(pl.col("_fr"))
                   .group_by(*by, "store_id").agg(pl.col("value").sum())
                   .join(tot, on=by)
                   .group_by(by).agg((pl.col("value") / pl.col("_tot")).pow(2).sum()
                                     .alias("_fsq")))
            named = pl.col("share").filter(~pl.col("_fr")).sort(descending=True)
            h = (s.group_by(by).agg(
                    (pl.col("share").filter(~pl.col("_fr")).pow(2).sum()).alias("_nsq"),
                    (pl.col("share").pow(2).sum() * HHI_SCALE).alias("hhi_pooled"),
                    named.head(1).sum().alias("cr1"),
                    named.head(3).sum().alias("cr3"),
                    named.head(top).sum().alias(f"cr{top}"),
                    (~pl.col("_fr")).sum().cast(pl.UInt32).alias("n_actors"),
                    pl.col("n_stores").sum().alias("n_stores"),
                    pl.col("share").filter(pl.col("_fr")).sum().alias("fringe_share"),
                    pl.col("value").sum().alias("total"))
                 .join(fsq, on=by, how="left")
                 .with_columns(((pl.col("_nsq") + pl.col("_fsq").fill_null(0.0))
                                * HHI_SCALE).alias("hhi"))
                 .sort(by))
            lab = [pl.lit(year).cast(pl.Int32).alias("year"), pl.lit(spec).alias("spec"),
                   pl.lit(lev).alias("level"), pl.lit(al).alias("actor_level")]
            S.append(s.with_columns(lab))
            H.append(h.with_columns(lab))

    order_s = ["spec", "year", "level", "region", "basis", "actor_level", "actor",
               "rank", "value", "share", "n_stores"]
    order_h = ["spec", "year", "level", "region", "basis", "actor_level", "hhi",
               "hhi_pooled", "cr1", "cr3", f"cr{top}", "n_actors", "fringe_share",
               "n_stores", "total"]
    return (pl.concat(S).select(order_s), pl.concat(H).select(order_h))


def alive_stores(stores: pl.DataFrame, year: int) -> list[str]:
    """Stores in the solve that exist in `year` (first_year <= y <= last_year)."""
    return (stores.filter((pl.col("first_year") <= year) & (pl.col("last_year") >= year))
            ["store_id"].cast(pl.Utf8).to_list())


def cube(
    assignments: Mapping[int, pl.DataFrame] | None, node_reg: pl.DataFrame | None,
    panel: pl.DataFrame, *, spec: str, stores: pl.DataFrame | None = None,
    alive_by_year: Mapping[int, Sequence[str]] | None = None,
    years: Sequence[int] | None = None,
    bases: Mapping[str, tuple[str, str]] = BASES, id_col: str = "TDLinx",
    region_col: str = "AGS5", gruppe_col: str = "Gruppe", chain_col: str = "Chain",
    year_col: str = "year", top: int = 5, exclude: Sequence[str] = (),
    fringe: Sequence[str] = (UNKNOWN,), actor_panel: pl.DataFrame | None = None,
    levels: Sequence[str] = LEVELS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Full cube for one spec: every year in `assignments`, all levels/bases/actors.

    `alive_by_year` is the store set that belonged to this spec in each year, as
    the assignment used it. Pass it whenever formats change over the panel: a
    store trading as a supermarket until 2019 and a discounter after is in the
    supermarket market only until 2019, and its sales area must leave the
    denominator in the same year its catchment does. Without it, `stores` is used
    and membership is whatever `prepare_stores` collapsed it to.

    `actor_panel` supplies the Gruppe/Chain labels when it must be wider than
    `panel`: a Land report over the national solve restricts the supply side to
    the Land's stores, but a store across the border that wins cells inside it
    still needs its operator.

    Falling back to the assignment's winners would silently drop a store that
    exists but wins no cell (two stores on one node), keeping its sales area out
    of the denominator.

    `assignments=None` builds a supply-side-only cube (`vkfl`, `umsatz` and their
    food variants) straight from the panel. That is the whole cube for a report
    that never asked for catchment population or area, and it needs no solve --
    pass `years` and `alive_by_year` instead. The pipeline drops snap-rejected
    stores from `alive_by_year` in both cases (run_pipeline.rejected_ids), so
    supply-side numbers do not depend on whether the run solved.

    `levels`: a Land run passes ("BL", "KREIS") -- a DE row built from one
    Land's stores would be that Land's total under Germany's name.
    """
    if years is None:
        years = sorted(assignments) if assignments is not None \
            else sorted(panel[year_col].unique().to_list())
    else:
        years = sorted(years)
    S, H = [], []
    for y in years:
        a = None if assignments is None else assignments.get(y)
        py = panel.filter(pl.col(year_col) == y)
        if py.is_empty():
            print(f"cube[{spec}]: no panel rows for {y}, skipped")
            continue
        # which stores make up the market this year, in order of preference:
        # the spec's members as the assignment used them (the pipeline always
        # passes this) > alive window of a store frame > the assignment's
        # winners > every panel store of the year
        if alive_by_year is not None and y in alive_by_year:
            ids = list(alive_by_year[y])
        elif stores is not None:
            ids = alive_stores(stores, y)
        elif a is not None:
            ids = a["store_id"].cast(pl.Utf8).unique().to_list()
        else:
            ids = py[id_col].cast(pl.Utf8).unique().to_list()
        sup = supply_atoms(py, bases=bases, id_col=id_col, region_col=region_col,
                           store_ids=ids)
        atoms = (sup if a is None or node_reg is None
                 else pl.concat([demand_atoms(a, node_reg), sup],
                                how="vertical_relaxed"))
        if atoms.is_empty():
            print(f"cube[{spec}]: nothing to report for {y}, skipped")
            continue
        apy = py if actor_panel is None else actor_panel.filter(pl.col(year_col) == y)
        s, h = shares_and_hhi(atoms,
                              actor_map(apy, id_col=id_col, gruppe_col=gruppe_col,
                                        chain_col=chain_col),
                              year=y, spec=spec, top=top, exclude=exclude,
                              fringe=fringe, levels=levels)
        S.append(s)
        H.append(h)
        shown = "pop" if a is not None else "vkfl"
        de = h.filter((pl.col("level") == levels[0]) & (pl.col("region") != UNKNOWN)
                      & (pl.col("actor_level") == "Gruppe") & (pl.col("basis") == shown))
        msg = f"  {spec} {y}: {len(ids):,} stores"
        if de.height:
            msg += f" | {de['region'][0]} HHI({shown}, Gruppe) {de['hhi'][0]:,.0f}"
        print(msg)
    if not S:
        raise ValueError(f"cube[{spec}]: no year produced any rows")
    return pl.concat(S), pl.concat(H)


# ==========================================================================
# 4. checks + convenience
# ==========================================================================

def panel_diagnostics(
    panel: pl.DataFrame, *, id_col: str = "TDLinx", year_col: str = "year",
    gruppe_col: str = "Gruppe", size_col: str = "VKFL",
    rev_col: str = "Umsatzpotential Brutto (Mio EUR)",
    unidentified: Sequence[str] = (),
) -> pl.DataFrame:
    """Per-year panel composition -- the first thing to look at when a series moves.

    A concentration series can rise for three different reasons and they leave
    different traces here:

    * the market really concentrated -- store counts fall for small operators
      while the large ones hold or grow;
    * the *panel* changed -- coverage, entries and exits move together, or a
      vintage lands with a different store universe;
    * the *labels* changed -- stores move out of the unidentified bucket into a
      named Gruppe, which raises measured concentration without a single store
      changing hands. `unidentified_share` and `regrouped` track that directly.

    `regrouped` counts stores whose Gruppe differs from their own previous year,
    and `newly_identified` the subset that came out of the unidentified bucket.
    A spike there in one year is a reclassification, not a merger wave.
    """
    uid = {u.casefold() for u in unidentified}
    df = panel.select(
        pl.col(id_col).cast(pl.Utf8).alias("store_id"),
        pl.col(year_col).cast(pl.Int32).alias("year"),
        pl.col(gruppe_col).cast(pl.Utf8).fill_null(UNKNOWN).alias("g"),
        *( [pl.col(size_col).cast(pl.Float64).alias("size")]
           if size_col in panel.columns else [pl.lit(None, pl.Float64).alias("size")]),
        *( [pl.col(rev_col).cast(pl.Float64).alias("rev")]
           if rev_col in panel.columns else [pl.lit(None, pl.Float64).alias("rev")]),
    ).with_columns(
        pl.col("g").str.strip_chars().str.to_lowercase().is_in(list(uid)).alias("unid"))

    prev = (df.select("store_id", "year", "g", "unid")
            .with_columns((pl.col("year") + 1).alias("year"))
            .rename({"g": "g_prev", "unid": "unid_prev"}))
    df = df.join(prev, on=["store_id", "year"], how="left")

    out = (df.group_by("year").agg(
        pl.len().alias("rows"),
        pl.col("store_id").n_unique().alias("stores"),
        pl.col("g").n_unique().alias("gruppen"),
        pl.col("unid").mean().alias("unidentified_share"),
        (pl.col("size").filter(pl.col("unid")).sum()
         / pl.col("size").sum()).alias("unidentified_share_vkfl"),
        pl.col("size").is_null().mean().alias("vkfl_null"),
        pl.col("rev").is_null().mean().alias("umsatz_null"),
        pl.col("size").sum().alias("vkfl_total"),
        pl.col("size").median().alias("vkfl_median"),
        pl.col("g_prev").is_null().mean().alias("entrants_share"),
        ((pl.col("g_prev").is_not_null()) & (pl.col("g") != pl.col("g_prev")))
        .sum().alias("regrouped"),
        ((pl.col("unid_prev")) & (~pl.col("unid"))).sum().alias("newly_identified"),
    ).sort("year"))
    return out


def validate(shares: pl.DataFrame, hhi: pl.DataFrame, *, tol: float = 1e-6) -> None:
    """Raise on anything that would make a published number wrong."""
    s = (shares.group_by("spec", "year", "level", "region", "basis", "actor_level")
         .agg(pl.col("share").sum().alias("s")))
    bad = s.filter((pl.col("s") - 1.0).abs() > tol)
    if bad.height:
        raise AssertionError(f"shares do not sum to 1 in {bad.height} cells:\n{bad.head()}")
    for col, lo, hi in (("hhi", 0.0, HHI_SCALE + 1e-6), ("hhi_pooled", 0.0, HHI_SCALE + 1e-6),
                        ("cr1", 0.0, 1.0 + tol),
                        ("cr3", 0.0, 1.0 + tol)):
        b = hhi.filter(~pl.col(col).is_between(lo, hi))
        if b.height:
            raise AssertionError(f"{col} out of range in {b.height} rows:\n{b.head()}")
    b = hhi.filter(pl.col("cr1") > pl.col("cr3") + tol)
    if b.height:
        raise AssertionError(f"cr1 > cr3 in {b.height} rows")
    b = hhi.filter(pl.col("hhi") > pl.col("hhi_pooled") + 1e-6)
    if b.height:
        raise AssertionError(f"atomised HHI above pooled HHI in {b.height} rows")
    print(f"validate: OK -- {shares.height:,} share rows, {hhi.height:,} hhi rows")


def coverage(assignment: pl.DataFrame, node_reg: pl.DataFrame) -> float:
    """Share of the population in `node_reg` that reached no store. Print it
    every run.

    Every join in the chain is inner, so unmapped or unreachable population
    leaves no trace in the shares -- this is the only place it is visible.
    `node_reg` sets the denominator, so a Land-filtered table gives the Land's.
    """
    got = (assignment.filter(pl.col("rank") == 1)
           .select(pl.col("cell_id").alias("node_id")).unique()
           .join(node_reg, on="node_id", how="inner")["pop"].sum())
    total = float(node_reg["pop"].sum())
    lost = 1.0 - float(got) / total if total else 0.0
    print(f"coverage: {lost:.3%} of population unallocated ({lost * total:,.0f} people)")
    return lost


def wmedian(value: str, weight: str) -> pl.Expr:
    """Weighted median of `value`: the smallest value at which the cumulative
    weight reaches half the total. For Kreis HHIs weighted by population it is
    the HHI of the local market the median *person* lives in, where the plain
    median is that of the median *Kreis* (Munich and Zweibruecken count the same).
    Use inside group_by().agg(); null where no weight is positive."""
    w = pl.col(weight).sort_by(value)
    return (pl.col(value).sort_by(value)
            .filter((w.cum_sum() / w.sum()) >= 0.5).first())


def headline(hhi: pl.DataFrame, *, level: str = "DE", basis: str = "pop",
             actor_level: str = "Gruppe") -> pl.DataFrame:
    """One row per (spec, year): the table that goes in the front of the report."""
    return (hhi.filter((pl.col("level") == level) & (pl.col("basis") == basis)
                       & (pl.col("actor_level") == actor_level))
            .select("spec", "year", "region", "hhi", "hhi_pooled", "cr1", "cr3",
                    "n_actors", "fringe_share", "n_stores", "total")
            .sort("spec", "year", "region"))


def top_actors(shares: pl.DataFrame, *, n: int = 10, **flt) -> pl.DataFrame:
    """Top-n actors for a slice, e.g. top_actors(shares, spec='super_disc',
    year=2025, level='DE', basis='vkfl', actor_level='Gruppe')."""
    q = shares
    for k, v in flt.items():
        q = q.filter(pl.col(k) == v)
    return q.filter(pl.col("rank") <= n).sort("region", "rank")