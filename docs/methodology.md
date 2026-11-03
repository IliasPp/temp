# Methodology

## Data

**Store panel.** TDLinx (Nielsen), 2014–2025, one row per store-year: location,
format (`Typ` aggregated to `Shop Type`), operator (`Gruppe`, `Chain`), sales area (`VKFL`,
`Verkaufsfläche Food`) and revenue potential (`Umsatzpotential Brutto`, total and
food, Mio EUR). Built by `scripts/tdlinx` (see below).

**Population.** GHS-POP 2025 (JRC), 100 m, Mollweide (ESRI:54009), mosaicked
from four tiles and clipped to Germany administrative borders.

**Road network.** OpenStreetMap Germany extract (Geofabrik), routed with a
self-hosted OSRM server, car profile, MLD.

**Administrative areas.** BKG VG5000 Gemeinden, DHDN / Gauss-Krüger zone 3
(EPSG:31467). Kreise and Länder are the Gemeinden dissolved on AGS5 / AGS2.

## Panel preparation (scripts/tdlinx)

- **Shop Type** is derived from `Typ`: *Supermarkt* (kleiner/großer Supermarkt, Verbrauchermarkt, SB-Warenhaus), *Discount Geschäft*, *Drogeriemarkt*, *Getränkemarkt* (including angeschlossener GAM).
- **Gruppe** is ownership / buying-group membership, not the banner: regional chains inside the Edeka cooperative (e.g. Feneberg) are Edeka; Netto Marken-Discount is Edeka, Netto Nord is Salling. Name rules, then a reviewed table (`scripts/tdlinx/actors.csv`) for what rules cannot know; a wholesaler's merchant customers are independent, not the wholesaler's.
- **Chain** is the banner, one spelling per banner; stores without a banner are `<Gruppe> (other)`, or unidentified like at Gruppe level. Full rules, the table format and the audit: [actors.md](actors.md).
- Outlier-nulled VKFL/revenue values stay null, and a zero VKFL is treated as missing.
- Every attribute is used **as it stood in that year**. A store that converted from Supermarkt to Discounter in 2019 is in the supermarket market until 2018 and in the discounter market from 2019. For example, a banner switch changes its Chain from that year on.

## Market definitions

| `spec` | Shop Types competing |
|---|---|
| `super` | Supermarkt |
| `super_disc` | Supermarkt, Discount Geschäft |
| `super_disc_drog` | Supermarkt, Discount Geschäft, Drogeriemarkt |

Definitions live in `runconfig.SHOP_TYPES` and can be changed or extended in a
run config without any new routing (see *Composition* below).

## Bases of measurement

**Supply side**: a store's own attribute, attributed to the Kreis it stands in:

| basis | column |
|---|---|
| `vkfl` | VKFL |
| `vkfl_food` | Verkaufsfläche Food |
| `umsatz` | Umsatzpotential Brutto (Mio EUR) |
| `umsatz_food` | Umsatzpotential Brutto Food (Mio EUR) |

The run's `scope` (`all` / `food`) picks the total or the food column.

**Demand side**: from drive-time catchments, attributed to the Kreis each
100 m cell lies in (a catchment crossing a Kreis border counts in both, pro
rata):

| basis | meaning |
|---|---|
| `pop` | population of the cells whose nearest store it is |
| `area` | area (km²) of those cells |

Supply-side shares are panel facts and need no OSRM routing. Demand-side shares need the catchments.

## Catchments

Each populated 100 m cell is assigned to the store it reaches fastest by car.

1. **Stores** are snapped to the nearest routable road; stores more than 300m from any road are rejected and listed in `snapped/rejected.txt`. A rejected store has no catchment, so it is out of the market on both sides: its sales area and revenue potential are dropped too, also in runs without a solve.
2. **Grid reduction.** Cells far from any store are merged into coarser cells (resolution adapts to the distance to the nearest store, so a catchment is always covered by several cells), then snapped and deduplicated on identical road nodes. Of the raster's ~35 M cells, the 7.27 M with positive population enter the solve and become 5.0 M solve nodes; every original cell keeps its link to its node, so results are reported on the 100 m grid.
3. **Candidates.** For each node, the *k* nearest stores of a shop type by straight-line distance.
4. **Drive times** node -> candidate via OSRM `/table`.
5. **Certification.** No road trip is faster than straight-line distance at `V_MAX` = 180 km/h. A node's winner is *proven* if its drive time is below `d_k / V_MAX`, the bound for every store outside the candidate set. Unproven nodes get larger candidate sets (k = 40, 60, 90, 140) until the certified share reaches `cert_target` (default 90 % of nodes).
6. **Assignment** per year uses only the stores trading in the market that year.

**Certification per year.** A matrix is certified against all its stores; in a given year some are closed or trade in another format, so a node's fastest *member* can lie beyond the bound. Every assignment reports the share of population not certified in that year. Uncertified therefore means not proven.

### Composition: one solve per shop type

The nearest store in a union of store sets is the nearest of the per-set nearest stores. So drive times are solved once per **shop type**, and every market definition is an argmin over the per-type results:

    nearest(S ∪ D)      = argmin(nearest(S), nearest(D))
    nearest(S ∪ D ∪ Dr) = argmin(nearest(S), nearest(D), nearest(Dr))

This is exact, including for the top-k isochrone frames and for certification (a composed winner is proven if it beats the smallest of the per-type bounds). `scripts/verify_compose.py` checks it against a direct solve on one Land; the only admissible differences are equal-time ties.

Consequences: a new or redefined market definition costs no routing; a store that changes format appears in the other type's solve from that year; a new shop type costs one solve.

### New panel vintages

New stores are appended to the store frame and routed only where they can matter: a new store can change a node's winner only if it lies within the node's candidate radius `d_k` (outside it the certification bound already covers it). Only those (node, store) pairs are routed; the result is identical to a full re-solve.

## Concentration measures

For each (spec, year, level, region, basis, actor level):

- **share** = actor's value / regional total, over actors present in that region.
- **HHI** = Σ (100 · share)², scale 0–10,000.
- **CR1, CR3, CR5** = summed shares of the largest 1, 3, 5 named actors.

**National and Land figures are not averages.** DE and BL shares and HHI are computed from each actor's totals over the whole area, not from Kreis numbers; the national HHI is typically far below a typical Kreis HHI, because different firms lead in different places.

**Summaries across Kreise come in two readings.** The median Kreis counts every Kreis once (Munich and Zweibrücken alike); the population-weighted median is the HHI of the market the median person lives in,

$$\tilde h_w = \min\Big\{h : \textstyle\sum_{r:\,h_r \le h} P_r \ge \tfrac12 \sum_r P_r\Big\},$$

with $P_r$ the Kreis' catchment population (`market_cube.wmedian`). Both appear in the Kreis band table (`tables.py`) and in `dist_hhi`; they need a run with the `pop` basis, since that is where $P_r$ comes from.

**Unidentified stores (fringe).** Stores without a Gruppe (or Chain) are not one firm. `hhi` counts each such store as its own firm; `hhi_pooled` treats them as one actor, for comparison. The shares table keeps one pooled row for them (`fringe_share` in the HHI table) with `rank` empty: ranks, CRn and the leader refer to named actors only.

**Excluding actors.** A run can drop unidentified actors from the market altogether (`exclude_actors`). Remaining shares are then conditional on the identified operators and are all higher; such runs write to their own directory.

**Regions.** Shares are regional: a Kreis with two Gruppen has an HHI of at least 5,000 by construction. Kreis codes follow the current layer; reforms inside the window (Göttingen + Osterode → 03159 in 2016, Eisenach → Wartburgkreis 16063 in 2021) are mapped in `market_cube.KREIS_XWALK`. The run warns about any panel Kreis code the layer lacks. 


## Known limitations

- Winner-take-all: every cell shops at its nearest store. No Huff/MCI
  probabilistic allocation yet.
- Car travel only; no walking or public transport.
- Umsatzpotential is Nielsen's modelled potential, not observed revenue.
- A Land-only solve (no national matrices on disk) drops stores across the
  border; border Kreise lose competitors. Smoke tests only.
- The grid is frozen once built: stores opened later in formerly empty areas
  get a slightly coarser grid there.
