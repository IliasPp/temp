"""
Run configuration — the one declarative record of what an analysis run asked for.

The GUI writes it, the pipeline reads it, and a copy lands next to the outputs as
`run_config.toml`, so any number can be traced to the selection that produced it.
Nothing in the pipeline reads the GUI, and the GUI never edits Python source:
a widget that rewrites `config.py` turns every run into an unversioned mutation
of the code, and two people cannot then run different selections at once.

    from scripts.runconfig import RunConfig
    rc = RunConfig.load("runs/2025_baseline.toml")
    rc = RunConfig(specs=("super_disc",), years=(2024, 2025))
    rc.save("runs/quick.toml")

    python -m scripts.run_pipeline --config runs/quick.toml

Two classes of setting, and they behave differently
---------------------------------------------------
**Solve-determining** — `specs`, `years`, `land`, `shop_types`, and everything
under `solve`. What they cost now that drive times are solved per shop type and
specs are composed from them: a new or redefined spec, a new year or a Land
costs no OSRM work (assignments only, minutes); a new SHOP TYPE costs one solve
(hours); new stores in the panel cost an extension (minutes). `k`,
`cert_target`, `grid` and `cells_per_catchment` only take effect when a matrix or
grid is built from scratch -- cached ones are reused as they are.

**Reporting** — `metrics`, `bases`, `levels`, `actor_levels`, `figures`.
These select what is *written out*, not what is computed. The cube computes every
basis, level and actor level for every spec-year regardless, because once the
assignment exists they cost seconds and a few MB together. Restricting the
computation to the current question would save nothing and would make the next
question a re-run.

So: the reporting fields are a view. Change them freely, re-run
`--stages cube,figures`, and it is done in seconds.
"""

from __future__ import annotations

import dataclasses
import getpass
import socket
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: spec -> the shop types that compete inside it (the "COMPETES_IN" constant).
#: `Shop Type` is derived in cleaning.derive_shop_type: Supermarkt (incl. SB-
#: Warenhaus, Verbrauchermarkt, kleiner/großer Supermarkt), Discount Geschäft,
#: Drogeriemarkt, Getränkemarkt (incl. angeschlossener GAM).
SHOP_TYPES: dict[str, list[str]] = {
    "super": ["Supermarkt"],
    "super_disc": ["Supermarkt", "Discount Geschäft"],
    "super_disc_drog": ["Supermarkt", "Discount Geschäft", "Drogeriemarkt"],
}

SPEC_LABEL = {"super": "Supermarkets",
              "super_disc": "Supermarkets + discounters",
              "super_disc_drog": "Supermarkets + discounters + drugstores"}

#: reporting basis -> (label, the cube basis it maps to)
BASIS_LABEL = {"umsatz": "Umsatzpotential (revenue)",
               "vkfl": "VKFL (sales area)",
               "pop": "Catchment population",
               "area": "Catchment area"}
FOOD_VARIANTS = {"umsatz": "umsatz_food", "vkfl": "vkfl_food"}

#: product scope — an either/or, not an add-on. It picks which column each
#: supply-side measure reads: total floor space and total revenue potential, or
#: the food part of each. Demand-side measures (catchment population, catchment
#: area) have no food variant: a person allocated to a store is not "food
#: population", so under either scope they are the same number.
SCOPE_LABEL = {"all": "Include all (food + non-food)", "food": "Food only"}

LEVEL_LABEL = {"DE": "Germany-wide", "BL": "By Bundesland", "KREIS": "By Kreis"}
METRICS = {"hhi": "HHI (+CR1/CR3/CR5)", "share": "Market shares"}
ACTORS = {"Gruppe": "Group (ownership: Schwarz)",
          "Chain": "Chain (banner: Lidl)"}

#: Actor labels that mean "not identified" rather than a real operator. Panels
#: differ, so this is a list to edit, not a rule.
UNIDENTIFIED = ("UNKNOWN", "Sonstige", "Sonstiger", "Sonstiges", "Übrige",
                "Uebrige", "Other", "Unabhängige", "Unabhaengige", "n.a.", "NA")

FIGURES = {"map_hhi": "HHI choropleth (Kreis)",
           "map_share": "Leader's share, % (Kreis)",
           "map_leader": "Who leads each Kreis",
           "heat_shares": "Share heatmap (actor x year)",
           "trend_hhi": "National HHI trend",
           "trend_shares": "National share trends",
           "dist_hhi": "Kreis HHI distribution",
           "scatter_basis": "Demand vs supply-side HHI",
           "bars_bl": "HHI by Bundesland, first/last",
           "grid_maps": "Grid maps (Mannheim, Hamburg)",
           "map_catchments": "Catchment polygons per Kreis"}

LAENDER = {"01": "Schleswig-Holstein", "02": "Hamburg", "03": "Niedersachsen",
           "04": "Bremen", "05": "Nordrhein-Westfalen", "06": "Hessen",
           "07": "Rheinland-Pfalz", "08": "Baden-Württemberg", "09": "Bayern",
           "10": "Saarland", "11": "Berlin", "12": "Brandenburg",
           "13": "Mecklenburg-Vorpommern", "14": "Sachsen",
           "15": "Sachsen-Anhalt", "16": "Thüringen"}

YEAR_MIN, YEAR_MAX = 2014, 2025
STAGES = ["stores", "cells", "matrix", "assign", "regions", "cube", "figures"]


@dataclass
class RunConfig:
    # --- what to solve ---------------------------------------------------
    specs: tuple[str, ...] = tuple(SHOP_TYPES)
    shop_types: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in SHOP_TYPES.items()})
    years: tuple[int, ...] = tuple(range(YEAR_MIN, YEAR_MAX + 1))
    year_mode: str = "range"          # "range" | "list" -- input convenience only
    #: report one Bundesland. Different from levels=("BL",), which reports every
    #: Land. Uses the national matrices when they exist (exact at the border);
    #: otherwise solves the Land alone, and border Kreise lose competitors from
    #: across it -- a smoke test, never a published number.
    land: str | None = None

    # --- what to report --------------------------------------------------
    metrics: tuple[str, ...] = ("hhi", "share")
    bases: tuple[str, ...] = ("pop", "area", "vkfl", "umsatz")
    scope: str = "all"                # "all" | "food" -- see SCOPE_LABEL
    levels: tuple[str, ...] = ("DE", "BL", "KREIS")
    actor_levels: tuple[str, ...] = ("Gruppe", "Chain")
    year_focus: int | None = None     # year for single-year figures; None = latest

    # --- outputs ---------------------------------------------------------
    figures: tuple[str, ...] = ("map_hhi", "map_share", "map_leader", "heat_shares",
                                "trend_hhi", "trend_shares", "dist_hhi",
                                "scatter_basis", "bars_bl")
    #: Actors dropped from the market before shares are computed. Empty keeps
    #: every store. Matching is case-insensitive; see UNIDENTIFIED.
    exclude_actors: tuple[str, ...] = ()
    formats: tuple[str, ...] = ("pdf", "png")
    export_csv: bool = True
    #: per-spec, per-year GeoTIFF of the winning store (~30-80 MB each)
    write_rasters: bool = False

    # --- execution -------------------------------------------------------
    stages: tuple[str, ...] = ("cube", "figures")
    force: tuple[str, ...] = ()
    osrm: str = "http://localhost:5000"
    #: the OSRM build the snaps belong to; bump it when the extract is rebuilt
    network: str = "germany-latest"
    cache: str | None = None
    grid: str = "de_100m"
    k: int = 20
    cert_target: float = 0.90
    iso_min: int = 20
    max_workers: int = 8
    cells_per_catchment: int | None = None

    # --- provenance ------------------------------------------------------
    #: names the output folder (outputs/catchments/<grid>/<name>). Stored in the
    #: file as [run] name; it is NOT derived from the file name when loading.
    #: The desktop GUI writes the file stem here when it saves.
    name: str = "baseline"
    note: str = ""
    created: str = ""
    created_by: str = ""

    # ------------------------------------------------------------ derive --
    def report_bases(self) -> tuple[str, ...]:
        """The cube columns this run reports: the selected bases, read at the
        selected product scope.

        Under `scope="food"` the supply-side measures switch to their food-only
        column; `pop` and `area` pass through unchanged because they have no food
        variant. The cube holds both columns either way — scope is a view, not a
        recomputation.
        """
        if self.scope == "food":
            return tuple(FOOD_VARIANTS.get(b, b) for b in self.bases)
        return tuple(self.bases)

    #: bases that need the catchment solve; everything else is a store attribute
    DEMAND_BASES = ("pop", "area")

    def needs_solve(self) -> bool:
        """Does this report need the catchment solve at all?

        Sales area and revenue potential are store attributes: shares and HHI on
        them come from the panel alone. Only catchment population and catchment
        area need cells, matrices and assignments -- and so does a grid map,
        which draws the assignment itself. Asking for hours of OSRM work to
        report a VKFL share would be absurd, so this is checked before anything
        expensive is scheduled.
        """
        return (bool(set(self.report_bases()) & set(self.DEMAND_BASES))
                or bool({"grid_maps", "map_catchments"} & set(self.figures))
                or self.write_rasters)

    def competes_in(self) -> dict[str, list[str]]:
        """spec -> shop types, restricted to the selected specs."""
        return {s: list(self.shop_types[s]) for s in self.specs}

    def validate(self) -> list[str]:
        """Problems, in plain words. Empty list means the run can start."""
        p = []
        if not self.specs:
            p.append("select at least one store specification")
        for s in self.specs:
            if s not in self.shop_types or not self.shop_types[s]:
                p.append(f"spec {s!r} has no shop types")
        if not self.years:
            p.append("select at least one year")
        if any(not YEAR_MIN <= y <= YEAR_MAX for y in self.years):
            p.append(f"years must lie in {YEAR_MIN}-{YEAR_MAX}")
        if not self.metrics:
            p.append("select at least one metric")
        if not self.bases:
            p.append("select at least one method (Umsatzpotential, VKFL, "
                     "population, catchment area)")
        if self.scope not in SCOPE_LABEL:
            p.append(f"product scope must be one of {tuple(SCOPE_LABEL)}")
        if self.scope == "food" and not (set(self.bases) & set(FOOD_VARIANTS)):
            p.append("note: 'Food only' changes VKFL and Umsatzpotential; with "
                     "neither selected it has no effect on the numbers")
        if not self.levels:
            p.append("select at least one geographic coverage level")
        if not self.actor_levels:
            p.append("select a breakdown: Group, Chain, or both")
        if self.land and self.land not in LAENDER:
            p.append(f"unknown Bundesland code {self.land!r}")
        bad = set(self.stages) - set(STAGES)
        if bad:
            p.append(f"unknown stage(s): {', '.join(sorted(bad))}")
        if self.land:
            p.append("note: a Bundesland run reuses the national matrices when they "
                     "exist (exact at the border); without them it solves the Land "
                     "alone, and Kreise at the Land border lose competitors from "
                     "across it — a smoke test, not a published number")
            if "DE" in self.levels:
                p.append("note: a Bundesland run has no Germany total; the 'DE' "
                         "figures and tables show the Land's own total (level BL)")
        if self.exclude_actors:
            p.append("note: excluded actors are removed from the market entirely — "
                     "every share and HHI below is conditional on the identified "
                     "operators, and is higher than the unconditional one")
        if ({"map_hhi", "map_share", "map_leader"} & set(self.figures)
                and "KREIS" not in self.levels):
            p.append("note: the Kreis maps need Kreis coverage; add it or drop "
                     "those figures")
        if "bars_bl" in self.figures and "BL" not in self.levels:
            p.append("note: the Bundesland bars need Bundesland coverage")
        return p

    def blocking(self) -> list[str]:
        return [x for x in self.validate() if not x.startswith("note:")]

    def pipeline_config(self, base=None):
        """Build the pipeline `Config` this run needs.

        Import is local so the GUI and the config file stay usable without the
        solve modules (and without OSRM) present.
        """
        import scripts.run_pipeline as P
        base = base or P.CFG
        # Only these fields travel from the TOML into Config; the isochrone
        # frame is written for the last selected year only.
        kw = dict(name=self.name, years=tuple(sorted(self.years)),
                  land=self.land, osrm=self.osrm,
                  grid=self.grid, network=self.network, k=self.k,
                  write_rasters=self.write_rasters, cert_target=self.cert_target,
                  iso_min=self.iso_min, max_workers=self.max_workers,
                  cells_per_catchment=self.cells_per_catchment,
                  iso_years=(max(self.years),))
        if self.cache:
            kw["cache"] = Path(self.cache)
        return dataclasses.replace(base, **kw)

    # -------------------------------------------------------------- I/O ---
    def to_dict(self) -> dict:
        return {
            "run": {"name": self.name, "note": self.note,
                    "created": self.created, "created_by": self.created_by},
            # shop_types carries *every* spec, not only the selected ones: it is
            # a definition (which formats compete), while `specs` is the
            # selection. Writing only the selected ones would make
            # `--specs super` on top of this file fail with no shop types.
            "spec": {"specs": list(self.specs),
                     "shop_types": {k: list(v) for k, v in self.shop_types.items()},
                     "land": self.land or ""},
            "years": {"mode": self.year_mode, "values": list(sorted(self.years))},
            "report": {"metrics": list(self.metrics), "bases": list(self.bases),
                       "scope": self.scope, "exclude_actors": list(self.exclude_actors),
                       "levels": list(self.levels),
                       "actor_levels": list(self.actor_levels),
                       "year_focus": self.year_focus or 0},
            "output": {"figures": list(self.figures), "formats": list(self.formats),
                       "export_csv": self.export_csv,
                       "write_rasters": self.write_rasters},
            "solve": {"stages": list(self.stages), "force": list(self.force),
                      "osrm": self.osrm, "network": self.network,
                      "cache": self.cache or "", "grid": self.grid,
                      "k": self.k, "cert_target": self.cert_target,
                      "iso_min": self.iso_min, "max_workers": self.max_workers,
                      "cells_per_catchment": self.cells_per_catchment or 0},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        g = lambda s, k, dflt: d.get(s, {}).get(k, dflt)          # noqa: E731
        st = {**{k: list(v) for k, v in SHOP_TYPES.items()},
              **(g("spec", "shop_types", None) or {})}      # partial files still work
        return cls(
            specs=tuple(g("spec", "specs", tuple(SHOP_TYPES))),
            shop_types={k: list(v) for k, v in st.items()},
            land=(g("spec", "land", "") or None),
            years=tuple(g("years", "values", tuple(range(YEAR_MIN, YEAR_MAX + 1)))),
            year_mode=g("years", "mode", "range"),
            metrics=tuple(g("report", "metrics", ("hhi", "share"))),
            bases=tuple(g("report", "bases", ("pop", "area", "vkfl", "umsatz"))),
            # legacy files carry `food = true/false` (an add-on flag) instead of
            # `scope`; the closest single choice for either value is "all"
            scope=g("report", "scope", "all"),
            exclude_actors=tuple(g("report", "exclude_actors", ())),
            levels=tuple(g("report", "levels", ("DE", "BL", "KREIS"))),
            actor_levels=tuple(g("report", "actor_levels", ("Gruppe", "Chain"))),
            year_focus=(g("report", "year_focus", 0) or None),
            figures=tuple(g("output", "figures", ())),
            formats=tuple(g("output", "formats", ("pdf", "png"))),
            export_csv=bool(g("output", "export_csv", True)),
            write_rasters=bool(g("output", "write_rasters", False)),
            stages=tuple(g("solve", "stages", ("cube", "figures"))),
            force=tuple(g("solve", "force", ())),
            osrm=g("solve", "osrm", "http://localhost:5000"),
            network=g("solve", "network", "germany-latest"),
            cache=(g("solve", "cache", "") or None),
            grid=g("solve", "grid", "de_100m"),
            k=int(g("solve", "k", 20)),
            cert_target=float(g("solve", "cert_target", 0.90)),
            iso_min=int(g("solve", "iso_min", 20)),
            max_workers=int(g("solve", "max_workers", 8)),
            cells_per_catchment=(int(g("solve", "cells_per_catchment", 0)) or None),
            name=g("run", "name", "baseline"), note=g("run", "note", ""),
            created=g("run", "created", ""), created_by=g("run", "created_by", ""),
        )

    def save(self, path: str | Path, *, stamp: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if stamp:
            self.created = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                self.created_by = f"{getpass.getuser()}@{socket.gethostname()}"
            except Exception:
                self.created_by = "unknown"
        path.write_text(_toml(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with open(path, "rb") as fh:
            return cls.from_dict(tomllib.load(fh))

    def summary(self) -> str:
        yrs = (f"{min(self.years)}–{max(self.years)}"
               if self.year_mode == "range" else ", ".join(map(str, sorted(self.years))))
        return "\n".join([
            f"specs      {', '.join(self.specs)}",
            f"years      {yrs}  ({len(self.years)} snapshots)",
            f"coverage   {', '.join(LEVEL_LABEL[l] for l in self.levels)}"
            + (f"   [reported for {LAENDER[self.land]}]" if self.land else ""),
            f"metrics    {', '.join(self.metrics)}",
            f"scope      {SCOPE_LABEL[self.scope]}"
            + (f"   excluding: {', '.join(self.exclude_actors)}"
               if self.exclude_actors else ""),
            f"method     {', '.join(self.report_bases())}",
            f"breakdown  {', '.join(self.actor_levels)}",
            f"outputs    {len(self.figures)} figure type(s), "
            f"{'CSV + ' if self.export_csv else ''}{'/'.join(self.formats)}",
            f"stages     {', '.join(self.stages)}"
            + (f"   force: {', '.join(self.force)}" if self.force else ""),
        ])

    def command(self, config_path: str | Path) -> str:
        return f"python -m scripts.run_pipeline --config {config_path}"


# --------------------------------------------------------------------------
# minimal TOML writer: enough for this schema (str, bool, int, float, list,
# and one level of table). Reading uses stdlib tomllib.
# --------------------------------------------------------------------------

def _val(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml(d: dict) -> str:
    out = ["# Generated by the market-structure GUI. Edit by hand or in the GUI;",
           "# run with: python -m scripts.run_pipeline --config <this file>", ""]
    for section, body in d.items():
        out.append(f"[{section}]")
        sub = {}
        for k, v in body.items():
            # (the next line has no effect -- body already holds k; the if/else
            #  below does the work: nested dicts become [section.key] tables)
            (sub if isinstance(v, dict) else body).setdefault(k, v)
            if isinstance(v, dict):
                sub[k] = v
            else:
                out.append(f"{k} = {_val(v)}")
        for k, v in sub.items():
            out.append(f"\n[{section}.{k}]")
            for kk, vv in v.items():
                out.append(f"{kk} = {_val(vv)}")
        out.append("")
    return "\n".join(out)