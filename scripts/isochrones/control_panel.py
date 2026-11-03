"""
Notebook control panel: pick the specification, see what it costs, pick the output.

    from scripts.isochrones.control_panel import panel
    gui = panel()                      # renders in the notebook

Two tabs.

**Build** selects the specification — specs, years, land, solve parameters — and
runs pipeline stages. It shows what is already on disk before anything starts,
because the difference between a cached re-run and re-solving the country is
hours, and that difference should be visible on the button, not discovered
afterwards. Stages whose artefacts are missing and expensive (`cells`, `matrix`)
are guarded by an explicit tick.

**Explore** selects the output — a slice of the cube, which figures, which
exports — and renders or writes it.

Every action prints the equivalent CLI command first:

    python -m scripts.run_pipeline --stages cube --specs super_disc --years 2014:2025

That line is the record. A GUI that leaves no trace of what produced a number is
worse than no GUI, so the panel is a front-end for `run_pipeline`, never a second
implementation of it: nothing is computed here that the CLI would not compute the
same way.
"""

from __future__ import annotations


import polars as pl

try:
    import ipywidgets as W
    from IPython.display import display, clear_output
except ImportError as e:                                    # pragma: no cover
    raise ImportError("control_panel needs ipywidgets: pip install ipywidgets") from e

#: one source for the figure list, shared with the desktop GUI and the config
from scripts.runconfig import (FIGURES as FIGURE_CHOICES, LAENDER,   # noqa: E402
                               UNIDENTIFIED)


def _box(title, *children, open_=True):
    a = W.Accordion(children=[W.VBox(list(children))])
    a.set_title(0, title)
    a.selected_index = 0 if open_ else None
    return a


class Panel:
    """Widget front-end for `scripts.run_pipeline`.

    `P` and `F` are injected so the panel can be built and tested without the
    solve modules present; in the notebook the defaults are the real ones.
    """

    def __init__(self, cfg=None, P=None, F=None, regions=None):
        if P is None:
            import scripts.run_pipeline as P
        if F is None:
            import scripts.visualizations.figures as F
        self.P, self.F = P, F
        self.cfg = cfg or P.CFG
        self._regions = regions
        self._cube = None
        self._build()

    # ------------------------------------------------------------------ UI --
    def _build(self):
        P = self.P
        c = self.cfg
        w = dict(style={"description_width": "110px"},
                 layout=W.Layout(width="420px"))

        self.specs = W.SelectMultiple(
            options=[(f"{k}   ({', '.join(v)})", k) for k, v in P.SPECS.items()],
            value=tuple(P.SPECS), description="specs", rows=3, **w)
        self.years = W.IntRangeSlider(value=(min(c.years), max(c.years)),
                                      min=2014, max=2025, description="years",
                                      continuous_update=False, **w)
        self.land = W.Dropdown(options=[("Germany (all)", None)]
                               + [(f"{v} ({k})", k) for k, v in LAENDER.items()],
                               value=c.land, description="restrict to", **w)
        self.osrm = W.Text(value=c.osrm, description="OSRM", **w)
        # names this run's output folder, out/<grid>[_<land>]/<name>; the inputs
        # (snapped stores, grid, matrices, assignments) are shared by every run
        self.name = W.Text(value=c.name, description="run name", **w)

        self.k = W.IntSlider(value=c.k, min=5, max=60, description="k (start)", **w)
        self.cert = W.FloatSlider(value=c.cert_target, min=0.5, max=1.0, step=0.01,
                                  description="cert. target", readout_format=".0%", **w)
        self.iso = W.IntSlider(value=c.iso_min, min=5, max=45,
                               description="isochrone min", **w)
        self.workers = W.IntSlider(value=c.max_workers, min=1, max=16,
                                   description="OSRM workers", **w)

        self.stages = W.SelectMultiple(options=P.ORDER, value=("cube", "figures"),
                                       description="stages", rows=7, **w)
        self.force = W.SelectMultiple(options=P.ORDER, value=(),
                                      description="force", rows=7, **w)
        self.confirm = W.Checkbox(value=False, indent=False,
                                  description="yes, recompute the expensive stages")
        self.status_btn = W.Button(description="Refresh status", icon="refresh")
        self.preview_btn = W.Button(description="Preview", icon="search")
        self.run_btn = W.Button(description="Run", icon="play",
                                button_style="primary")
        self.log = W.Output(layout=W.Layout(border="1px solid #ddd", padding="4px",
                                            max_height="380px", overflow="auto"))

        self.status_btn.on_click(lambda _: self.show_status())
        self.preview_btn.on_click(lambda _: self.preview())
        self.run_btn.on_click(lambda _: self.run())
        for x in (self.specs, self.years, self.land, self.name, self.stages, self.force,
                  self.confirm):
            x.observe(lambda ch: self._guard(), names="value")
        self._guard()

        build = W.VBox([
            W.HBox([W.VBox([self.specs, self.years, self.land, self.osrm, self.name]),
                    W.VBox([self.stages, self.force])]),
            _box("Solve parameters", self.k, self.cert, self.iso, self.workers,
                 open_=False),
            self.confirm,
            W.HBox([self.status_btn, self.preview_btn, self.run_btn]),
        ])

        # ---------------------------------------------------------- explore --
        self.x_spec = W.Dropdown(options=list(P.SPECS), description="spec", **w)
        self.x_year = W.IntSlider(value=max(c.years), min=min(c.years),
                                  max=max(c.years), description="year",
                                  continuous_update=False, **w)
        self.x_level = W.Dropdown(options=["DE", "BL", "KREIS"], value="KREIS",
                                  description="level", **w)
        self.x_basis = W.SelectMultiple(
            options=["pop", "area", "vkfl", "umsatz"], value=("pop", "vkfl"),
            description="basis", rows=4, **w)
        # either/or, not an add-on: the same stores read at two product scopes
        self.x_scope = W.RadioButtons(
            options=[("Include all (food + non-food)", "all"), ("Food only", "food")],
            value="all", description="scope", **w)
        # removed from the market, not reassigned: shares become conditional on
        # the operators the panel can name
        self.x_drop_other = W.Checkbox(
            value=False, indent=False,
            description="exclude unidentified operators ('Sonstige')")
        self.x_actor = W.Dropdown(options=["Gruppe", "Chain"], description="actor", **w)
        self.x_top = W.IntSlider(value=10, min=3, max=25, description="top n", **w)
        self.x_region = W.Text(value="", description="region (AGS)",
                               placeholder="08222, or blank for all", **w)
        self.figs = W.SelectMultiple(options=[(v, k) for k, v in FIGURE_CHOICES.items()],
                                     value=("map_hhi", "trend_hhi"),
                                     description="figures", rows=7, **w)
        self.save_files = W.Checkbox(value=False, indent=False,
                                     description="also write PDF/PNG to outputs/figures")
        self.table_btn = W.Button(description="Show table", icon="table")
        self.fig_btn = W.Button(description="Draw figures", icon="area-chart",
                                button_style="primary")
        self.csv_btn = W.Button(description="Export CSV", icon="download")
        self.table_btn.on_click(lambda _: self.table())
        self.fig_btn.on_click(lambda _: self.draw())
        self.csv_btn.on_click(lambda _: self.export_csv())

        explore = W.VBox([
            W.HBox([W.VBox([self.x_spec, self.x_year, self.x_level, self.x_actor]),
                    W.VBox([self.x_basis, self.x_scope, self.x_drop_other,
                            self.x_top, self.x_region])]),
            self.figs, self.save_files,
            W.HBox([self.table_btn, self.fig_btn, self.csv_btn]),
        ])

        self.tabs = W.Tab(children=[build, explore])
        self.tabs.set_title(0, "Build")
        self.tabs.set_title(1, "Explore")
        self.ui = W.VBox([self.tabs, self.log])

    # -------------------------------------------------------------- state --
    @property
    def config(self):
        """The pipeline `Config` the current widget state describes -- built the
        same way `run_pipeline --config` builds it, so the two cannot drift."""
        return self.runconfig().pipeline_config(self.cfg)

    @property
    def command(self) -> str:
        """The CLI line that runs exactly what `run()` runs."""
        rc = self.runconfig()
        lo, hi = self.years.value
        c = [f"python -m scripts.run_pipeline --stages {','.join(rc.stages)}",
             f"--specs {','.join(rc.specs)}", f"--years {lo}:{hi}",
             f"--bases {','.join(rc.bases)}", f"--scope {rc.scope}"]
        if rc.land:
            c.append(f"--land {rc.land}")
        if rc.exclude_actors:
            c.append("--exclude-actors other")
        if rc.force:
            c.append(f"--force {','.join(rc.force)}")
        return " ".join(c)

    def _pending(self) -> list[str]:
        """Expensive stages this selection would actually compute.

        Dependencies resolve, so wanting `cube` implies everything up to it --
        unless the report is supply-side only (VKFL, Umsatzpotential), which
        needs no solve at all.
        """
        needed = self.P.needed_stages(self.runconfig())
        if not needed:
            return []
        st = self.P.artefacts(self.config, self.specs.value)
        out = []
        for s in self.P.EXPENSIVE:
            if s not in needed:
                continue
            missing = st.filter((pl.col("stage") == s) & ~pl.col("done")).height > 0
            if missing or s in self.force.value:
                out.append(s)
        return out

    def _guard(self):
        try:
            need = self._pending()
        except Exception as e:                       # no artefacts yet, bad path
            need = []
            self.confirm.description = f"status unavailable ({type(e).__name__})"
        self.confirm.disabled = not need
        self.confirm.description = (
            f"yes, compute {', '.join(need)} "
            f"(~{', '.join(self.P.COST[s] for s in need)})" if need
            else "nothing expensive pending")
        self.run_btn.disabled = bool(need) and not self.confirm.value

    # ------------------------------------------------------------- actions --
    def show_status(self):
        with self.log:
            clear_output()
            print(f"snapped : {self.config.snapdir}\n"
                  f"solve   : {self.config.solvedir('<shop type>').parent}\n"
                  f"outputs : {self.config.outdir}\n")
            with pl.Config(tbl_rows=40, fmt_str_lengths=48):
                print(self.P.artefacts(self.config, self.specs.value)
                      .select("stage", "spec", "artefact", "ready", "cost"))

    def preview(self):
        with self.log:
            clear_output()
            print(self.command, "\n")
            st = self.P.artefacts(self.config, self.specs.value)
            todo = st.filter(~pl.col("done"))
            if todo.is_empty():
                print("everything this selection needs is already on disk -- "
                      "the run will only re-read and re-derive.")
            else:
                print("will compute:")
                for r in todo.iter_rows(named=True):
                    print(f"  {r['stage']:<8} {r['spec']:<16} {r['artefact']:<32} "
                          f"~{r['cost']}")
            self._guard()

    def run(self):
        with self.log:
            clear_output()
            print(self.command, "\n")
            # the whole selection travels as a RunConfig -- the same record the
            # desktop GUI and the CLI run from, so scope, bases and exclusions
            # chosen on the Explore tab are what the run builds
            self.state = self.P.run(cfg=self.cfg, rc=self.runconfig())
            self._cube = None
            print("\ndone.")

    # --------------------------------------------------------------- cube --
    def cube(self):
        """The cube for the selected specs, or None with an explanation.

        Every Explore action goes through here, so a missing cube is reported
        once, in words, instead of a traceback per button.
        """
        # cached for the panel's lifetime and reset only by run(): after
        # rebuilding the cube from the CLI, re-create the panel (or set
        # gui._cube = None) to see the new numbers
        if self._cube is None:
            try:
                self._cube = self.P.read_cube(self.config, list(self.specs.value),
                                              self.runconfig().exclude_actors)
            except FileNotFoundError as e:
                print(e)
                return None
        return self._cube

    def bases(self) -> list[str]:
        """Selected bases at the selected product scope."""
        return list(self.runconfig().report_bases())

    def slice(self) -> pl.DataFrame:
        """The shares rows the Explore tab currently selects."""
        got = self.cube()
        if got is None:
            return pl.DataFrame()
        shares, _ = got
        q = shares.filter(pl.col("spec") == self.x_spec.value,
                          pl.col("year") == self.x_year.value,
                          pl.col("level") == self.x_level.value,
                          pl.col("actor_level") == self.x_actor.value,
                          pl.col("basis").is_in(self.bases()))
        if self.x_region.value.strip():
            q = q.filter(pl.col("region") == self.x_region.value.strip())
        return q.filter(pl.col("rank") <= self.x_top.value).sort("region", "basis", "rank")

    def table(self):
        with self.log:
            clear_output()
            got = self.cube()
            if got is None:
                return
            _, hhi = got
            h = hhi.filter(pl.col("spec") == self.x_spec.value,
                           pl.col("year") == self.x_year.value,
                           pl.col("level") == self.x_level.value,
                           pl.col("actor_level") == self.x_actor.value,
                           pl.col("basis").is_in(self.bases()))
            if self.x_region.value.strip():
                h = h.filter(pl.col("region") == self.x_region.value.strip())
            with pl.Config(tbl_rows=30, fmt_str_lengths=40):
                print(h.select("region", "basis", "hhi", "cr1", "cr3", "n_actors",
                               "n_stores", "total").sort("region", "basis"))
                print(self.slice().select("region", "basis", "actor", "rank",
                                          "share", "value"))

    def _gdfs(self):
        if self._regions is None:
            self._regions = self.P.region_layers(self.config)
        return self._regions

    def draw(self):
        import matplotlib.pyplot as plt
        F, P = self.F, self.P
        got = self.cube()
        if got is None:
            with self.log:
                clear_output()
                self.cube()
            return
        shares, hhi = got
        spec, year, actor = self.x_spec.value, self.x_year.value, self.x_actor.value
        bases = self.bases() or ["pop"]
        figdir = self.config.outdir / "figures" / (
            "excl" if self.runconfig().exclude_actors else "")
        with self.log:
            clear_output()
            F.set_style()
            for what in self.figs.value:
                for basis in (bases if what in ("map_hhi", "dist_hhi") else bases[:1]):
                    ax = self._one_figure(what, hhi, shares, spec, year, actor, basis)
                    if ax is None:
                        continue
                    if self.save_files.value:
                        print(F.save(ax.figure,
                                     figdir / P.fig_stem(what, spec, actor, basis, year),
                                     close=False)[0])
                    plt.show()

    def runconfig(self):
        """The current selection as a RunConfig -- same record the desktop GUI
        writes, so a notebook session can be saved and re-run from the CLI."""
        from scripts.runconfig import RunConfig
        lo, hi = self.years.value
        # levels and actor_levels come from the Explore tab's single selection.
        # The cube itself always holds every level and actor level, but figures
        # and CSVs from a run started here cover only the selected ones.
        return RunConfig(name=self.name.value or "baseline",
                         specs=tuple(self.specs.value),
                         years=tuple(range(lo, hi + 1)), land=self.land.value,
                         bases=tuple(self.x_basis.value) or ("pop",),
                         scope=self.x_scope.value,
                         exclude_actors=(UNIDENTIFIED if self.x_drop_other.value else ()),
                         levels=(self.x_level.value,),
                         actor_levels=(self.x_actor.value,),
                         figures=tuple(self.figs.value), year_focus=self.x_year.value,
                         stages=tuple(self.stages.value), force=tuple(self.force.value),
                         osrm=self.osrm.value, k=self.k.value,
                         cert_target=self.cert.value, iso_min=self.iso.value,
                         max_workers=self.workers.value)

    def _one_figure(self, what, hhi, shares, spec, year, actor, basis):
        if what == "grid_maps":
            # the only output that needs the solve itself, not the cube: the grid
            # is cells x mapping x assignment, none of which survives in the cube
            st = getattr(self, "state", {})
            if spec not in st or st.get("cells") is None or not st[spec]["assignments"]:
                print("grid maps need the solve in memory: run the 'assign' stage "
                      "in this session first (Build tab), then draw.")
                return None
            kreis_gdf, _ = self._gdfs()
            self.P.stage_grid_maps(spec, st["cells"], st["mapping"],
                                   st[spec]["assignments"], st[spec]["stores"],
                                   kreis_gdf, self.config, year=year,
                                   alive_by_year=st[spec]["alive"], panel=st["panel"])
            return None
        # one dispatcher for notebook, GUI and batch: a figure that exists in two
        # code paths eventually exists in two versions
        return self.P.draw_figure(what, hhi, shares, spec=spec, year=year,
                                  basis=basis, actor_level=actor,
                                  gdfs=self._gdfs(), cfg=self.config,
                                  rc=self.runconfig())

    def export_csv(self):
        """Delegates to the batch exporter, so notebook CSVs and batch CSVs have
        the same columns, the same filters and the same names."""
        with self.log:
            clear_output()
            got = self.cube()
            if got is None:
                return
            shares, hhi = got
            names = {}
            if self._regions is not None:
                kre, bl = self._regions
                names = {**dict(zip(kre[self.config.kreis_id].astype(str),
                                    kre["LABEL"].astype(str))),
                         **dict(zip(bl["AGS2"].astype(str), bl["LABEL"].astype(str)))}
            self.P.stage_export(shares, hhi, self.config, self.runconfig(), names)


def panel(cfg=None, P=None, F=None, regions=None) -> Panel:
    """Build and display the control panel. Keep the return value: `gui.config`,
    `gui.command` and `gui.slice()` are the reproducible handles on what it did."""
    g = Panel(cfg=cfg, P=P, F=F, regions=regions)
    display(g.ui)
    g.show_status()
    return g