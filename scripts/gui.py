"""
Configuration GUI — pick the run, write the config, launch the pipeline.

    python -m scripts.gui
    python -m scripts.gui runs/baseline.toml        # open an existing config
    python -m scripts.run_pipeline --gui

Tkinter, so it runs on the ZEW workstation with nothing installed. The window
writes a `RunConfig` TOML and then shells out to

    python -m scripts.run_pipeline --config <that file>

streaming the run's output into the log pane. That indirection is the design:
the GUI produces a *file*, never a hidden in-memory state, so the same run can be
repeated from the command line, committed, mailed to a co-author, or diffed
against last month's — and a crashed window cannot take a six-hour solve with it.

The GUI never edits Python source. A widget that rewrites `config.py` turns every
run into an unversioned mutation of the code, and two people can no longer run
different selections at the same time.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from scripts.runconfig import (ACTORS, BASIS_LABEL, FIGURES, LAENDER, LEVEL_LABEL,
                               METRICS, SCOPE_LABEL, SPEC_LABEL, STAGES, UNIDENTIFIED,
                               YEAR_MAX, YEAR_MIN, RunConfig)

ROOT = Path(__file__).resolve().parent.parent      # project root (.../edeka_project_2)
PAD = dict(padx=6, pady=3)


class CheckGroup(ttk.LabelFrame):
    """A titled group of checkboxes over a {key: label} mapping."""

    def __init__(self, parent, title, options: dict, initial=(), **kw):
        super().__init__(parent, text=title, **kw)
        self.vars = {}
        self.rows = len(options)          # where extra widgets may start
        for i, (k, lab) in enumerate(options.items()):
            v = tk.BooleanVar(value=k in initial)
            ttk.Checkbutton(self, text=lab, variable=v).grid(row=i, column=0,
                                                             sticky="w", **PAD)
            self.vars[k] = v

    @property
    def value(self) -> tuple:
        return tuple(k for k, v in self.vars.items() if v.get())

    @value.setter
    def value(self, keys):
        for k, v in self.vars.items():
            v.set(k in keys)


class YearPicker(ttk.LabelFrame):
    """Range (every year between two bounds) or an explicit list of snapshots.

    Both produce a set of years; the distinction is only how you type it. Each
    selected year is a separate cross-section — nothing in the pipeline pools
    years, so 'range' does not mean an average over the range.
    """

    def __init__(self, parent, years=(YEAR_MIN, YEAR_MAX), mode="range"):
        super().__init__(parent, text="Years (one snapshot per year)")
        self.mode = tk.StringVar(value=mode)
        ttk.Radiobutton(self, text="Range", variable=self.mode, value="range",
                        command=self._sync).grid(row=0, column=0, sticky="w", **PAD)
        self.lo = tk.IntVar(value=min(years))
        self.hi = tk.IntVar(value=max(years))
        self.s_lo = ttk.Spinbox(self, from_=YEAR_MIN, to=YEAR_MAX, width=6,
                                textvariable=self.lo)
        self.s_hi = ttk.Spinbox(self, from_=YEAR_MIN, to=YEAR_MAX, width=6,
                                textvariable=self.hi)
        self.s_lo.grid(row=0, column=1, **PAD)
        ttk.Label(self, text="to").grid(row=0, column=2)
        self.s_hi.grid(row=0, column=3, **PAD)

        ttk.Radiobutton(self, text="Single years", variable=self.mode, value="list",
                        command=self._sync).grid(row=1, column=0, sticky="nw", **PAD)
        self.box = tk.Listbox(self, selectmode="extended", height=12, width=8,
                              exportselection=False)
        for y in range(YEAR_MIN, YEAR_MAX + 1):
            self.box.insert("end", y)
        self.box.grid(row=1, column=1, columnspan=3, sticky="w", **PAD)
        for y in years:
            self.box.selection_set(y - YEAR_MIN)
        self._sync()

    def _sync(self):
        listing = self.mode.get() == "list"
        self.box.configure(state="normal" if listing else "disabled")
        for s in (self.s_lo, self.s_hi):
            s.configure(state="disabled" if listing else "normal")

    @property
    def value(self) -> tuple[int, ...]:
        if self.mode.get() == "range":
            lo, hi = sorted((self.lo.get(), self.hi.get()))
            return tuple(range(lo, hi + 1))
        return tuple(YEAR_MIN + i for i in self.box.curselection())

    def set(self, years, mode):
        self.mode.set(mode)
        # a disabled Listbox ignores selection commands, so enable it first and
        # let _sync() put the state back
        self.box.configure(state="normal")
        if years:
            self.lo.set(min(years))
            self.hi.set(max(years))
            self.box.selection_clear(0, "end")
            for y in years:
                if YEAR_MIN <= y <= YEAR_MAX:
                    self.box.selection_set(y - YEAR_MIN)
        self._sync()


class App(ttk.Frame):
    def __init__(self, master, rc: RunConfig | None = None):
        super().__init__(master, padding=8)
        self.grid(sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.proc = None
        self.q: queue.Queue[str] = queue.Queue()
        self.config_path = tk.StringVar(value=str(ROOT / "runs" / "baseline.toml"))
        self._build(rc or RunConfig())
        self._pump()

    # ------------------------------------------------------------ layout --
    def _build(self, rc: RunConfig):
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew")

        self.g_spec = CheckGroup(top, "Specification", SPEC_LABEL, rc.specs)
        # product scope is an either/or, so radios: "all" and "food only" are two
        # readings of the same stores, not two things to add together
        r = self.g_spec.rows
        ttk.Separator(self.g_spec, orient="horizontal").grid(
            row=r, column=0, sticky="ew", padx=6, pady=(8, 2))
        ttk.Label(self.g_spec, text="Product scope").grid(row=r + 1, column=0,
                                                          sticky="w", **PAD)
        self.scope = tk.StringVar(value=rc.scope)
        for i, (k, lab) in enumerate(SCOPE_LABEL.items()):
            ttk.Radiobutton(self.g_spec, text=lab, variable=self.scope, value=k
                            ).grid(row=r + 2 + i, column=0, sticky="w", **PAD)
        self.g_years = YearPicker(top, rc.years, rc.year_mode)
        self.g_metric = CheckGroup(top, "Metrics", METRICS, rc.metrics)
        self.g_spec.grid(row=0, column=0, sticky="nsew", **PAD)
        self.g_years.grid(row=0, column=1, rowspan=2, sticky="nsew", **PAD)
        self.g_metric.grid(row=1, column=0, sticky="nsew", **PAD)

        self.g_basis = CheckGroup(top, "Method (share basis)", BASIS_LABEL, rc.bases)
        ttk.Label(self.g_basis, foreground="#555", wraplength=330,
                  text="Product scope applies to Umsatzpotential and VKFL; "
                       "catchment population and area have no food variant."
                  ).grid(row=self.g_basis.rows, column=0, sticky="w", **PAD)
        self.g_basis.grid(row=0, column=2, sticky="nsew", **PAD)

        self.g_level = CheckGroup(top, "Geographic coverage (report at)",
                                  LEVEL_LABEL, rc.levels)
        self.land = tk.StringVar(value=rc.land or "")
        ttk.Label(self.g_level, text="restrict solve to:").grid(
            row=self.g_level.rows, column=0, sticky="w", **PAD)
        ttk.Combobox(self.g_level, textvariable=self.land, width=22, state="readonly",
                     values=[""] + [f"{k} {v}" for k, v in LAENDER.items()]
                     ).grid(row=self.g_level.rows + 1, column=0, sticky="w", **PAD)
        self.g_level.grid(row=1, column=2, sticky="nsew", **PAD)

        self.g_actor = CheckGroup(top, "Breakdown", ACTORS, rc.actor_levels)
        self.drop_other = tk.BooleanVar(value=bool(rc.exclude_actors))
        r = self.g_actor.rows
        ttk.Separator(self.g_actor, orient="horizontal").grid(
            row=r, column=0, sticky="ew", padx=6, pady=(8, 2))
        ttk.Checkbutton(self.g_actor, variable=self.drop_other,
                        text="exclude unidentified ('Sonstige')").grid(
            row=r + 1, column=0, sticky="w", **PAD)
        ttk.Label(self.g_actor, foreground="#555", wraplength=230,
                  text="Removed from the market, not reassigned: every share and "
                       "HHI is then conditional on the identified operators."
                  ).grid(row=r + 2, column=0, sticky="w", **PAD)
        self.g_actor.grid(row=0, column=3, sticky="nsew", **PAD)
        self.g_fig = CheckGroup(top, "Figures", FIGURES, rc.figures)
        self.csv = tk.BooleanVar(value=rc.export_csv)
        ttk.Checkbutton(self.g_fig, text="export CSV tables",
                        variable=self.csv).grid(row=self.g_fig.rows + 1, column=0,
                                                sticky="w", **PAD)
        self.rasters = tk.BooleanVar(value=rc.write_rasters)
        ttk.Checkbutton(self.g_fig, variable=self.rasters,
                        text="write per-year GeoTIFFs").grid(
            row=self.g_fig.rows + 2, column=0, sticky="w", **PAD)
        self.g_fig.grid(row=1, column=3, rowspan=2, sticky="nsew", **PAD)

        self.g_stage = CheckGroup(top, "Stages to run",
                                  {s: s for s in STAGES}, rc.stages)
        self.g_force = CheckGroup(top, "Force recompute",
                                  {s: s for s in STAGES}, rc.force)
        self.g_stage.grid(row=2, column=0, sticky="nsew", **PAD)
        self.g_force.grid(row=2, column=1, sticky="nsew", **PAD)

        adv = ttk.LabelFrame(top, text="Solve parameters")
        adv.grid(row=2, column=2, sticky="nsew", **PAD)
        self.fields = {}
        for i, (key, lab, val) in enumerate([
                ("osrm", "OSRM URL", rc.osrm), ("network", "road network", rc.network),
                ("grid", "grid namespace", rc.grid),
                ("cache", "snap cache dir", rc.cache or ""),
                ("k", "k (start)", rc.k), ("cert_target", "cert. target", rc.cert_target),
                ("iso_min", "isochrone min", rc.iso_min),
                ("max_workers", "OSRM workers", rc.max_workers)]):
            ttk.Label(adv, text=lab).grid(row=i, column=0, sticky="w", **PAD)
            v = tk.StringVar(value=str(val))
            ttk.Entry(adv, textvariable=v, width=24).grid(row=i, column=1, **PAD)
            self.fields[key] = v

        # ------------------------------------------------------- actions --
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="ew", pady=(6, 2))
        ttk.Label(bar, text="config file").pack(side="left")
        ttk.Entry(bar, textvariable=self.config_path, width=58).pack(side="left", padx=4)
        for text, cmd in (("Load…", self.load), ("Save", self.save),
                          ("Status", self.status), ("Preview", self.preview)):
            ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=2)
        self.run_btn = ttk.Button(bar, text="Run", command=self.start)
        self.run_btn.pack(side="left", padx=(10, 2))
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")

        self.log = tk.Text(self, height=18, wrap="none", font=("Consolas", 9))
        sb = ttk.Scrollbar(self, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.grid(row=2, column=0, sticky="nsew")
        sb.grid(row=2, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

    # ------------------------------------------------------------- state --
    def config(self) -> RunConfig:
        """The RunConfig the widgets currently describe."""
        f = self.fields
        num = lambda k, cast, d: _cast(f[k].get(), cast, d)          # noqa: E731
        return RunConfig(
            specs=self.g_spec.value, years=self.g_years.value,
            year_mode=self.g_years.mode.get(),
            land=(self.land.get().split()[0] if self.land.get().strip() else None),
            metrics=self.g_metric.value, bases=self.g_basis.value,
            scope=self.scope.get(), levels=self.g_level.value,
            actor_levels=self.g_actor.value, figures=self.g_fig.value,
            exclude_actors=(UNIDENTIFIED if self.drop_other.get() else ()),
            export_csv=self.csv.get(), write_rasters=self.rasters.get(),
            stages=self.g_stage.value,
            force=self.g_force.value, osrm=f["osrm"].get().strip(),
            grid=f["grid"].get().strip() or "de_100m",
            network=f["network"].get().strip() or "germany-latest",
            cache=(f["cache"].get().strip() or None),
            k=num("k", int, 20), cert_target=num("cert_target", float, 0.90),
            iso_min=num("iso_min", int, 20), max_workers=num("max_workers", int, 8),
            name=Path(self.config_path.get()).stem)

    def apply(self, rc: RunConfig):
        self.g_spec.value = rc.specs
        self.g_years.set(rc.years, rc.year_mode)
        self.g_metric.value = rc.metrics
        self.g_basis.value = rc.bases
        self.scope.set(rc.scope)
        self.g_level.value = rc.levels
        self.land.set(f"{rc.land} {LAENDER[rc.land]}" if rc.land else "")
        self.g_actor.value = rc.actor_levels
        self.drop_other.set(bool(rc.exclude_actors))
        self.g_fig.value = rc.figures
        self.csv.set(rc.export_csv)
        self.rasters.set(rc.write_rasters)
        self.g_stage.value = rc.stages
        self.g_force.value = rc.force
        for k, v in self.fields.items():
            val = getattr(rc, k, "")
            v.set("" if val is None else str(val))

    # ----------------------------------------------------------- actions --
    def say(self, *msg):
        self.log.insert("end", " ".join(str(m) for m in msg) + "\n")
        self.log.see("end")

    def _check(self) -> RunConfig | None:
        rc = self.config()
        for p in rc.validate():
            self.say(("  " if p.startswith("note:") else "  ERROR: ") + p)
        if rc.blocking():
            messagebox.showerror("Incomplete selection", "\n".join(rc.blocking()))
            return None
        return rc

    def load(self):
        p = filedialog.askopenfilename(title="Open run config",
                                       filetypes=[("TOML", "*.toml")],
                                       initialdir=str(ROOT / "runs"))
        if p:
            self.apply(RunConfig.load(p))
            self.config_path.set(p)
            self.say(f"loaded {p}")

    def save(self) -> Path | None:
        rc = self._check()
        if rc is None:
            return None
        p = Path(self.config_path.get())
        rc.save(p)
        self.say(f"saved {p}\n{rc.summary()}\n\n{rc.command(p)}")
        return p

    def status(self):
        """What already exists on disk, so an expensive run is a choice."""
        rc = self.config()
        try:
            import polars as pl
            import scripts.run_pipeline as P
            with pl.Config(tbl_rows=50, fmt_str_lengths=44):
                self.say(str(P.artefacts(rc.pipeline_config(), rc.specs)))
        except Exception as e:
            self.say(f"status unavailable ({type(e).__name__}: {e})")

    def preview(self):
        rc = self._check()
        if rc is None:
            return
        self.say("\n" + rc.summary())
        pend = self._pending(rc)
        if pend:
            self.say(f"\nwould COMPUTE: {', '.join(pend)} — this is the expensive part")
        elif not rc.needs_solve():
            self.say("\nsupply-side report: shares come from the panel, so no catchment "
                     "solve is needed. Seconds.")
        else:
            self.say("\nnothing expensive pending; everything else is seconds")
        self.say(rc.command(self.config_path.get()))

    def _pending(self, rc: RunConfig) -> list[str]:
        """Expensive stages this selection would actually compute."""
        try:
            import polars as pl
            import scripts.run_pipeline as P
            needed = P.needed_stages(rc)
            if not needed:
                return []
            st = P.artefacts(rc.pipeline_config(), rc.specs)
            return [s for s in P.EXPENSIVE if s in needed
                    and (st.filter((pl.col("stage") == s) & ~pl.col("done")).height
                         or s in rc.force)]
        except Exception:
            return []

    def start(self):
        if self.proc:
            return
        p = self.save()
        if p is None:
            return
        rc = self.config()
        pend = self._pending(rc)
        if pend and not messagebox.askyesno(
                "Expensive stages",
                f"This run will compute: {', '.join(pend)}.\n\n"
                "That is hours of OSRM work, not seconds. It checkpoints, so it can "
                "be stopped and resumed.\n\nStart anyway?"):
            return
        cmd = [sys.executable, "-m", "scripts.run_pipeline", "--config", str(p)]
        self.say("\n$ " + " ".join(cmd) + "\n")
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True).start()
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

    def _reader(self):
        for line in self.proc.stdout:
            self.q.put(line.rstrip("\n"))
        self.q.put(f"\n--- finished, exit code {self.proc.wait()} ---")
        self.q.put(None)

    def _pump(self):
        while True:
            try:
                line = self.q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self.proc = None
                self.run_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
            else:
                self.say(line)
        self.after(150, self._pump)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.say("--- stop requested; checkpointed stages resume where they were ---")


def _cast(s, cast, default):
    try:
        return cast(s)
    except (TypeError, ValueError):
        return default


def main(config: str | None = None):
    root = tk.Tk()
    root.title("Market structure — run configuration")
    root.geometry("1400x870")
    root.minsize(1200, 700)
    rc = RunConfig.load(config) if config else RunConfig()
    App(root, rc)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
