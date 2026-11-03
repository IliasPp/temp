"""GUI test under a virtual display: widgets -> RunConfig -> TOML -> CLI.

Builds the real window, pokes the widgets, and checks that what the user sees
is what the config file says and what the pipeline would be asked to do. The
pipeline itself is never started here; `start()` is exercised with a stub
subprocess so the streaming path is covered without a six-hour solve.
"""
import subprocess
import sys
import time

import pytest

tk = pytest.importorskip("tkinter")
from scripts.runconfig import RunConfig  # noqa: E402
import scripts.gui as G  # noqa: E402


def _display():
    try:
        tk.Tk().destroy()
        return True
    except tk.TclError:
        return False


@pytest.mark.skipif(not _display(), reason="no display for Tk")
def test_gui_widgets_round_trip(tmp_path):
    root = tk.Tk()
    root.withdraw()
    app = G.App(root, RunConfig())
    root.update()

    # ---- defaults survive the round trip through the widgets --------------------
    rc = app.config()
    assert rc.specs == ("super", "super_disc", "super_disc_drog"), rc.specs
    assert rc.years == tuple(range(2014, 2026))
    assert set(rc.bases) == {"pop", "area", "vkfl", "umsatz"} and rc.scope == "all"
    assert rc.levels == ("DE", "BL", "KREIS") and rc.actor_levels == ("Gruppe", "Chain")
    assert rc.blocking() == []

    # ---- a narrowed selection ---------------------------------------------------
    app.g_spec.value = ("super_disc",)
    app.g_years.mode.set("list")
    app.g_years.set((2019, 2022, 2025), "list")
    app.g_metric.value = ("hhi",)
    app.g_basis.value = ("vkfl", "umsatz")
    app.scope.set("food")
    app.g_level.value = ("KREIS",)
    app.g_actor.value = ("Gruppe",)
    app.g_fig.value = ("map_hhi", "dist_hhi")
    app.g_stage.value = ("cube", "figures")
    app.fields["k"].set("30")
    app.fields["osrm"].set("http://10.0.0.5:5000")
    app.land.set("09 Bayern")
    root.update()

    rc = app.config()
    assert rc.specs == ("super_disc",) and rc.years == (2019, 2022, 2025)
    assert rc.year_mode == "list" and rc.metrics == ("hhi",)
    assert set(rc.bases) == {"vkfl", "umsatz"} and rc.scope == "food"
    # scope is either/or: the supply measures switch column, they are not added to
    assert set(rc.report_bases()) == {"vkfl_food", "umsatz_food"}, rc.report_bases()
    assert set(RunConfig(bases=("pop", "vkfl"), scope="food").report_bases()) == \
        {"pop", "vkfl_food"}, "demand-side bases have no food variant"
    assert rc.levels == ("KREIS",) and rc.land == "09" and rc.k == 30
    assert rc.osrm == "http://10.0.0.5:5000"
    assert rc.competes_in() == {"super_disc": ["Supermarkt", "Discount Geschäft"]}
    notes = [x for x in rc.validate() if x.startswith("note:")]
    assert any("border" in n for n in notes), notes          # Land restriction warned
    assert rc.blocking() == []

    # ---- write, reload into a fresh window, compare ----------------------------
    app.config_path.set(str(tmp_path / "narrow.toml"))
    p = app.save()
    assert p and p.exists()
    app2 = G.App(tk.Toplevel(root), RunConfig.load(p))
    root.update()
    rc2 = app2.config()
    for f in ("specs", "years", "year_mode", "metrics", "bases", "scope", "levels",
              "actor_levels", "figures", "export_csv", "stages", "land", "k", "osrm"):
        assert getattr(rc, f) == getattr(rc2, f), (f, getattr(rc, f), getattr(rc2, f))
    print("widget -> config -> file -> widget: identical")

    # ---- blocking selection is refused ------------------------------------------
    app.g_spec.value = ()
    assert "select at least one store specification" in app.config().blocking()
    app.g_spec.value = ("super_disc",)

    # ---- streaming: start() with a stub child ------------------------------------
    G.messagebox.askyesno = lambda *a, **k: True
    real_popen = subprocess.Popen
    G.subprocess.Popen = lambda cmd, **kw: real_popen(
        [sys.executable, "-c", "import time;print('line 1');print('line 2');time.sleep(.1)"],
        **kw)
    app.start()
    for _ in range(60):
        root.update()
        time.sleep(0.05)
        if app.proc is None:
            break
    log = app.log.get("1.0", "end")
    assert "line 1" in log and "exit code 0" in log, log[-400:]
    assert str(app.run_btn["state"]) == "normal"
    print("subprocess streaming + button state: ok")

    root.destroy()
