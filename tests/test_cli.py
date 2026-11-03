"""Run config <-> CLI round trip, directory layout and manifest warnings. No GUI, no OSRM."""
import contextlib
import io
import subprocess
import sys
from pathlib import Path

import scripts.run_pipeline as P
from scripts.runconfig import RunConfig

ROOT = Path(__file__).resolve().parent.parent


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "scripts.run_pipeline", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def test_config_round_trip_and_overrides(tmp_path):
    rc = RunConfig(specs=("super_disc",), years=(2019, 2022, 2025), year_mode="list",
                   bases=("vkfl", "umsatz"), scope="food", levels=("KREIS",), land="09")
    p = rc.save(tmp_path / "narrow.toml")
    assert set(rc.report_bases()) == {"vkfl_food", "umsatz_food"}
    assert set(RunConfig(bases=("pop", "vkfl"), scope="food").report_bases()) == \
        {"pop", "vkfl_food"}, "demand-side bases have no food variant"
    assert any("border" in n for n in rc.validate() if n.startswith("note:"))

    out = _cli("--config", p, "--save-config", tmp_path / "echo.toml")
    assert out.returncode == 0, out.stderr[-800:]
    rc2 = RunConfig.load(tmp_path / "echo.toml")
    for f in ("specs", "years", "bases", "scope", "levels", "land"):
        assert getattr(rc, f) == getattr(rc2, f), f

    out = _cli("--config", p, "--years", "2014:2016", "--specs", "super",
               "--save-config", tmp_path / "over.toml")
    assert out.returncode == 0, out.stderr[-800:]
    rc3 = RunConfig.load(tmp_path / "over.toml")
    assert rc3.years == (2014, 2015, 2016) and rc3.specs == ("super",)


def test_invalid_config_exits_2(tmp_path):
    RunConfig(specs=(), years=()).save(tmp_path / "bad.toml")
    out = _cli("--config", tmp_path / "bad.toml")
    assert out.returncode == 2 and "ERROR" in out.stdout, out.stdout


def test_layout():
    a = RunConfig().pipeline_config()
    b = RunConfig(land="09").pipeline_config()
    # snaps and the grid are network-level: one copy for every run, Land or not
    assert a.snapdir == b.snapdir and a.griddir == b.griddir
    assert a.cachedir == a.snapdir / "cache"
    # matrices are per shop type and per tag; a Land-only solve must not overwrite
    assert a.solvedir("Supermarkt") != b.solvedir("Supermarkt")
    assert a.solvedir("Discount Geschäft").name == "discount_geschaeft"
    assert a.derivdir("super_disc").name == "super_disc"


def test_manifest_warns_on_changed_input(tmp_path):
    mf = tmp_path / "man.json"
    P.manifest(mf, {"network": "germany-latest", "n": 1}, what="the store snap")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        P.manifest(mf, {"network": "germany-latest", "n": 1}, what="the store snap")
    assert buf.getvalue() == "", "an unchanged manifest must be silent"
    with contextlib.redirect_stdout(buf):
        P.manifest(mf, {"network": "germany-2026-05", "n": 2}, what="the store snap")
    assert "network: was 'germany-latest'" in buf.getvalue()
