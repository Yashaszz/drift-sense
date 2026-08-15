"""The README's results block must match the tracked evidence.

Two people rewrote that block by hand from separate harness runs and reported
latencies 1.7x apart, because one measured on a Mac and the other on Windows and
neither said which. The figures were both honest and the merge would have picked
one by accident.

So the block is generated from ``results/*.csv`` and this test is the thing that
stops it drifting back into prose. It reads only tracked CSVs, never the
dataset, so it runs on a clean checkout.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CSV = REPO_ROOT / "results" / "full_324.csv"


def test_readme_results_block_matches_the_csvs():
    """A hand-edited number, or a stale one after a re-run, fails here."""
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.render_results", "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{completed.stdout}\n{completed.stderr}\n"
        "Regenerate with: uv run python -m scripts.render_results --write"
    )


def test_results_csv_records_the_machine_it_was_measured_on():
    """Latency without a machine is not a defensible number.

    The sidecar is written by ``src.evaluate.write_run_metadata`` beside every
    results CSV. Absence means someone produced evidence without provenance,
    which is exactly the state that let 206 ms and 356 ms both look correct.
    """
    sidecar = FULL_CSV.with_suffix(".meta.json")
    if not sidecar.is_file():
        pytest.fail(
            f"{sidecar.name} is missing — re-run `python -m src.evaluate "
            "--data dataset --out results/full_324.csv` to stamp it"
        )

    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    for field in ("platform", "python", "commit", "cases"):
        assert meta.get(field), f"sidecar is missing {field!r}"


def test_tolerances_match_the_problem_statement():
    """The spec asks for 5, 4, 2 and 1 px. Nothing else, and nothing missing.

    An earlier revision had ``(0.5, 1.0, 2.0)`` — two of the four required
    tolerances absent, one asked for that nobody wants — and the constant was
    referenced nowhere, so no pass rate was reported at any tolerance. Pinning
    the values is cheap; the reporting is covered by the test below.
    """
    from src import config

    assert set(config.TOLERANCES_PX) == {1.0, 2.0, 4.0, 5.0}
    assert config.HEADLINE_TOLERANCE_PX in config.TOLERANCES_PX


def test_readme_reports_every_required_pass_rate():
    """Each required tolerance must appear in the generated block."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("RESULTS:BEGIN")[1].split("RESULTS:END")[0]
    for tol in (1, 2, 4, 5):
        assert f"≤ {tol} px" in block, f"pass rate at {tol} px is not reported"
