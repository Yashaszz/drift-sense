"""Every documented test count must match what pytest actually collects.

This number has been wrong three times. It was 576 in the README and the deck
outline, 586 in the handoff and 578 on the slide -- four places, four numbers,
none of them right. Correcting all four by hand fixed it for about an hour: a PR
merged six tests in between measuring and landing, so the "fix" shipped 588
against a real 596.

Hand-typing a derived number does not survive a day of parallel merges, and this
one is quoted to a judge. So it is pinned here instead. The count comes from
``pytest --collect-only`` rather than from counting ``def test_`` -- the
functions parametrise into far more cases, and the collected figure is the one
the docs mean.

Runs in a subprocess so the answer does not depend on how the current session
was invoked: ``pytest -k something`` still checks the whole suite.
"""

from __future__ import annotations

import re
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Files carrying a prose test count, and how to read their text.
_MARKDOWN_SOURCES = (
    Path("README.md"),
    Path("docs/deck_outline.md"),
    Path("docs/r4_handoff.md"),
)
_DECK = Path("docs/DriftSense_PS02.pptx")

_CLAIM = re.compile(r"\b(\d{3,4}) tests\b")


def _collected_test_count() -> int:
    """Return how many tests pytest collects from ``tests/``."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(f"collection failed:\n{completed.stdout}\n{completed.stderr}")

    total = 0
    for line in completed.stdout.splitlines():
        match = re.fullmatch(r"tests/\S+\.py: (\d+)", line.strip())
        if match:
            total += int(match.group(1))

    assert total, f"could not parse a count from collection output:\n{completed.stdout}"
    return total


def _deck_text(path: Path) -> str:
    """Return the visible text of every slide, without needing python-pptx.

    The deck is not a dependency of this project and a grader's checkout will
    not have one installed, so the slide XML is read directly.
    """
    with zipfile.ZipFile(path) as archive:
        slides = [n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        return " ".join(
            " ".join(re.findall(r"<a:t>(.*?)</a:t>", archive.read(name).decode("utf-8", "ignore")))
            for name in slides
        )


def _claims() -> list[tuple[Path, int]]:
    """Return every ``N tests`` claim in the tracked deliverables."""
    found: list[tuple[Path, int]] = []
    for relative in _MARKDOWN_SOURCES:
        path = REPO_ROOT / relative
        if not path.exists():
            continue
        found += [(relative, int(n)) for n in _CLAIM.findall(path.read_text(encoding="utf-8"))]

    deck = REPO_ROOT / _DECK
    if deck.exists():
        found += [(_DECK, int(n)) for n in _CLAIM.findall(_deck_text(deck))]
    return found


def test_no_document_claims_more_tests_than_exist():
    """Overstating the count fails; understating it only warns.

    An exact match would be stricter and worse: the count lives in four files,
    one of them a binary ``.pptx``, so every PR that adds a test would go red
    until someone hand-edited all four -- and two such PRs would conflict on the
    deck. That is a poor trade for a number nobody reads twice.

    What actually matters is the direction. Claiming *more* tests than exist is
    a false statement to a judge, so it fails. Claiming fewer is merely
    conservative, so it warns and stays green.
    """
    actual = _collected_test_count()
    claims = _claims()

    assert claims, "no documented test count found -- has the README stopped quoting one?"

    overstated = sorted({(str(where), n) for where, n in claims if n > actual})
    assert not overstated, (
        f"pytest collects {actual} tests, but these claim more: "
        + "; ".join(f"{where} says {n}" for where, n in overstated)
        + f". Correct them to {actual}."
    )

    understated = sorted({(str(where), n) for where, n in claims if n < actual})
    if understated:
        warnings.warn(
            f"pytest collects {actual} tests; "
            + "; ".join(f"{where} says {n}" for where, n in understated)
            + f". Conservative, so not a failure -- refresh to {actual} when convenient.",
            stacklevel=2,
        )
