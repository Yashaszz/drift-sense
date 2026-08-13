#!/usr/bin/env bash
# Prove the submission works on a machine that has never seen the project.
#
# `git archive` gives exactly what a grader unzips: tracked files only, no
# .venv, no dataset (gitignored), no trained artefacts. Everything that works
# in the development tree because of a file git does not carry fails here, and
# nowhere else. This is the check that catches "works on my machine".
#
# It has already caught one: an ignore rule that excluded the results
# .meta.json sidecars, which would have failed CI on a fresh checkout only.
#
# Usage:
#   bash scripts/clean_room_check.sh [git-ref]     # default: HEAD

set -euo pipefail

REF="${1:-HEAD}"
ROOT="$(git rev-parse --show-toplevel)"
CLEAN="$(mktemp -d)"
trap 'rm -rf "$CLEAN"' EXIT

echo "==> Unpacking $REF into $CLEAN"
git -C "$ROOT" archive --format=tar "$REF" | tar -x -C "$CLEAN"
echo "    $(find "$CLEAN" -type f | wc -l | tr -d ' ') tracked files"

echo "==> Resolving the environment from the lockfile"
(cd "$CLEAN" && uv sync --all-extras >/dev/null)

# Piping these into `tail` would be a trap: tail exits early, the producer takes
# SIGPIPE, and `set -o pipefail` aborts the script at 141 having reported
# success for every stage it did reach. Log to a file and read the tail instead.
echo "==> Test suite"
(cd "$CLEAN" && uv run pytest -q >"$CLEAN/pytest.log" 2>&1)
tail -2 "$CLEAN/pytest.log"

echo "==> Lint and types"
(cd "$CLEAN" && uv run ruff check src tests benchmarks)
(cd "$CLEAN" && uv run mypy >"$CLEAN/mypy.log" 2>&1)
tail -1 "$CLEAN/mypy.log"

# The CLI is the deliverable, so it is exercised on images the clean tree has
# never seen. The dataset is gitignored and absent by design; a grader supplies
# their own pair, and so do we.
# A glob, not `ls | head -1`: head exits after one line, ls takes SIGPIPE, and
# pipefail turns a successful lookup into a 141 that aborts the script.
if [[ -z "${PAIR:-}" ]]; then
  shopt -s nullglob
  candidates=("$ROOT"/dataset/reference/*.png)
  shopt -u nullglob
  PAIR="${candidates[0]:-}"
fi
if [[ -n "$PAIR" && -f "$PAIR" ]]; then
  NAME="$(basename "$PAIR")"
  cp "$PAIR" "$CLEAN/reference.png"
  cp "$ROOT/dataset/search/$NAME" "$CLEAN/search.png"
  echo "==> CLI end to end on $NAME"
  (cd "$CLEAN" && uv run python -m src.localize search.png reference.png --json \
    >"$CLEAN/cli.json" 2>&1)
  grep -E '"(x|y|confidence|low_confidence_flag|mode_used)"' "$CLEAN/cli.json"
else
  echo "==> CLI skipped: no dataset/ locally to borrow a pair from"
fi

echo
echo "Clean-room check passed for $REF"
