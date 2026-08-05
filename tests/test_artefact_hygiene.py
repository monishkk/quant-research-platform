"""Guards on the artefacts the repository actually publishes.

Committed notebooks and reports are read by people who will never run the code,
so a stale or alarming output is a defect in the deliverable even when the code
is correct. These caught a real one: a checksum bug emitted "the file has
changed since it was written" into all three notebooks, which reads as data
corruption to anyone scrolling past.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))

# Substrings that should never appear in a published notebook's output.
FORBIDDEN = (
    "does not match the recorded",   # checksum false alarm
    "NOT REAL MARKET DATA",          # synthetic data in a committed artefact
    "Traceback (most recent call last)",
)


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks in this checkout")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_error_output(path: Path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    errors = [
        o for cell in nb.get("cells", [])
        for o in cell.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"{path.name} contains {len(errors)} error output(s)"


@pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks in this checkout")
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_output_carries_no_alarming_text(path: Path):
    text = path.read_text(encoding="utf-8")
    hits = [needle for needle in FORBIDDEN if needle in text]
    assert not hits, (
        f"{path.name} publishes text that will alarm a reader: {hits}. "
        "Re-run the notebook once the underlying cause is fixed."
    )


def test_every_source_module_is_listed_in_the_readme():
    """A module the README does not mention is invisible to anyone reading it."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    modules = {p.name for p in (ROOT / "src" / "quant_platform").glob("*.py")
               if p.name != "__init__.py"}
    missing = sorted(m for m in modules if m not in readme)
    assert not missing, f"README's layout section omits: {missing}"
