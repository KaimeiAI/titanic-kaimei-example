"""The committed notebooks are clean: run top to bottom, no errors, no secrets, no local paths."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("[0-9][0-9]-*.ipynb"))
SECRET = re.compile(r"km_(live|test)_[A-Za-z0-9_-]{8,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")
LOCAL_PATH = re.compile(r"[A-Za-z]:\\\\?Users\\\\?|/Users/[a-z]|/home/[a-z]", re.IGNORECASE)


def _text_outputs(notebook):
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if "text" in output:
                yield "".join(output["text"])
            for kind, value in output.get("data", {}).items():
                if kind.startswith("text/") or kind == "application/json":
                    yield value if isinstance(value, str) else "".join(value)
            if output.get("output_type") == "error":
                yield "\n".join(output.get("traceback", []))


@pytest.fixture(params=NOTEBOOKS, ids=lambda path: path.stem)
def notebook(request):
    return json.loads(request.param.read_text(encoding="utf-8"))


def test_there_is_a_notebook_series():
    assert [path.name[:2] for path in NOTEBOOKS] == ["00", "01", "02", "03", "04"]


def test_runs_top_to_bottom_without_errors(notebook):
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code]
    assert all(counts), "every code cell has run (python scripts/run_notebooks.py --all)"
    assert counts == sorted(counts) and len(set(counts)) == len(counts), "cells ran in order, once"
    for cell in code:
        assert not any(output.get("output_type") == "error" for output in cell.get("outputs", []))


def test_outputs_hold_no_secrets_or_local_paths(notebook):
    for text in _text_outputs(notebook):
        assert not SECRET.search(text), "an API key or session token is in an output"
        assert not LOCAL_PATH.search(text), f"a local path is in an output: {LOCAL_PATH.search(text).group(0)}"
    sources = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert not SECRET.search(sources)


def test_starts_with_a_title(notebook):
    first = notebook["cells"][0]
    assert first["cell_type"] == "markdown" and "".join(first["source"]).startswith("# ")
