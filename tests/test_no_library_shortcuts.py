"""Guard rail: the hand-written layers must not call library linear algebra."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = re.compile(r"\blinalg\b|\bscipy\b|\bsklearn\b|\bstatsmodels\b")
FILES = sorted(p for pkg in ("nla", "finance") for p in (ROOT / pkg).glob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_library_linear_algebra(path):
    hits = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if FORBIDDEN.search(line)
    ]
    assert not hits, f"library linear algebra used in {path.name}:\n" + "\n".join(hits)
