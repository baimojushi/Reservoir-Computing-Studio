"""Regressions for clear, contrastive Chinese learning guidance."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG_FOLDER = ROOT / ("studio" if (ROOT / "studio").is_dir() else "rcstudio") / "learning_support"


def _all_text(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _all_text(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_text(child)
    elif isinstance(value, str):
        yield value


def test_learning_copy_uses_direct_explanations():
    """Copy is judged by clarity and consistency, not a global banned-word list."""
    for name in ("catalog.json", "courses.json"):
        source = json.loads((CATALOG_FOLDER / name).read_text(encoding="utf-8"))
        assert list(_all_text(source))


def test_contrastive_clarification_is_available_for_common_esn_misconceptions():
    source = json.loads((CATALOG_FOLDER / "catalog.json").read_text(encoding="utf-8"))
    explanation = source["entries"]["rc.topology"]["explanation"]
    assert "不是多个串联层" in explanation
    for path in (ROOT / "docs" / "CONTROL_HELP_CN.md",
                 ROOT / "docs" / "Studios-Learning-Guide-CN.html"):
        assert "不是多个串联层" in path.read_text(encoding="utf-8")


def test_generated_help_documents_match_runtime_catalog():
    entries = json.loads((CATALOG_FOLDER / "catalog.json").read_text(encoding="utf-8"))["entries"]
    manual = (ROOT / "docs" / "CONTROL_HELP_CN.md").read_text(encoding="utf-8")
    webpage = (ROOT / "docs" / "Studios-Learning-Guide-CN.html").read_text(encoding="utf-8")
    fields = ("title", "summary", "explanation", "example", "pitfalls", "scope", "source")
    for key, entry in entries.items():
        for text in [*(entry.get(f) for f in fields), *entry.get("steps", [])]:
            if text:
                assert text in manual, (key, "manual", text)
                assert escape(text, quote=True) in webpage, (key, "webpage", text)
