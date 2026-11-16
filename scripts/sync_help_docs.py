"""Keep the two help documents in step with rcstudio/learning_support/catalog.json.

    python scripts/sync_help_docs.py           # rewrite changed or missing sections
    python scripts/sync_help_docs.py --check   # exit 1 if a section is out of date

Each catalog entry owns one section in docs/CONTROL_HELP_CN.md and one <details>
block in docs/Studios-Learning-Guide-CN.html, located by its help ID. Text
outside those sections is left alone.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "rcstudio" / "learning_support" / "catalog.json"
MANUAL = ROOT / "docs" / "CONTROL_HELP_CN.md"
WEBPAGE = ROOT / "docs" / "Studios-Learning-Guide-CN.html"
PARTS = (("操作前请注意", "risk"), ("为什么这样设计", "explanation"), ("一个具体例子", "example"),
         ("常见误区与排查", "pitfalls"), ("职责边界", "scope"), ("对应实现", "source"))


def manual_section(key: str, entry: dict) -> str:
    out = [f"## {entry['title']}", "", f"分类：{entry['category']} · ID：`{key}`", "", entry["summary"], "",
           "### 怎样使用", ""]
    out += [f"{n}. {step}" for n, step in enumerate(entry["steps"], 1)] + [""]
    for heading, field in PARTS:
        if entry.get(field):
            out += [f"### {heading}", "", entry[field], ""]
    return "\n".join(out) + "\n"


def webpage_block(key: str, entry: dict) -> str:
    esc = lambda text: html.escape(text, quote=True)
    block = (f'<details class="entry" data-app="rcstudio" data-kind="help"><summary>{esc(entry["title"])}'
             f'<small>{esc(entry["category"])} · {key}</small></summary><div class="body">'
             f'<p class="lead">{esc(entry["summary"])}</p><h3>怎样使用</h3><ol>')
    block += "".join(f"<li>{esc(step)}</li>" for step in entry["steps"]) + "</ol>"
    for heading, field in PARTS:
        if entry.get(field):
            block += f"<h3>{heading}</h3><p>{esc(entry[field])}</p>"
    return block + "</div></details>"


def sync(write: bool) -> list[str]:
    entries = json.loads(CATALOG.read_text(encoding="utf-8"))["entries"]
    manual, page = MANUAL.read_text(encoding="utf-8"), WEBPAGE.read_text(encoding="utf-8")
    stale: list[str] = []
    previous = None
    for key, entry in entries.items():
        section, block = manual_section(key, entry), webpage_block(key, entry)
        marker = f" · ID：`{key}`\n"
        if section.rstrip("\n") not in manual:
            stale.append(f"manual:{key}")
            if marker in manual:
                at = manual.index(marker)
                start = manual.rfind("\n## ", 0, at) + 1
                end = manual.find("\n## ", at)
                end = len(manual) if end < 0 else end + 1
                manual = manual[:start] + section + "\n" + manual[end:]
            else:                                   # new entry: place it after the previous one
                at = manual.index(f" · ID：`{previous}`\n") if previous else -1
                end = manual.find("\n## ", at) if at >= 0 else -1
                end = len(manual) if end < 0 else end + 1
                manual = manual[:end].rstrip("\n") + "\n\n" + section + "\n" + manual[end:]
        tag = f" · {key}</small>"
        if block not in page:
            stale.append(f"webpage:{key}")
            if tag in page:
                at = page.index(tag)
                start, end = page.rfind("<details", 0, at), page.index("</details>", at) + len("</details>")
                page = page[:start] + block + page[end:]
            else:
                at = page.index(f" · {previous}</small>")
                end = page.index("</details>", at) + len("</details>")
                page = page[:end] + "\n" + block + page[end:]
        previous = key
    if write and stale:
        MANUAL.write_text(manual, encoding="utf-8")
        WEBPAGE.write_text(page, encoding="utf-8")
    return stale


if __name__ == "__main__":
    check = "--check" in sys.argv[1:]
    changed = sync(write=not check)
    print(json.dumps({"mode": "check" if check else "write", "out_of_date": changed}, ensure_ascii=False))
    sys.exit(1 if check and changed else 0)
