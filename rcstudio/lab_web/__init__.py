"""V0.7 intuitive lab: one self-contained web page plus Qt-free host helpers.

``lab.html`` holds the whole lab (markup, styles, numerics, drawing). It needs
no network and no build step, so it can be opened three ways:

* inside RC Studio, in a ``QWebEngineView`` (see ``rcstudio.lab_web_view``);
* in any browser via ``python -m rcstudio lab``;
* in headless Chromium for automated visual checks.

Nothing in this module imports Qt.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..intuitive_lab import relaxed_positions

HOST_PREFIX = "rclab:"
FRAME_COUNT = 1200
EDGE_COUNT = 4000
CONTEXT_FRAMES = 40          # frames shown before a validation/test segment begins


def page_path() -> Path:
    """Absolute path of the lab page shipped with the package."""
    return Path(__file__).resolve().with_name("lab.html")


def run_positions(units: int) -> list[list[float]]:
    """Layout for a trained run: depends on the node count only, never on weights."""
    units = int(units)
    return relaxed_positions(units, seed=202611 + units * 13).round(4).tolist()


def run_payload(structure: dict, frames: dict, run_id: str) -> dict:
    """Bundle the two read-only endpoint results for ``RCLab.loadRun``."""
    return {
        "structure": structure,
        "frames": frames,
        "positions": run_positions(structure["units"]),
        "runId": str(run_id),
    }


def _finite(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def js_call(function: str, *args: Any) -> str:
    """Serialise a call into the page. Arguments travel as JSON, never as code."""
    try:
        encoded = ", ".join(json.dumps(arg, ensure_ascii=False, allow_nan=False) for arg in args)
    except ValueError:               # NaN/inf are not JSON; the page treats null as "missing"
        encoded = ", ".join(json.dumps(_finite(arg), ensure_ascii=False, allow_nan=False) for arg in args)
    # U+2028/2029 are valid JSON but end a line in JavaScript source.
    encoded = encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return f"window.RCLab && window.RCLab.{function}({encoded});"


def parse_host_message(title: str) -> dict | None:
    """Decode a page-to-host message carried in the document title."""
    if not isinstance(title, str) or not title.startswith(HOST_PREFIX):
        return None
    try:
        message = json.loads(title[len(HOST_PREFIX):])
    except ValueError:
        return None
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        return None
    return message


def segment_start(frames: dict, key: str) -> int | None:
    """First frame to request for the train / validation / test segment."""
    first = (frames or {}).get("split_starts", {}).get(key)
    if first is None:
        return None
    return max(0, int(first) - (0 if key == "train" else CONTEXT_FRAMES))
