"""Regenerate the demo reservoirs embedded in rcstudio/lab_web/lab.html.

The browser lab cannot compute eigenvalues cheaply, so the unit-spectral-radius
recurrent matrices are produced here by the project's own sandbox generator and
written between the NETWORKS markers of the page. Run from the repository root:

    python scripts/build_lab_networks.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcstudio.intuitive_lab import _network, relaxed_positions  # noqa: E402

UNITS = 48
DENSITY = 0.10
SEEDS = (2026, 2027, 2031, 2040, 2053, 2068)
BEGIN = "/*NETWORKS-BEGIN*/"
END = "/*NETWORKS-END*/"


def build_networks():
    nets = []
    for seed in SEEDS:
        W, win, _bias, radius = _network(UNITS, DENSITY, seed, 1.0, 1.0)
        dst, src = np.nonzero(W)                     # W[i, j] carries node j -> node i
        nets.append({
            "seed": seed,
            "n": UNITS,
            "radius": round(float(radius), 9),
            "src": src.tolist(),
            "dst": dst.tolist(),
            "w": [round(float(v), 7) for v in W[dst, src]],
            "win": [round(float(v), 6) for v in win],
            # Bias is zeroed for the teaching pools so that "no input" really looks
            # still; the training engine keeps its own non-zero bias.
            "bias": [0.0] * UNITS,
            "pos": [[round(float(x), 4), round(float(y), 4)]
                    for x, y in relaxed_positions(UNITS, seed=seed + 3001)],
        })
    return nets


def main():
    page = ROOT / "rcstudio" / "lab_web" / "lab.html"
    text = page.read_text(encoding="utf-8")
    payload = json.dumps(build_networks(), separators=(",", ":"))
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise SystemExit("NETWORKS markers not found in lab.html")
    page.write_text(pattern.sub(lambda _m: BEGIN + payload + END, text), encoding="utf-8")
    print(f"embedded {len(SEEDS)} networks x {UNITS} units into {page.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
