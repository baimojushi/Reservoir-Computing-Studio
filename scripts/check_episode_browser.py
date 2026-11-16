"""Run the exact shipped HTML in Chromium against a genuinely trained EBrush model.

Optional developer tool: `pip install playwright`, Chromium executable required.
The test does not make or request any network connections.

Usage: `python scripts/check_episode_browser.py /path/to/evidence`
"""
from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcstudio.core import reservoir_frames, reservoir_structure
from rcstudio.ebrush import make_demo, normalize_ebrush_config, train_motion
from rcstudio.lab_web import page_path, run_payload
from rcstudio.storage import Workspace


def main(target: Path) -> None:
    from playwright.sync_api import sync_playwright

    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        workspace = Workspace(root / "workspace")
        demo = make_demo(workspace, count=5, length=260)
        cfg = normalize_ebrush_config({
            "dataset": {"session_ids": demo["session_ids"]},
            "model": {"units": 18, "washout": 16, "horizon": 10, "seed": 9},
        })
        run_dir = root / "run"
        train_motion(cfg, workspace, run_dir)
        # Window deliberately spans two independent recording boundaries at 250 and 500.
        frames = reservoir_frames(run_dir, start=170, count=480)
        payload = run_payload(reservoir_structure(run_dir, max_edges=160), frames, "actual-ebrush-5-recordings")
        html = page_path().read_text(encoding="utf-8")
        errors = []
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=shutil.which("chromium") or shutil.which("chromium-browser") or None,
                headless=True, args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1400, "height": 860}, device_scale_factor=1)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda e: errors.append(e.text) if e.type == "error" else None)
            page.set_content(html, wait_until="load")
            page.evaluate("p => RCLab.loadRun(p, {focus:true})", payload)
            page.wait_for_timeout(250)
            # Window index 80 is new recording 1, absolute index 250.
            local = 80
            page.evaluate("n => RCLab.seek(n)", local)
            page.wait_for_timeout(300)
            state = page.evaluate("RCLab.state()")
            first = page.evaluate("n => RCLab.probe(n)", local)
            prev = page.evaluate("n => RCLab.probe(n)", local - 1)
            around = [page.evaluate("n => RCLab.probe(n)", i) for i in (69, 70, 79, 80, 81, 95, 96, 100)]
            page.screenshot(path=str(target / "RC_Studio_V072_Real_EBrush_Boundary.png"), full_page=True)
            # Visit the following recording boundary (absolute index 500, local index 330)
            page.evaluate("n => RCLab.seek(n)", 330)
            page.wait_for_timeout(280)
            second = page.evaluate("RCLab.state()")
            page.screenshot(path=str(target / "RC_Studio_V072_Real_EBrush_Second_Boundary.png"), full_page=True)
            # Old data without episode tags must not show a fictitious continuous forecast.
            old = dict(frames)
            for key in ("episode_id", "episode_start", "local_step", "target_local_step", "recording_id", "session_id"):
                old.pop(key, None)
            old["has_episode_metadata"] = False
            page.evaluate("p => RCLab.loadRun(p, {focus:true})", run_payload(payload["structure"], old, "legacy-no-meta"))
            oldstate = page.evaluate("RCLab.state()")
            oldprobe = page.evaluate("RCLab.probe(80)")
            page.screenshot(path=str(target / "RC_Studio_V072_Legacy_Safe_Fallback.png"), full_page=True)
            browser.close()

        report = {
            "project": "RC Studio v0.7.2",
            "verification": "Native HTML (unchanged) in real headless Chromium with actual NumPy/SciPy EBrush train_motion() output",
            "window_start": frames["start"], "window_frames": frames["count"],
            "episode_boundary_indices": [i for i in range(1, frames["count"]) if frames["episode_id"][i] != frames["episode_id"][i-1]],
            "first_boundary": {"probe": first, "previous": prev,
                               "state": {k: state[k] for k in ("episode", "localStep", "episodeBoundary", "clock", "segment")}},
            "second_boundary_state": {k: second[k] for k in ("episode", "localStep", "episodeBoundary", "clock", "segment")},
            "around_first": around,
            "legacy": {"state": {k: oldstate[k] for k in ("legacyMissingBoundaries", "clock", "kind")}, "probe": oldprobe},
            "javascript_errors": errors,
            "checks": {
                "real_boundaries": [i for i in range(1, frames["count"]) if frames["episode_id"][i] != frames["episode_id"][i-1]] == [80, 330],
                "first_boundary_flag": first["episodeBoundary"] is True,
                "new_episode_has_local_step_zero": first["localStep"] == 0,
                "prediction_absent_in_new_washout": first["alignedPrediction"] is None,
                "source_target_never_crosses": prev["targetIndex"] == -1,
                "page_at_boundary": state["episodeBoundary"] is True,
                "second_boundary_highlight": second["episodeBoundary"] is True,
                "old_data_warns_and_hides_forecast": oldstate["legacyMissingBoundaries"] is True and oldprobe["alignedPrediction"] is None,
                "no_javascript_errors": not errors,
            },
        }
        import math
        def clean(obj):
            if isinstance(obj, float) and not math.isfinite(obj):
                return None
            if isinstance(obj, dict):
                return {key: clean(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [clean(value) for value in obj]
            return obj
        (target / "RC_Studio_V072_Real_Chromium_Results.json").write_text(
            json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        for name, good in report["checks"].items():
            print(("PASS" if good else "FAIL") + "  " + name)
        if errors:
            print("Browser errors:", errors)
        assert all(report["checks"].values()), "Chromium acceptance checks failed"


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "v072_browser_evidence")
