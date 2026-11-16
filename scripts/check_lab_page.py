"""Walk the V0.7 lab page in headless Chromium and save screenshots.

This is the visual check the Qt canvas never had: it loads the same file the
desktop app embeds, performs the five chapters the way a newcomer would, asserts
what the page reports, and writes PNGs for a human to look at.

    pip install playwright && playwright install chromium
    python scripts/check_lab_page.py [output_folder]

Optional: pass real-run payloads through RCLab.loadRun to check "我的实验".
"""
from __future__ import annotations

import json
import sys
import tempfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcstudio import lab_web  # noqa: E402


def real_run_payloads():
    """Train two small ESNs with the project's engine and read them back.

    Returns (run, tail_frames, slow, slow_as_v070): a 96-unit Mackey-Glass run, a frame
    window of it that spans validation and test, a slow-but-fading 48-unit run, and that
    same run as a V0.7.0 service would have sent it (no long-window echo verdict).
    """
    from rcstudio.core import reservoir_frames, reservoir_structure, train_esn
    from rcstudio.storage import Workspace
    with tempfile.TemporaryDirectory() as tmp:
        ws, run, slow = Workspace(Path(tmp)), Path(tmp) / "run", Path(tmp) / "slow"
        train_esn({"dataset": {"kind": "mackey_glass", "length": 2000, "noise": 0.0}, "model": {"units": 96}}, ws, run)
        train_esn({"dataset": {"length": 900, "noise": 0.0},
                   "model": {"units": 48, "spectral_radius": 0.90, "leak_rate": 0.10}}, ws, slow)
        frames = reservoir_frames(run, start=0, count=lab_web.FRAME_COUNT)
        tail = reservoir_frames(run, start=lab_web.segment_start(frames, "validation"), count=lab_web.FRAME_COUNT)
        slow_st, slow_fr = reservoir_structure(slow), reservoir_frames(slow, start=0, count=400)
        old = {k: v for k, v in slow_st.items() if k != "echo"}
        return (lab_web.run_payload(reservoir_structure(run, max_edges=lab_web.EDGE_COUNT), frames, "demo-run-0001"), tail,
                lab_web.run_payload(slow_st, slow_fr, "slow-run-0002"), lab_web.run_payload(old, slow_fr, "slow-run-v070"))


def main(out: Path):
    from playwright.sync_api import sync_playwright
    out.mkdir(parents=True, exist_ok=True)
    url = lab_web.page_path().as_uri()
    problems: list[str] = []
    state = lambda page: page.evaluate("RCLab.state()")

    def expect(condition, text):
        if not condition:
            problems.append(text)

    with sync_playwright() as p:
        # Prefer a system Chromium in offline CI; otherwise Playwright's managed browser.
        system_chromium = shutil.which("chromium") or shutil.which("chromium-browser")
        browser = p.chromium.launch(executable_path=system_chromium or None,
                                    args=["--no-sandbox"])

        def open_page(width, height, query):
            page = browser.new_page(viewport={"width": width, "height": height})
            page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
            page.on("console", lambda m: problems.append(f"console: {m.text}") if m.type == "error" else None)
            if system_chromium:
                # Offline environments can forbid file:// navigation.
                page.set_content(lab_web.page_path().read_text(encoding="utf-8"), wait_until="load")
                if "theme=dark" in query:
                    page.evaluate("RCLab.setTheme('dark')")
                if "embedded=1" in query:
                    page.evaluate("RCLab.setEmbedded(true)")
            else:
                page.goto(url + query)
            page.wait_for_timeout(300)
            return page

        page = open_page(1280, 760, "?theme=light")
        expect(state(page)["echo"] is None, "pool should be silent before the first poke")
        page.click("[data-act=poke]"); page.wait_for_timeout(900)
        s = state(page)
        expect(s["echo"] and s["echo"]["settled"] and 15 <= s["echo"]["len"] <= 80, f"default echo out of range: {s['echo']}")
        expect(s["done"][0], "chapter 1 should be marked done after a poke")
        page.screenshot(path=str(out / "1-poke.png"))

        page.click("#nextBtn"); page.wait_for_timeout(300)
        geo, box = state(page)["geo"], page.locator("#pond").bounding_box()
        for node in (7, 20):
            page.mouse.click(box["x"] + geo["sx"][node], box["y"] + geo["sy"][node]); page.wait_for_timeout(150)
        page.mouse.move(4, 4); page.wait_for_timeout(300)
        expect(state(page)["sel"] == 20 and state(page)["done"][1], "node selection did not register")
        page.screenshot(path=str(out / "2-nodes.png"))
        page.focus("#pond"); page.keyboard.press("Escape"); page.keyboard.press("ArrowRight"); page.keyboard.press("ArrowDown")
        expect(state(page)["sel"] >= 0, "nodes cannot be selected from the keyboard")

        page.click("#nextBtn"); page.wait_for_timeout(300)
        expect(state(page)["fit"] is None, "readout must not exist before training")
        page.click("[data-act=train]"); page.wait_for_timeout(3500)
        fit = state(page)["fit"]
        expect(fit and fit["rmse"] * 5 < fit["baseline"], f"readout should beat persistence 5x: {fit}")
        for t in range(30, 230, 20):                       # the columns and the numbers add up to the prediction
            q = page.evaluate("t => RCLab.probe(t)", t)
            pieces = q["parts"]
            expect(abs(pieces["pos"] + pieces["neg"] + pieces["direct"] + pieces["bias"] - q["predicted"]) < 1e-9,
                   f"readout pieces do not add up to the prediction at t={t}")
        expect("合计＝预测" in page.inner_text("#sum"), "the signed-sum row is missing")
        page.screenshot(path=str(out / "3-readout.png"))

        page.click("#nextBtn"); page.wait_for_timeout(300)
        before = state(page)["pos"]
        page.click("[data-preset=forget]"); page.wait_for_timeout(200)
        short = state(page)
        expect(short["echo"]["len"] <= 10, f"'forgetful' preset should shorten the echo: {short['echo']}")
        expect(abs(short["pos"] - before) < 12, "changing a knob must not rewind time")
        page.click("[data-preset=wild]"); page.wait_for_timeout(1800)
        wild = state(page)["echo"]
        expect(wild["status"] == "no_decay_seen", f"'can't stop' preset should show no decay in the window: {wild}")
        expect("未见衰减" in page.inner_text("#head") and "这段窗口内" in page.inner_text("#obs"),
               "a non-decaying echo must be worded as a statement about the window")
        page.screenshot(path=str(out / "4-knobs.png"))
        page.click("[data-preset=default]")

        page.click("#nextBtn"); page.click("[data-input=chirp]"); page.click("[data-show=edges]"); page.wait_for_timeout(2500)
        expect(state(page)["fit"] is not None, "free play should retrain the readout automatically")
        page.screenshot(path=str(out / "5-free.png"))
        page.close()

        page = open_page(1280, 760, "?theme=dark&embedded=1")
        payload, tail, slow, slow_old = real_run_payloads()
        page.evaluate(lab_web.js_call("loadRun", payload, {"focus": False}))
        expect(state(page)["chapter"] == 0, "a background run must not take over the lab")
        page.evaluate(lab_web.js_call("loadRun", payload, {"focus": True})); page.wait_for_timeout(2500)
        expect(state(page)["kind"] == "run", "a picked run should open '我的实验'")
        # fidelity: drawn input and model input agree in sign everywhere; pieces add up
        ro = payload["structure"]["readout"]
        for t in range(0, payload["frames"]["count"], 23):
            q = page.evaluate("t => RCLab.probe(t)", t)
            want = (payload["frames"]["inputs"][t] - ro["input_mean"][0]) / ro["input_std"][0]
            expect(abs(q["model"] - want) < 1e-9, f"input drive is not the standardised input at t={t}")
            expect(q["shown"] * q["model"] >= 0, f"drawn input has the wrong sign at t={t}")
            pieces = q["parts"]
            expect(abs(pieces["pos"] + pieces["neg"] + pieces["direct"] + pieces["bias"] - pieces["net"]) < 1e-9, "pieces != net")
        expect(state(page)["parts"]["residual"] < 0.02, "readout pieces do not rebuild the saved prediction")
        page.screenshot(path=str(out / "6-my-run-dark.png"))
        page.click("[data-segment=validation]")
        message = lab_web.parse_host_message(page.title())
        expect(message and message["type"] == "segment" and message["key"] == "validation", f"segment request missing: {page.title()}")
        # segment button, clock label and playhead always agree
        page.evaluate(lab_web.js_call("loadFrames", tail))
        for t in (0, 60, tail["count"] - 1):
            page.evaluate("t => RCLab.seek(t)", t); page.wait_for_timeout(120)
            s = state(page)
            expect(s["segment"] == s["segmentPressed"] and {"train": "训练段", "validation": "验证段", "test": "测试段"}[s["segment"]] in s["clock"],
                   f"segment out of sync at t={t}: {s['segment']} / {s['segmentPressed']} / {s['clock']}")
        # echo window: long-window verdict when available, an honest 'cannot tell' otherwise
        page.evaluate(lab_web.js_call("loadRun", slow, {"focus": True})); page.wait_for_timeout(200)
        expect(state(page)["echo"]["status"] == "decayed", f"slow but fading run misjudged: {state(page)['echo']}")
        page.evaluate(lab_web.js_call("loadRun", slow_old, {"focus": True})); page.wait_for_timeout(200)
        expect(state(page)["echo"]["status"] == "window_too_short" and "不能判定" in page.inner_text("#obs"),
               f"a 96-step window must read as 'cannot tell': {state(page)['echo']}")
        page.screenshot(path=str(out / "8-window-too-short.png"))
        page.close()

        page = open_page(360, 640, "?theme=light")
        box = page.locator("#stageAct").bounding_box()
        expect(box and box["y"] + box["height"] <= 640, f"first action is below the fold on a small phone: {box}")
        page.click("#stageAct"); page.wait_for_timeout(900)
        expect(state(page)["echo"] is not None, "the on-stage action button did not poke")
        width = page.evaluate("document.documentElement.scrollWidth")
        expect(width <= 360, f"page scrolls sideways on a phone: {width}px")
        page.screenshot(path=str(out / "7-phone.png"), full_page=True)
        browser.close()

    print(json.dumps({"page": url, "screenshots": sorted(f.name for f in out.glob("*.png")), "problems": problems},
                     ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "validation" / "v07-screens"))
