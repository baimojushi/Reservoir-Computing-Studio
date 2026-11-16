# RC Studio V0.6 — engineering notes

## Source baseline and work completed

Starting source: user-uploaded `Reservoir-Computing-Studio-main.zip` (V0.1), not a missing V0.2 update. Added offline research project `ebrush_motion_v1`, separate default TCP port 8766, local handoff manifest verifier, six-input ESN, project-aware storage/UI/CLI/HTTP/MCP read/write endpoints, Qt Material-3-inspired light/dark themes, CSV-column bug fix, safe Windows launcher, tests and documentation.

## Non-negotiable ownership boundary

- E-Brush Studio: exclusive capture device serial port, camera, calibration, raw logs, UI, E-Brush service on 8765.
- RC Studio: reads offline copies of files and manifests; parses training episodes and sensor timestamps; owns ESN models, SQLite experimental runs, score reproducibility and 8766 service.
- No embedded Agent; MCP default read-only and write tools require `--allow-write`; HTTP read/write bearer capabilities remain separate.
- No raw logs or source code are fetched from or POSTed to E-Brush; E-Brush device API is never called by RC Studio.

## EBrush data-science choices

- Source recording identity = SHA-256 of entire `.brlog` or CSV; fragments from one recording share identity.
- Train/validation/test groups are disjoint recordings, requiring 3 or more groups. 1 file yields diagnostic episodes but **not** 3 independent captures.
- CRC, COBS, packet length, IMU validity, stale bits, frame counters and MCU timestamps are validated before building episodes.
- 5000-frame maximum per retained segment, 60,000 retained frames per original recording, 256 MiB per file and ZIP limits bound laptop costs. Quality counters still traverse entire accepted source.
- Input channels: gyro 3 axes + accel 3 axes, physical units. Output: future angular-speed magnitude, 10 steps nominally 50 ms at 200 Hz. Not learned Cartesian tip motion.
- Normalization stats fitted on training recording groups only; hidden state resets per segment; washout excludes segment startup; readout uses ridge regression.
- Synthetic and real sources are never mixed in one run. Test vs persistence baseline is always saved; `synthetic_included` is explicit.

## Compatibility and validation gaps

- Existing generic ESN SQL layout, API, trained models preserved. New project uses only derived `.npz` session cache and JSON metadata, not E-Brush workspace.
- GUI desktop render and third-party MCP protocol tests remain pending in actual Windows 11/PySide6 environment. Unit and service testing does not imply these tests passed.
- LICENCE document in uploaded source is GPLv3, despite old README declaring MIT; documentation corrected without removing license.
- Future candidate work: per-episode plots that show boundaries, cross-recording holdout diagnostics, model comparison, hyperparameter search, model input schema and external inference API, Windows packaged executable.

## V0.6 Intuitive Reservoir Laboratory

- Replaced the main echo-layered view with `rcstudio/intuitive_view.py`: one reservoir body, stable node identities, one shared timeline, and five observation lenses. The previous `echo_view.py` implementation remains in the source tree for historical reference but is no longer the primary GUI tab.
- `rcstudio/intuitive_lab.py` is Qt-free. It builds a deterministic, topology-independent node layout and an isolated 64-node ESN sandbox using the same leaky-tanh state update as the training engine. It compares (a) two initial states under the same input and (b) a baseline input against one extra input pulse.
- The sandbox supports sine, pulse, step, chirp and hand-drawn inputs, plus spectral radius, leak-rate and input-scaling controls. Sandbox parameters never mutate a saved experiment or the production training configuration.
- Visual semantics are invariant across lenses: node fill = actual `x_i(t)`; edge sign/width = fixed `W_rec`; moving particles = current recurrent drive; perturbation halo = `Δx`; readout overlay = `w_i x_i(t)`. The readout overlay does not recolor nodes.
- `core.reservoir_structure` now includes output/input normalization statistics, forecast horizon, washout and explicit strongest-edge selection metadata. Generic one-channel readout decomposition can reconstruct the direct-input term; EBrush six-channel direct-input contribution remains explicitly unavailable from target-only trace data.
- The GUI labels Washout as warm-up, labels the prediction target as `y(t+h)`, and reports edge count / total / selection rule. The sidebar topology sketch now depicts input -> one reservoir -> readout rather than stacked pseudo-layers.
- Learning copy tests no longer enforce a global ban on contrastive words. The help content directly clarifies that time positions show the same reservoir at different moments, not serial network layers. Course users can choose “先探索下一步”; the step remains marked incomplete.
- Verification: `python -m pytest -q tests learning_tests` -> 119 passed, 1 skipped, 6 subtests passed. The skipped module is the optional Qt acceptance module. PySide6 is not installed in this environment, so native rendering, pointer interaction, Windows DPI and playback frame rate remain pending target-machine acceptance.

## V0.7 lab redesign

- Adversarial review of V0.6 is in `docs/V07_DESIGN_REVIEW_CN.md`; design and validation of this release in `docs/V07_LAB_CN.md`.
- The intuitive lab is now `rcstudio/lab_web/lab.html`, a self-contained page (no network, no build step) with five guided chapters and a read-only "my run" view. `rcstudio/lab_web_view.py` hosts it in `QWebEngineView`; `rcstudio/lab_web/__init__.py` holds the Qt-free host helpers. `python -m rcstudio lab` opens the same file in a browser.
- Reason for the move: the V0.6 Qt canvas had never been executed in the build environment, and two defects shipped because of it (sandbox sliders never connected; `QPointF` not imported in `gui.py`). The page can be rendered and walked through in headless Chromium (`scripts/check_lab_page.py`).
- The page's numeric engine repeats the leaky-tanh update in JavaScript. `tests/test_lab_web.py` checks it against `intuitive_lab.run_states` (needs Node.js, skipped otherwise) and checks the embedded pools against `scripts/build_lab_networks.py`.
- Teaching pools differ from the training engine in three declared ways: 48 units, zero bias, raw inputs in [-1, 1] without normalisation.
- `intuitive_lab.relaxed_positions` replaces dart-throwing with a relaxed, still topology-independent layout (minimum spacing about 2.4x larger).
- `gui.py`: `QPointF` import; "认识 ESN" / "研究工作台" shell modes stored in QSettings (`shell_mode`, default `learn`); only a run picked in the history table may take over the lab (`show_run(run_id, focus=picked)`).
- `intuitive_view.py` is the fallback when QtWebEngine is missing: sliders wired with a 60 ms coalescing timer and no time reset, one ring scale for the whole perturbation response, packets frozen while paused, stronger wiring in the topology lens, zoom bounded, quiet run loading.
- Help catalog: `rc.echo`, `rc.echo.playback`, `rc.echo.layers` rewritten for the web lab; `ui.mode` added; both generated help documents updated in step. Courses are unchanged and still do not use the lab; `LabWebPanel.chapter_completed` is available for that.
- Verification: 135 passed, 1 skipped with a local stand-in runner (pytest is not installed in the build environment; the V0.6 baseline gives 119 passed, 1 skipped under the same runner). Qt glue was exercised with stub modules only. QtWebEngine embedding, DPI and layout on Windows remain pending.

## V0.7.1 fidelity fixes and merge onto main

- Driven by the adversarial re-review of V0.7; item-by-item record in `docs/V071_RELEASE_CN.md`.
- Real-run inputs: the page keeps `uModel = (u - input_mean) / input_std` (what the engine feeds the pool) apart from the drawn value, which is `uModel` times one positive factor. V0.7.0 drew and drove with a min/max rescale of the window, which flipped the sign on 69 of 1200 frames of a Mackey-Glass run. Multi-input traces carry no model input, so no input lines are drawn and the direct readout term is inferred as "prediction minus the rest".
- Echo verdict: `core.classify_echo` on a 2000-step impulse response (`ECHO_VERDICT_STEPS`), returned as `structure["echo"]` with status `decayed | window_too_short | no_decay_seen | silent`. `structure["impulse"]` (96 frames) is unchanged for the native panel. The page mirrors the classifier in JavaScript for the sandbox and for payloads from older services.
- Readout: `readoutParts` in the page's engine block yields signed node sums, the direct term and the bias per step; the pool shows them as two stacked columns that end at the prediction. Recurrent weights reach the page in model units; no per-run rescale.
- `prepareRun` moved into the DOM-free engine block so `tests/test_fidelity_v071.py` can run it under Node.js against payloads produced by the training engine.
- Courses: `rc_lab` (first in `courses.json`) uses predicate gates `rc.lab.<chapter>` answered by `integration.lab_observation` from `MainWindow.lab_progress`, which is filled by `LabWebPanel.chapter_completed(chapter, detail)`. The reveal hook switches shell mode per step target.
- `scripts/sync_help_docs.py` regenerates each catalog entry's section in the two help documents and has a `--check` mode used by the tests.
- Merge base: uploaded `Reservoir-Computing-Studio-main` at 0.4.0. No file of main is removed; 31 are byte-identical. Main's own tests pass on the new code except the two copy-style tests whose banned-word rule V0.6 replaced on purpose.
- Parent project `E-Brush-as-soft-actuator` references this repository as the `rcstudio` submodule and only probes `GET /v1/health`; its `probe_rcstudio` verifies the 0.7.1 service. `tests/test_fidelity_v071.py` pins the two fields it checks.
- Verification: 148 passed, 1 skipped (local stand-in runner). Qt glue exercised with stubs only.

## Earlier echo-layered implementation (historical)

`rcstudio/echo_view.py` and `rcstudio/reservoir_scene.py` retain the previous layered visualization implementation for reference and compatibility testing. It is no longer exposed as the primary V0.6 user experience.
