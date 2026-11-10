# RC Studio V0.3 — engineering notes

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
