# RC Studio v0.1 — implementation notes

## Design decisions

- The compute service owns training and SQLite state; GUI and CLI are clients.
- MCP is an optional stdio bridge to the same service, with read-only tools
  unless explicitly launched using `--allow-write`.
- No LLM is embedded in the application.
- NumPy/SciPy ESN is used as a tested baseline; ReservoirPy integration is
  reserved for a replaceable future engine adapter.
- CPU-only training is serialized (`max_workers=1`) to avoid experiment
  resource contention on laptops; jobs are cancelable.
- Local HTTP binds 127.0.0.1 only, validates Host/Origin, and requires
  separate read/write tokens for all non-health endpoints.
- Credentials and raw data stay under the local workspace by default.
- Data preprocessing is chronological, train-only scaling, no random
  train/test shuffle. Test prediction is observed-input forecasting.

## Extension guide

V0.2:
  - Introduce `ModelAdapter` so ReservoirPy and custom ESN can be selected.
  - Add GUI train/validation split controls and richer dataset preview.
  - Add statistics/log/event subscription to avoid polling.
  - Add agent tool resource paging and GUI captures with explicit opt-in.
  - Add model comparison plots and hyperparameter sweeps.
  - Package a signed Windows executable after compatibility testing.

V0.3:
  - Node editor supporting explicit loops and delayed edges.
  - Recursive free-running forecast separate from teacher forcing.
  - Multiple reservoirs, online readout, memory capacity diagnostics.
  - HTTP/MCP streamable transport if needed for separate processes.

## Validation

Run `python -m pytest -q tests`.
For a real CLI end-to-end test, run `python -m rcstudio run --config
examples/sine.json --wait`, then `python -m rcstudio traces <run_id>`.
GUI/MCP require the optional packages and target machine checks.
