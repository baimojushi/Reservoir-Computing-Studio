"""Windows PowerShell and cross-platform CLI for RCStudio."""
import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from . import __version__
from .client import Client, ServiceError, current_port, ensure_server
from .config import DEFAULT, normalize_config
from .core import predict_with_model, train_esn
from .storage import Workspace

def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))

def parser():
    p = argparse.ArgumentParser(
        prog="rcstudio",
        description="Reservoir Computing Studio: GUI, CLI, local service and MCP.",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="Run a loopback HTTP service in foreground")
    serve.add_argument("--port", type=int, default=current_port())
    sub.add_parser("gui", help="Open PySide6 desktop workbench (auto-start service)")
    sub.add_parser("status", help="Read workspace state as JSON")
    sub.add_parser("datasets", help="List imported CSV datasets")
    import_cmd = sub.add_parser("import-csv", help="Import a local CSV into managed storage")
    import_cmd.add_argument("file")
    runs = sub.add_parser("runs", help="List recent experiments")
    runs.add_argument("--limit", type=int, default=30)
    run = sub.add_parser("run", help="Queue an experiment; optionally wait for result")
    run.add_argument("--config", type=Path)
    run.add_argument("--wait", action="store_true")
    run.add_argument("--timeout", type=int, default=300)
    show = sub.add_parser("show", help="Get a run by ID")
    show.add_argument("run_id")
    traces = sub.add_parser("traces", help="Get sampled neuron activity as JSON")
    traces.add_argument("run_id")
    traces.add_argument("--limit", type=int, default=150)
    traces.add_argument("--units", type=int, default=12)
    stop = sub.add_parser("stop", help="Request cancellation of a run")
    stop.add_argument("run_id")
    sub.add_parser("diagnostics", help="Show recent job errors")
    config = sub.add_parser("sample-config", help="Print a default JSON configuration")
    config.add_argument("--output", type=Path)
    mcp = sub.add_parser("mcp", help="Launch an MCP v2 stdio server for external agents")
    mcp.add_argument("--allow-write", action="store_true",
                     help="Explicitly allow MCP clients to queue/stop experiments")
    predict = sub.add_parser("predict", help="Use a saved model on imported CSV")
    predict.add_argument("run_id", help="Completed training run")
    predict.add_argument("--dataset-id", required=True)
    predict.add_argument("--column")
    predict.add_argument("--output", type=Path)
    sub.add_parser("check", help="Run a standalone built-in ESN self-test")
    return p

def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.cmd == "serve":
            from .service import run_server
            run_server(port=args.port)
            return 0
        if args.cmd == "sample-config":
            cfg = json.dumps(DEFAULT, ensure_ascii=False, indent=2)
            if args.output:
                args.output.write_text(cfg, encoding="utf-8")
                output({"saved_to": str(args.output.resolve())})
            else:
                print(cfg)
            return 0
        if args.cmd == "check":
            with tempfile.TemporaryDirectory(prefix="rcstudio-check-") as tmp:
                ws = Workspace(Path(tmp) / "workspace")
                cfg = normalize_config({
                    "dataset": {"length": 380, "noise": 0.0},
                    "model": {"units": 35, "washout": 25},
                })
                results = train_esn(cfg, ws, Path(tmp) / "artifacts")
                output({"ok": True, "test_metrics": results["scores"]["test"]})
            return 0
        if args.cmd == "import-csv":
            output(Workspace().import_csv(args.file))
            return 0
        if args.cmd == "mcp":
            from .mcp_adapter import start_mcp
            start_mcp(allow_write=args.allow_write)
            return 0
        if args.cmd == "gui":
            client = ensure_server()
            from .gui import start_gui
            return start_gui(client)
        if args.cmd == "predict":
            ws = Workspace()
            row = ws.get_run(args.run_id)
            if not row or row["status"] != "finished":
                raise ValueError("Run must exist and be finished")
            data, col = ws.read_csv_series(args.dataset_id, args.column)
            preds = predict_with_model(
                ws.artifact_dir(args.run_id) / "model.npz", data
            )
            target = args.output or (
                ws.artifact_dir(args.run_id) / f"inference_{int(time.time())}.csv"
            )
            target = target.expanduser().resolve()
            with target.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestep", "input", "prediction", "truth"])
                horizon = row["config"]["model"]["horizon"]
                for t, pred in enumerate(preds):
                    writer.writerow([t + horizon, data[t], pred, data[t + horizon]])
            output({"file": str(target), "rows": len(preds), "column": col})
            return 0

        client = ensure_server()
        if args.cmd == "status":
            output(client.workspace_state())
        elif args.cmd == "datasets":
            output({"datasets": client.datasets()})
        elif args.cmd == "runs":
            output({"runs": client.runs(args.limit)})
        elif args.cmd == "show":
            output(client.run(args.run_id))
        elif args.cmd == "traces":
            output(client.traces(args.run_id, args.limit, args.units))
        elif args.cmd == "stop":
            output(client.stop(args.run_id))
        elif args.cmd == "diagnostics":
            output(client.diagnostics())
        elif args.cmd == "run":
            raw = json.loads(args.config.read_text(encoding="utf-8")) \
                if args.config else {}
            row = client.submit(normalize_config(raw))
            if not args.wait:
                output(row)
            else:
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    row = client.run(row["id"])
                    if row["status"] not in ("queued", "running"):
                        output(row)
                        return 0 if row["status"] == "finished" else 1
                    time.sleep(0.4)
                output({"id": row["id"], "status": "timeout",
                        "message": "Job continues in background; use stop to cancel"})
                return 2
        else:
            raise RuntimeError("Unknown command")
        return 0
    except (ValueError, OSError, ServiceError, ImportError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
