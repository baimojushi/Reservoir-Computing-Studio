"""Loopback JSON HTTP server with scoped tokens and serialized training jobs."""
import hmac
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .config import ConfigurationError, normalize_config
from .core import ExperimentCancelled, sample_traces, train_esn
from .storage import Workspace
from .ebrush import PROJECT_ID, normalize_ebrush_config, list_sessions, preflight_training, train_motion

class RCService:
    def __init__(self, workspace=None):
        self.workspace = workspace or Workspace()
        self.workspace.mark_interrupted()
        self.tokens = self.workspace.credentials()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rc-train")
        self.jobs = {}
        self.lock = threading.Lock()

    def workspace_state(self):
        runs = self.workspace.list_runs(30)
        return {
            "app": "RCStudio", "version": __version__,
            "workspace": str(self.workspace.root),
            "runs": runs,
            "datasets": self.workspace.list_datasets(),
            "projects": [{"id": "generic_esn", "name": "Generic ESN"},
                         {"id": PROJECT_ID, "name": "EBrush Motion Predictor V1",
                          "session_count": len(list_sessions(self.workspace))}],
        }

    def submit(self, raw_config):
        if not isinstance(raw_config, dict):
            raise ConfigurationError("config: expected an object")
        is_ebrush = raw_config.get("project") == PROJECT_ID
        cfg = (normalize_ebrush_config(raw_config) if is_ebrush
               else normalize_config(raw_config))
        if is_ebrush:
            preflight_training(cfg, self.workspace)
        if not is_ebrush and cfg["dataset"]["kind"] == "csv":
            # Fail fast on unknown dataset/invalid CSV.
            self.workspace.read_csv_series(
                cfg["dataset"]["dataset_id"], cfg["dataset"]["column"]
            )
        run_id = self.workspace.create_run(cfg)
        stop = threading.Event()
        with self.lock:
            self.jobs[run_id] = stop
        self.pool.submit(self._execute, run_id, cfg, stop)
        return self.workspace.get_run(run_id)

    def _execute(self, run_id, cfg, stop):
        try:
            if stop.is_set():
                raise ExperimentCancelled("Cancelled while queued")
            self.workspace.update_run(run_id, status="running", progress=0.01)
            last_percent = -1
            def report(value):
                nonlocal last_percent
                percent = int(100 * value)
                if percent > last_percent:
                    last_percent = percent
                    self.workspace.update_run(run_id, progress=value)
            trainer = train_motion if cfg.get("project") == PROJECT_ID else train_esn
            summary = trainer(
                cfg, self.workspace, self.workspace.artifact_dir(run_id),
                cancel=stop, progress=report,
            )
            self.workspace.update_run(
                run_id, status="finished", progress=1.0, metrics=summary
            )
        except ExperimentCancelled as exc:
            self.workspace.update_run(
                run_id, status="cancelled", error=str(exc)
            )
        except Exception as exc:
            self.workspace.update_run(
                run_id, status="failed",
                error=f"{type(exc).__name__}: {exc}"
            )
            with (self.workspace.root / "service_errors.log").open(
                "a", encoding="utf-8"
            ) as log:
                log.write(f"{run_id}\n{traceback.format_exc()}\n")
        finally:
            with self.lock:
                self.jobs.pop(run_id, None)

    def stop(self, run_id):
        run = self.workspace.get_run(run_id)
        if not run:
            raise FileNotFoundError("Run not found")
        if run["status"] not in ("queued", "running"):
            return run
        with self.lock:
            event = self.jobs.get(run_id)
        if event is not None:
            event.set()
        return {"id": run_id, "stop_requested": event is not None}

class RCHandler(BaseHTTPRequestHandler):
    server_version = "RCStudio/0.3"
    def log_message(self, fmt, *args):
        # Do not leak bearer tokens or request data in HTTP logs.
        return

    def _json(self, status, payload):
        blob = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(blob)

    def _serve(self):
        try:
            host = self.headers.get("Host", "")
            port = self.server.server_port
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}",
                            "127.0.0.1", "localhost"):
                return self._json(403, {"error": "Host not allowed"})
            if self.headers.get("Origin"):
                return self._json(403, {"error": "Browser Origin disallowed"})
            url = urlsplit(self.path)
            parts = [p for p in url.path.split("/") if p]
            query = parse_qs(url.query)
            svc = self.server.rcservice
            if self.command == "GET" and parts == ["v1", "health"]:
                return self._json(200, {
                    "app": "RCStudio", "version": __version__, "pid": os.getpid(),
                    "port_role": "rc_training_only", "projects": ["generic_esn", PROJECT_ID]
                })
            auth = self.headers.get("Authorization", "")
            provided = auth[7:] if auth.startswith("Bearer ") else ""
            can_write = hmac.compare_digest(provided, svc.tokens["write_token"])
            can_read = can_write or hmac.compare_digest(
                provided, svc.tokens["read_token"]
            )
            if not can_read:
                return self._json(401, {"error": "Bearer token required"})
            if self.command == "POST" and not can_write:
                return self._json(403, {"error": "Write capability required"})

            if self.command == "GET":
                if parts == ["v1", "workspace"]:
                    return self._json(200, svc.workspace_state())
                if parts == ["v1", "datasets"]:
                    return self._json(200, {"datasets": svc.workspace.list_datasets()})
                if parts == ["v1", "projects"]:
                    return self._json(200, {"projects": svc.workspace_state()["projects"]})
                if parts == ["v1", "projects", "ebrush", "sessions"]:
                    return self._json(200, {"sessions": list_sessions(svc.workspace)})
                if parts == ["v1", "runs"]:
                    limit = int(query.get("limit", ["30"])[0])
                    return self._json(200, {"runs": svc.workspace.list_runs(limit)})
                if parts == ["v1", "diagnostics"]:
                    runs = svc.workspace.list_runs(100)
                    return self._json(200, {"failures": [
                        {"id": r["id"], "status": r["status"], "error": r["error"]}
                        for r in runs if r["status"] in ("failed", "interrupted")
                    ][:20]})
                if len(parts) == 3 and parts[:2] == ["v1", "runs"]:
                    run = svc.workspace.get_run(parts[2])
                    return self._json(200, run) if run else self._json(
                        404, {"error": "Run not found"}
                    )
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] \
                        and parts[3] == "traces":
                    run = svc.workspace.get_run(parts[2])
                    if not run:
                        return self._json(404, {"error": "Run not found"})
                    if run["status"] != "finished":
                        return self._json(409, {"error": "Run is not finished"})
                    trace = sample_traces(
                        svc.workspace.artifact_dir(parts[2]),
                        limit=int(query.get("limit", ["600"])[0]),
                        units=int(query.get("units", ["24"])[0]),
                    )
                    return self._json(200, trace)

            if self.command == "POST":
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > 65536:
                    return self._json(413, {"error": "Body must be <= 64 KB"})
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(body, dict):
                    return self._json(400, {"error": "Expected JSON object"})
                if parts == ["v1", "runs"]:
                    return self._json(202, svc.submit(body.get("config", {})))
                if parts == ["v1", "projects", "ebrush", "train"]:
                    raw = body.get("config", {})
                    if not isinstance(raw, dict):
                        raise ConfigurationError("config must be an object")
                    raw = {**raw, "project": PROJECT_ID}
                    return self._json(202, svc.submit(raw))
                if len(parts) == 4 and parts[:2] == ["v1", "runs"] \
                        and parts[3] == "stop":
                    return self._json(200, svc.stop(parts[2]))
            return self._json(404, {"error": "Unknown endpoint"})
        except (ConfigurationError, ValueError, TypeError, KeyError) as exc:
            return self._json(400, {"error": str(exc)})
        except FileNotFoundError as exc:
            return self._json(404, {"error": str(exc)})
        except json.JSONDecodeError as exc:
            return self._json(400, {"error": f"Invalid JSON: {exc}"})
        except Exception:
            return self._json(500, {"error": "Internal service error"})

    def do_GET(self):
        self._serve()

    def do_POST(self):
        self._serve()

def run_server(port=8766, workspace=None):
    if not (1024 <= int(port) <= 65535):
        raise ValueError("Port must be 1024..65535")
    if int(port) == 8765:
        raise ValueError("Port 8765 is reserved for E-Brush Studio; use 8766")
    service = RCService(workspace=workspace)
    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), RCHandler)
    httpd.rcservice = service
    print(f"RCStudio local service on http://127.0.0.1:{port}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        service.pool.shutdown(wait=False, cancel_futures=True)
