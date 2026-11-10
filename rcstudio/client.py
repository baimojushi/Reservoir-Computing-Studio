"""Shared client for CLI, GUI and MCP stdio adapter."""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from .storage import Workspace

class ServiceError(RuntimeError):
    pass

def current_port():
    return int(os.environ.get("RCSTUDIO_PORT", "8766"))

class Client:
    def __init__(self, port=None, permission="write", workspace=None):
        self.port = port or current_port()
        self.workspace = workspace or Workspace()
        tokens = self.workspace.credentials()
        if permission not in ("read", "write"):
            raise ValueError("permission: read or write")
        self.token = tokens[f"{permission}_token"]
        self.base_url = f"http://127.0.0.1:{self.port}"

    def request(self, method, endpoint, payload=None, timeout=4):
        url = self.base_url + endpoint
        data = None
        if payload is not None:
            data = json.dumps(payload, allow_nan=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                problem = json.loads(exc.read().decode("utf-8"))
            except Exception:
                problem = {"error": str(exc)}
            raise ServiceError(f"HTTP {exc.code}: {problem.get('error')}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ServiceError(f"RCStudio service unavailable: {exc}") from exc

    def health(self):
        return self.request("GET", "/v1/health", timeout=0.5)

    def workspace_state(self):
        return self.request("GET", "/v1/workspace")

    def runs(self, limit=30):
        return self.request("GET", f"/v1/runs?limit={int(limit)}")["runs"]

    def datasets(self):
        return self.request("GET", "/v1/datasets")["datasets"]

    def projects(self):
        return self.request("GET", "/v1/projects")["projects"]

    def ebrush_sessions(self):
        return self.request("GET", "/v1/projects/ebrush/sessions")["sessions"]

    def run(self, run_id):
        return self.request("GET", f"/v1/runs/{run_id}")

    def traces(self, run_id, limit=600, units=24):
        return self.request(
            "GET", f"/v1/runs/{run_id}/traces?limit={int(limit)}&units={int(units)}",
            timeout=10,
        )

    def diagnostics(self):
        return self.request("GET", "/v1/diagnostics")

    def submit(self, config):
        return self.request("POST", "/v1/runs", {"config": config}, timeout=10)

    def submit_ebrush(self, config=None):
        return self.request("POST", "/v1/projects/ebrush/train",
                            {"config": config or {}} , timeout=10)

    def stop(self, run_id):
        return self.request("POST", f"/v1/runs/{run_id}/stop", {})

def ensure_server(port=None, workspace=None):
    """Start a detached loopback service if absent (Windows and Unix)."""
    client = Client(port=port, workspace=workspace)
    try:
        status = client.health()
        if status.get("app") == "RCStudio":
            if "ebrush_motion_v1" not in status.get("projects", []):
                raise ServiceError("Port occupied by an older RC Studio service; stop that service before starting V0.3")
            client.workspace_state()  # Also confirm our token matches.
            return client
        raise ServiceError("Port occupied by an unexpected HTTP server")
    except ServiceError as exc:
        if "unexpected" in str(exc) or "older RC Studio" in str(exc) or "401" in str(exc):
            raise
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "env": os.environ.copy(),
        "close_fds": True,
    }
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(
        [sys.executable, "-m", "rcstudio", "serve", "--port", str(client.port)],
        **kwargs,
    )
    for _ in range(60):
        try:
            health = client.health()
            if health.get("app") == "RCStudio" and "ebrush_motion_v1" in health.get("projects", []):
                client.workspace_state()
                return client
        except ServiceError:
            pass
        time.sleep(0.1)
    raise ServiceError("Failed to start the local RCStudio service")
