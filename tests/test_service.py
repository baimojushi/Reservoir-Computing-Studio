import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from rcstudio.client import Client, ServiceError
from rcstudio.service import RCHandler, RCService
from rcstudio.storage import Workspace
from rcstudio.ebrush import make_demo, PROJECT_ID
from rcstudio.service import run_server

class TestService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.workspace = Workspace(Path(cls.temp.name))
        cls.backend = RCService(cls.workspace)
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), RCHandler)
        cls.httpd.rcservice = cls.backend
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.backend.pool.shutdown(wait=True)
        cls.temp.cleanup()

    def test_http_submission_traces_and_tokens(self):
        client = Client(port=self.httpd.server_port, workspace=self.workspace)
        self.assertEqual(client.health()["app"], "RCStudio")
        config = {
            "dataset": {"length": 350, "noise": 0},
            "model": {"units": 26, "washout": 15},
        }
        row = client.submit(config)
        self.assertEqual(row["status"], "queued")
        for _ in range(100):
            row = client.run(row["id"])
            if row["status"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        self.assertEqual(row["status"], "finished", row["error"])
        self.assertIsNotNone(row["metrics"]["scores"]["test"]["rmse"])
        self.assertEqual(len(client.traces(row["id"], limit=60)["timestep"]), 60)
        readonly = Client(
            port=self.httpd.server_port, workspace=self.workspace,
            permission="read",
        )
        self.assertGreaterEqual(len(readonly.runs()), 1)
        with self.assertRaises(ServiceError):
            readonly.submit(config)

    def test_reserved_device_port_refused(self):
        with self.assertRaisesRegex(ValueError, "reserved for E-Brush"):
            run_server(port=8765, workspace=self.workspace)

    def test_ebrush_endpoints_readonly_and_submission(self):
        created = make_demo(self.workspace, count=3, length=220)
        client = Client(port=self.httpd.server_port, workspace=self.workspace)
        readonly = Client(port=self.httpd.server_port, workspace=self.workspace,
                          permission="read")
        self.assertIn(PROJECT_ID, client.health()["projects"])
        self.assertTrue(any(item["id"] == PROJECT_ID for item in readonly.projects()))
        self.assertGreaterEqual(len(readonly.ebrush_sessions()), 3)
        config = {"dataset": {"session_ids": created["session_ids"]},
                  "model": {"units": 18, "washout": 20}}
        with self.assertRaises(ServiceError):
            readonly.submit_ebrush(config)
        with self.assertRaisesRegex(ServiceError, "independent capture files"):
            client.submit_ebrush({"dataset": {"session_ids": [created["session_ids"][0]]}})
        row = client.submit_ebrush(config)
        for _ in range(150):
            row = client.run(row["id"])
            if row["status"] not in ("running", "queued"):
                break
            time.sleep(.03)
        self.assertEqual(row["status"], "finished", row.get("error"))
        self.assertEqual(row["metrics"]["project"], PROJECT_ID)
        self.assertEqual(row["metrics"]["nominal_horizon_ms"], 50)
        self.assertEqual(len(client.traces(row["id"], limit=51)["timestep"]), 51)

if __name__ == "__main__":
    unittest.main()
