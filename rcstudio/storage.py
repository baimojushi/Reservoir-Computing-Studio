"""Per-user workspace with SQLite WAL, imported datasets and scoped tokens."""
import csv
import json
import os
import re
import secrets
import shutil
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

MAX_CSV_BYTES = 20 * 1024 * 1024

def workspace_dir():
    custom = os.environ.get("RCSTUDIO_HOME")
    if custom:
        return Path(custom).expanduser().resolve()
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "RCStudio"
    return Path.home() / ".local" / "share" / "rcstudio"

class Workspace:
    def __init__(self, root=None):
        self.root = Path(root).expanduser().resolve() if root else workspace_dir()
        self.datasets = self.root / "datasets"
        self.artifacts = self.root / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(exist_ok=True)
        self.artifacts.mkdir(exist_ok=True)
        self.db_path = self.root / "experiments.sqlite3"
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @contextmanager
    def _session(self):
        # sqlite3.Connection.__exit__ only commits; the handle must be closed
        # explicitly or Windows keeps the database file locked until a
        # generational GC pass happens to collect the connection.
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._session() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs(
                    id TEXT PRIMARY KEY, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    config TEXT NOT NULL, metrics TEXT, error TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS ix_runs_created ON runs(created_at DESC)")

    def credentials(self):
        path = self.root / "credentials.json"
        if not path.exists():
            creds = {
                "read_token": secrets.token_urlsafe(32),
                "write_token": secrets.token_urlsafe(32),
            }
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(creds, file)
        return json.loads(path.read_text(encoding="utf-8"))

    def create_run(self, cfg):
        run_id = uuid.uuid4().hex[:16]
        now = time.time()
        with self._session() as conn:
            conn.execute(
                "INSERT INTO runs (id,created_at,updated_at,status,config) VALUES (?,?,?,?,?)",
                (run_id, now, now, "queued", json.dumps(cfg, ensure_ascii=False)),
            )
        return run_id

    def update_run(self, run_id, *, status=None, progress=None, metrics=None, error=None):
        changes = {"updated_at": time.time()}
        if status is not None:
            changes["status"] = status
        if progress is not None:
            changes["progress"] = float(progress)
        if metrics is not None:
            changes["metrics"] = json.dumps(metrics, ensure_ascii=False, allow_nan=False)
        if error is not None:
            changes["error"] = str(error)[:3000]
        sql = "UPDATE runs SET " + ",".join(f"{k}=?" for k in changes) + " WHERE id=?"
        with self._session() as conn:
            conn.execute(sql, (*changes.values(), run_id))

    @staticmethod
    def _convert(row):
        if row is None:
            return None
        item = dict(row)
        item["config"] = json.loads(item["config"])
        item["metrics"] = json.loads(item["metrics"]) if item["metrics"] else None
        return item

    def get_run(self, run_id):
        if not re.fullmatch(r"[a-f0-9]{16}", str(run_id)):
            return None
        with self._session() as conn:
            return self._convert(conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())

    def list_runs(self, limit=30):
        limit = max(1, min(200, int(limit)))
        with self._session() as conn:
            return [
                self._convert(row) for row in conn.execute(
                    "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            ]

    def mark_interrupted(self):
        with self._session() as conn:
            conn.execute(
                "UPDATE runs SET status='interrupted',error='Server restarted before completion' "
                "WHERE status IN ('running','queued')"
            )

    def artifact_dir(self, run_id):
        if not re.fullmatch(r"[a-f0-9]{16}", str(run_id)):
            raise ValueError("Invalid run ID")
        folder = self.artifacts / run_id
        folder.mkdir(exist_ok=True)
        return folder

    def import_csv(self, file_path):
        source = Path(file_path).expanduser().resolve()
        if source.suffix.lower() != ".csv" or not source.is_file():
            raise ValueError("Choose an existing .csv file")
        if source.stat().st_size > MAX_CSV_BYTES:
            raise ValueError("CSV exceeds 20 MB import limit")
        with source.open("r", encoding="utf-8-sig", newline="") as file:
            sample = file.read(8192)
            file.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t") if sample else csv.excel
            reader = csv.reader(file, dialect)
            header = next(reader, [])
            if not header:
                raise ValueError("CSV has no header row")
            header = [h.strip() for h in header]
            if not any(header):
                raise ValueError("CSV has no named columns")
        dataset_id = uuid.uuid4().hex
        dest = self.datasets / f"{dataset_id}.csv"
        shutil.copyfile(source, dest)
        meta = {
            "id": dataset_id, "name": source.name,
            "columns": header[:100], "bytes": dest.stat().st_size,
            "imported_at": time.time(),
        }
        (self.datasets / f"{dataset_id}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta

    def list_datasets(self):
        datasets = []
        for meta_file in self.datasets.glob("*.json"):
            try:
                item = json.loads(meta_file.read_text(encoding="utf-8"))
                if (self.datasets / f"{item['id']}.csv").is_file():
                    datasets.append(item)
            except (KeyError, ValueError, OSError):
                continue
        return sorted(datasets, key=lambda item: item["imported_at"], reverse=True)

    def read_csv_series(self, dataset_id, column=None):
        if not isinstance(dataset_id, str) or not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
            raise ValueError("Invalid dataset ID")
        file_path = self.datasets / f"{dataset_id}.csv"
        if not file_path.is_file():
            raise FileNotFoundError("Imported dataset not found")
        with file_path.open("r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t") if sample else csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            if not reader.fieldnames:
                raise ValueError("CSV has no columns")
            fields = list(reader.fieldnames)
            if column and column not in fields:
                raise ValueError(f"Column '{column}' not found; available: {fields}")
            rows = []
            for i, row in enumerate(reader):
                if i >= 20000:
                    raise ValueError("CSV limit: 20,000 data rows in V0.1")
                rows.append(row)
            if not rows:
                raise ValueError("CSV has no data")
            candidates = [column] if column else fields
            for field in candidates:
                try:
                    values = [float(row[field]) for row in rows]
                except (TypeError, ValueError, KeyError):
                    if column:
                        raise ValueError(f"Column '{column}' contains nonnumeric values")
                    continue
                import math
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Column '{field}' contains NaN / Infinity")
                return values, field
            raise ValueError("No numeric column found; specify a numeric column")
