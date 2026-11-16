"""Qt host for the V0.7 intuitive lab (``rcstudio/lab_web/lab.html``).

The lab itself is a web page, so what the user sees here is the same thing the
automated Chromium checks render. This wrapper only loads the page, pushes the
theme and real-run data into it, and listens for the few requests it sends back.
Importing this module raises ``ImportError`` when QtWebEngine is unavailable;
``gui.py`` then falls back to the native V0.6 panel.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, QUrl, QUrlQuery, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .lab_web import (
    EDGE_COUNT, FRAME_COUNT, js_call, page_path, parse_host_message, run_payload,
    run_positions, segment_start,
)


class _RunLoader(QThread):
    """Fetch one run's structure and a frame window off the GUI thread."""
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, client, run_id, start=0, with_structure=True, parent=None):
        super().__init__(parent)
        self.client = client
        self.run_id = run_id
        self.first_frame = int(start)
        self.with_structure = bool(with_structure)

    def run(self):
        try:
            result = {"run_id": self.run_id}
            if self.with_structure:
                structure = self.client.reservoir(self.run_id, edges=EDGE_COUNT)
                result["structure"] = structure
                result["positions"] = run_positions(structure["units"])
            result["frames"] = self.client.reservoir_frames(
                self.run_id, start=self.first_frame, count=FRAME_COUNT)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class LabWebPanel(QWidget):
    """Same public surface as the V0.6 panel: set_theme, show_run, shutdown."""
    workbench_requested = Signal()
    chapter_completed = Signal(str, str)      # chapter key, what the learner actually did
    is_web = True

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.theme = "light"
        self.run_id = None
        self.structure = None
        self.positions = None
        self.frames = None
        self.loader = None
        self._pending = None          # (run_id, start, with_structure, focus)
        self._request = None
        self._ready = False
        self._queued_js = []
        self._sent = False            # has the page received this run's structure yet?

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView(self)
        # The page explains itself; keep the Qt hover-help card from covering it.
        self.view.setProperty("learningUi", True)
        self.view.setMinimumHeight(420)
        layout.addWidget(self.view)
        self.view.loadFinished.connect(self._page_loaded)
        self.view.titleChanged.connect(self._host_message)
        self._load_page()

    # ---- page plumbing -------------------------------------------------
    def _load_page(self):
        url = QUrl.fromLocalFile(str(page_path()))
        query = QUrlQuery()
        query.addQueryItem("embedded", "1")
        query.addQueryItem("theme", self.theme)
        url.setQuery(query)
        self._ready = False
        self.view.load(url)

    def _page_loaded(self, ok):
        self._ready = bool(ok)
        if not self._ready:
            return
        queued, self._queued_js = self._queued_js, []
        self.view.page().runJavaScript(js_call("setTheme", self.theme))
        for code in queued:
            self.view.page().runJavaScript(code)

    def _js(self, function, *args):
        code = js_call(function, *args)
        if self._ready:
            self.view.page().runJavaScript(code)
        else:
            self._queued_js.append(code)

    def _host_message(self, title):
        message = parse_host_message(title)
        if message is None:
            return
        kind = message["type"]
        if kind == "workbench":
            self.workbench_requested.emit()
        elif kind == "progress":
            self.chapter_completed.emit(str(message.get("chapter", "")), str(message.get("detail", ""))[:200])
        elif kind == "segment":
            self._load_segment(str(message.get("key", "")))

    # ---- public surface ------------------------------------------------
    def set_theme(self, theme):
        self.theme = theme if theme in ("light", "dark") else "light"
        self._js("setTheme", self.theme)

    def show_run(self, run_id, focus=False):
        """Load a finished run. ``focus`` is True only when the user asked for it."""
        if not run_id:
            return
        busy = self.loader is not None and self.loader.isRunning()
        if run_id == self.run_id and (self.frames is not None or busy or self._pending is not None):
            if focus and self.frames is not None:
                self._js("go", 5)
            return
        self.run_id = run_id
        self.structure = None
        self.positions = None
        self.frames = None
        self._sent = False
        self._pending = (run_id, 0, True, bool(focus))
        if self.isVisible():
            self._start_pending()

    def shutdown(self):
        if self.loader is not None and self.loader.isRunning():
            self.loader.wait(3000)

    # ---- run loading ---------------------------------------------------
    def _load_segment(self, key):
        if self.frames is None or self.run_id is None:
            return
        start = segment_start(self.frames, key)
        if start is None:
            return
        self._pending = (self.run_id, start, False, True)
        self._start_pending()

    def _start_pending(self):
        if self._pending is None or (self.loader is not None and self.loader.isRunning()):
            return
        run_id, start, with_structure, _focus = self._pending
        self._request = self._pending
        self._pending = None
        self.loader = _RunLoader(self.client, run_id, start, with_structure, self)
        self.loader.completed.connect(self._loaded)
        self.loader.failed.connect(self._load_failed)
        self.loader.finished.connect(self._loader_finished)
        self.loader.start()

    def _loader_finished(self):
        if self.loader is not None:
            self.loader.wait()
        if self._pending is not None:
            self._start_pending()

    def _load_failed(self, message):
        if self._request and self._request[0] == self.run_id:
            self._js("notice", f"实验数据读取失败：{message}")

    def _loaded(self, result):
        if result.get("run_id") != self.run_id or self._request is None:
            return
        focus = bool(self._request[3])
        first_load = "structure" in result
        if first_load:
            self.structure = result["structure"]
            self.positions = result["positions"]
        self.frames = result["frames"]
        if first_load:
            # Open on the training segment, as V0.6 did, when the trace starts elsewhere.
            train_start = self.frames.get("split_starts", {}).get("train")
            if train_start not in (None, 0):
                self._pending = (self.run_id, int(train_start), False, focus)
                return
            self._send_run(focus)
        elif self.structure is not None:
            if self._sent:
                self._js("loadFrames", self.frames)
            else:
                self._send_run(focus)

    def _send_run(self, focus):
        payload = run_payload(self.structure, self.frames, self.run_id)
        payload["positions"] = self.positions or payload["positions"]
        self._sent = True
        self._js("loadRun", payload, {"focus": bool(focus)})

    def showEvent(self, event):
        super().showEvent(event)
        if self._pending is not None:
            self._start_pending()
