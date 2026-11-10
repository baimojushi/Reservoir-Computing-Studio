"""RC teaching adapter. Only cached RC results are observed; no device or E-Brush calls."""
from pathlib import Path
from . import anchor
PROJECT_ID = "ebrush_motion_v1"

def selected_metadata(window):
    selected = set(window._selected_sessions())
    return [m for m in window.latest_sessions if m["id"] in selected]

def selection_kind(entries):
    if not entries:
        return "none"
    if all(m.get("synthetic") is True for m in entries):
        return "demo"
    if all(m.get("synthetic") is False for m in entries):
        return "real"
    return "mixed"

def submitted(window, cfg, run_id):
    """Called only after the existing service confirms actual submission."""
    manager = getattr(window, "learning", None)
    if manager is None:
        return
    entries = selected_metadata(window)
    manager.record_context("submitted_run", str(run_id))
    manager.record_context("submitted_project", cfg.get("project", "generic_esn"))
    manager.record_context("submitted_data_kind", cfg.get("dataset", {}).get("kind", ""))
    manager.record_context("submitted_selection_kind", selection_kind(entries))
    manager.record_context("submitted_recordings", len({m["recording_id"] for m in entries}))
    manager.notify("rc.run.submitted")

def evaluate(window, name, session):
    """Read existing GUI state/cache only; polling does not create extra HTTP requests."""
    project = window.project_box.currentData()
    if name == "rc.generic":
        return project == "generic_esn", "当前项目：" + str(project)
    if name == "rc.ebrush":
        return project == PROJECT_ID, "当前项目：" + str(project)
    if name == "rc.sine":
        info = window.dataset_box.currentData() or {}
        return project == "generic_esn" and info.get("kind") == "sine", (
            "当前数据类型：" + str(info.get("kind", "未选中")))
    if name in ("rc.demo_selected", "rc.real_selected"):
        selected = selected_metadata(window)
        independent = len({m["recording_id"] for m in selected})
        kind = selection_kind(selected)
        required = "demo" if name == "rc.demo_selected" else "real"
        ok = project == PROJECT_ID and kind == required and independent >= 3
        return ok, f"勾选 {len(selected)} 个片段、{independent} 份独立记录；数据类别：{kind}。"
    if name == "rc.course_finished":
        context = session.progress.context
        wanted = context.get("submitted_run")
        row = getattr(window, "_learning_last_run", None) or {}
        if not wanted:
            return False, "尚无本课程期间成功提交的实验；历史成功记录不算本次实操。"
        if row.get("id") != wanted:
            return False, f"请在实验表中选择本课程的实验 {wanted}；不会自动切换或提交实验。"
        course_id = session.course["id"]
        cfg = row.get("config") or {}
        if course_id == "rc_first":
            task_ok = cfg.get("project", "generic_esn") == "generic_esn" and cfg.get("dataset", {}).get("kind") == "sine"
        else:
            needed = "demo" if course_id == "rc_ebrush_demo" else "real"
            task_ok = (cfg.get("project") == PROJECT_ID
                       and context.get("submitted_selection_kind") == needed
                       and context.get("submitted_recordings", 0) >= 3)
        if not task_ok:
            return False, "此实验的数据/项目不符合本课程目标。请核对项目与样本选择，再自行提交。"
        status = row.get("status", "unknown")
        if status in ("failed", "interrupted", "cancelled"):
            return False, f"本课程实验状态为 {status}，不算成功。先检查诊断，再决定是否重试。"
        return status == "finished", f"本课程实验 {wanted}：{status}。"
    return False, "本入口没有该状态检查；可暂停或明确跳过。"

def install(window):
    from PySide6 import QtWidgets
    from .qt import LearningManager
    for attr, key in {
        "topology": "rc.topology", "project_caption": "rc.project",
        "horizon_caption": "rc.model.horizon", "log_text": "rc.diagnostics",
        "session_summary": "rc.sessions", "connection": "rc.connection",
    }.items():
        anchor(getattr(window, attr), key)
    for form in window.findChildren(QtWidgets.QFormLayout):
        for row in range(form.rowCount()):
            field = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.FieldRole)
            label = form.itemAt(row, QtWidgets.QFormLayout.ItemRole.LabelRole)
            if field and label and field.widget() and label.widget():
                key = field.widget().property("learningHelpId")
                if key: anchor(label.widget(), key)
    def reveal(key):
        # Do not change project_box: it would also reset model defaults.
        target = {"rc.plot": window.plot, "rc.states": window.heatmap,
                  "rc.diagnostics": window.metrics_panel}.get(key)
        if target is not None:
            window.tabs.setCurrentWidget(target)
    return LearningManager(
        window, Path(__file__).parent, app_id="RCStudio", app_name="RC Studio",
        mode=window.theme, reveal_hook=reveal,
        predicate_hook=lambda name, session: evaluate(window, name, session))
