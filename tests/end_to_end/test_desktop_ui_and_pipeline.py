"""End-to-end Desktop UI (PySide6) test verifying all 20 pages, Light/Dark themes, 3D viewport,
Motion Editor playback/cleanup/undo-redo, and project creation.
"""

from __future__ import annotations

from pathlib import Path


from desktop.ui.app_controller import StudioController
from desktop.ui.main_window import NuranovaMainWindow
from engine.motion_cleanup.cleanup import MotionCleanupConfig


def test_desktop_studio_20_pages_theme_viewport_and_editor(qtbot, tmp_path: Path):
    ctrl = StudioController(settings_path=tmp_path / "settings.json")
    win = NuranovaMainWindow(controller=ctrl)
    qtbot.addWidget(win)
    win.show()

    # Verify all 20 required pages exist and can be navigated cleanly without exceptions
    expected_pages = [
        "Welcome", "Dashboard", "Projects", "Camera Manager", "Camera Configuration",
        "Calibration Wizard", "Calibration Profiles", "Capture Studio", "Recording Library",
        "Processing Queue", "Motion Editor", "Marker Editor", "Actor Manager",
        "Facial Capture", "Hand Capture", "Character Retargeting", "Animation Exports",
        "Diagnostics", "Settings", "About",
    ]
    assert list(win.pages.keys()) == expected_pages
    assert win.nav_list.count() == 20

    # Create project & generate synthetic demo take
    proj = ctrl.create_project(tmp_path / "projects", "UI_E2E_Project")
    assert proj.manifest_path.exists()
    doc = ctrl.generate_synthetic_demo_take("ui_demo_walk")
    assert doc.num_frames == 90

    # Visit all 20 pages and verify theme toggle (Light <-> Dark)
    for page_name in expected_pages:
        win.navigate_to(page_name)
        qtbot.wait(5)

    win.apply_theme("dark")
    assert ctrl.settings.general.theme == "dark"
    win.apply_theme("light")
    assert ctrl.settings.general.theme == "light"

    # Exercise Motion Editor scrubbing, cleanup, Undo & Redo
    win.navigate_to("Motion Editor")
    me = win.pages["Motion Editor"]
    me.set_frame(25)
    assert me._frame == 25
    ctrl.apply_motion_cleanup(MotionCleanupConfig(strength=0.8, filter_type="one_euro", enable_foot_locking=True))
    assert ctrl.active_motion.corrected_motion is not None
    assert ctrl.undo_motion_edit() is True
    assert ctrl.redo_motion_edit() is True

    # Exercise 3D viewport view presets & repaint cleanly
    for preset in ("perspective", "front", "rear", "left", "right", "top"):
        me.viewport.canvas.set_view_preset(preset)
        me.viewport.canvas.repaint()

    win.close()
