"""Saved actions export without borrowing another take's duration or root settings."""

from PySide6.QtWidgets import QFileDialog

from desktop.ui.app_controller import StudioController
from desktop.ui.pages.pages_export_system import AnimationExportsPage


def test_existing_scene_uses_its_own_animation_without_an_active_take(qtbot, tmp_path, monkeypatch):
    controller = StudioController(tmp_path/"settings.json")
    executable = tmp_path/"blender.exe"
    executable.write_bytes(b"checked by the mocked export runner")
    controller.settings.blender.blender_executable = str(executable)
    scene = tmp_path/"already_corrected.blend"
    scene.write_bytes(b"existing saved scene")
    page = AnimationExportsPage(controller)
    qtbot.addWidget(page)
    output = tmp_path/"corrected.glb"
    monkeypatch.setattr(QFileDialog,"getSaveFileName",lambda *args,**kwargs:(str(output),""))
    calls = []

    def run(executable, settings, cancel):
        calls.append(settings)
        return {"file":str(output),"samples":15,"max_joint_error_m":0.,"passed":True}

    monkeypatch.setattr("engine.export.blender_job.run_blender_export",run)
    try:
        assert controller.active_motion is None
        page.character_file.setText(str(scene))
        page.target_armature.setText("ChosenActor")
        page.in_place.setChecked(True)
        page.keep_scene_action.setChecked(True)
        assert not page.in_place.isEnabled()
        page._exp_3d(".glb")
        qtbot.waitUntil(lambda:len(calls)==1)
        assert calls[0]["scene_file"]==str(scene)
        assert calls[0]["frame_end"] is None
        assert not calls[0].get("in_place",False)
        assert "fps" not in calls[0] and "motion_json" not in calls[0]
        page.keep_scene_action.setChecked(False)
        assert page.in_place.isEnabled() and page.in_place.isChecked()
    finally:
        page.shutdown()
        controller.shutdown()
