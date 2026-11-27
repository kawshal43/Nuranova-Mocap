"""Generate printable calibration boards, wearable AprilTag sheets, Blender extension zip,
and a complete portable sample project for Nuranova Mocap Studio v1.0.0.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from desktop.calibration.charuco_board import (
    ChArUcoBoardConfig,
    generate_printable_charuco_pdf,
    render_charuco_png,
)
from desktop.ui.app_controller import StudioController
from engine.export.bvh_exporter import export_bvh
from engine.marker_tracking.marker_kit import (
    create_default_cinema_marker_kit,
    generate_printable_marker_sheet_pdf,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    printable_dir = root / "documentation" / "printable"
    printable_dir.mkdir(parents=True, exist_ok=True)

    # 1. Printable A4 & A3 ChArUco Calibration Boards
    cfg_a4 = ChArUcoBoardConfig(squares_x=7, squares_y=5, square_length_mm=35.0, marker_length_mm=26.0, dictionary_name="DICT_5X5_250")
    generate_printable_charuco_pdf(cfg_a4, printable_dir / "Nuranova_ChArUco_A4_7x5_35mm.pdf", paper_size="A4", dpi=300)
    render_charuco_png(cfg_a4, printable_dir / "Nuranova_ChArUco_A4_7x5_35mm.png", dpi=200)

    cfg_a3 = ChArUcoBoardConfig(squares_x=9, squares_y=6, square_length_mm=40.0, marker_length_mm=30.0, dictionary_name="DICT_5X5_250")
    generate_printable_charuco_pdf(cfg_a3, printable_dir / "Nuranova_ChArUco_A3_9x6_40mm.pdf", paper_size="A3", dpi=300)

    # 2. Printable Wearable AprilTag 36h11 Marker Kit Sheet (valid OpenCV-generated tags!)
    kit = create_default_cinema_marker_kit()
    kit.save(printable_dir / "Nuranova_Default_Cinema_MarkerKit.json")
    tag_markers = [m for m in kit.markers if m.tag_id is not None]
    generate_printable_marker_sheet_pdf(
        tag_ids=[int(m.tag_id) for m in tag_markers],
        output_pdf=printable_dir / "Nuranova_Wearable_AprilTag_36h11_40mm.pdf",
        dictionary_name="DICT_APRILTAG_36h11",
        tag_size_mm=40.0,
        labels=[m.name for m in tag_markers],
    )

    # 3. Installable Blender Extension Zip (`installers/NuranovaMocapBridge.zip`)
    addon_src = root / "blender_addon" / "nuranova_mocap_bridge"
    bridge_zip = root / "installers" / "NuranovaMocapBridge.zip"
    bridge_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bridge_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in addon_src.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                # Blender 4.2+ extension zip puts blender_manifest.toml at archive root or top folder
                zf.write(f, arcname=f"nuranova_mocap_bridge/{f.relative_to(addon_src).as_posix()}")
        ex_map = root / "blender_addon" / "example_rigify_mapping.json"
        if ex_map.exists():
            zf.write(ex_map, arcname="nuranova_mocap_bridge/example_rigify_mapping.json")

    # 4. Complete Portable Sample Motion Capture Project (`sample_projects/Nuranova_Sample_Walk`)
    sample_parent = root / "sample_projects"
    sample_dir = sample_parent / "Nuranova_Sample_Walk"
    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    ctrl = StudioController(settings_path=sample_parent / "_tmp_settings.json")
    try:
        proj = ctrl.create_project(sample_parent, "Nuranova_Sample_Walk")
        doc = ctrl.generate_synthetic_demo_take("sample_synthetic_walk_01")
        ctrl.active_actor.save(proj.subdir("actors") / "default_actor.json")
        kit.save(proj.subdir("character_mappings") / "default_marker_kit.json")
        shutil.copy2(
            root / "blender_addon" / "example_rigify_mapping.json",
            proj.subdir("character_mappings") / "example_rigify_mapping.json",
        )
        export_bvh(doc, proj.subdir("exports") / "sample_synthetic_walk_01.bvh", use_corrected=True)
        ctrl.project_manager.save_project()
    finally:
        ctrl.shutdown()
        (sample_parent / "_tmp_settings.json").unlink(missing_ok=True)

    print("DELIVERABLES_OK")
    for p in [
        printable_dir / "Nuranova_ChArUco_A4_7x5_35mm.pdf",
        printable_dir / "Nuranova_ChArUco_A3_9x6_40mm.pdf",
        printable_dir / "Nuranova_Wearable_AprilTag_36h11_40mm.pdf",
        bridge_zip,
        sample_dir / "project.nmocap",
    ]:
        print(f"  {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
