# -*- mode: python ; coding: utf-8 -*-
# ruff: noqa: F821
"""PyInstaller specification for Nuranova Mocap Studio v1.0.0 (Windows x64 standalone bundle)."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

ROOT = Path(SPECPATH).parent.resolve()

datas = [
    (str(ROOT / "model_registry" / "models.json"), "model_registry"),
    (str(ROOT / "LICENSE_REGISTER.md"), "."),
    (str(ROOT / "licenses" / "THIRD_PARTY_NOTICES.txt"), "licenses"),
    (str(ROOT / "blender_addon" / "example_rigify_mapping.json"), "blender_addon"),
    (str(ROOT / "scripts" / "blender_export_job.py"), "scripts"),
]
datas += [(str(path), "blender_addon/nuranova_mocap_bridge")
          for path in (ROOT / "blender_addon/nuranova_mocap_bridge").iterdir()
          if path.is_file() and path.suffix in {".py", ".toml"}]
datas += collect_data_files("mediapipe")
datas += copy_metadata("mediapipe") + copy_metadata("rtmlib")

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "zeroconf._utils.ipaddress",
    "zeroconf._handlers.answers",
    "h5py.defs",
    "h5py.utils",
    "h5py.h5ac",
    "h5py._proxy",
    "scipy.spatial.transform._rotation_groups",
    "rtmlib",
]

a = Analysis(
    [str(ROOT / "desktop" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib.tests", "pytest", "ruff"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NuranovaMocapStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="NuranovaMocapStudio",
)
