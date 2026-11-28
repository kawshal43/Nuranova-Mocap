"""Cancellable Blender exports using structured job files and unique completion receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import uuid
import shutil
from urllib.parse import quote, unquote


def run_blender_export(executable, settings, cancel=None, progress=None, timeout_s=1800):
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Export cancelled before starting.")
    settings = dict(settings, run_id=uuid.uuid4().hex)
    script = Path(__file__).resolve().parents[2] / "scripts" / "blender_export_job.py"
    with tempfile.TemporaryDirectory(prefix="nuranova_export_job_") as folder:
        job_file, receipt = Path(folder) / "job.json", Path(folder) / "receipt.json"
        settings["receipt"] = str(receipt)
        destination = Path(settings["output"]).resolve()
        staging = Path(folder) / "deliverable" / destination.name
        settings["output"] = str(staging)
        job_file.write_text(json.dumps(settings), encoding="utf-8")
        log_path = Path(folder) / "blender.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen([str(executable), "--background", "--factory-startup", "--python",
                str(script), "--", str(job_file)], stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if cancel is not None and cancel.wait(.15):
                        raise RuntimeError("Export cancelled. The previous completed file is preserved.")
                    if time.monotonic() - started > timeout_s:
                        raise TimeoutError("Blender export exceeded its time limit.")
                    if progress:
                        progress("Blender is baking/exporting and checking the imported animation…")
                    if cancel is None:
                        time.sleep(.15)
                if process.returncode != 0 or not receipt.exists():
                    raise RuntimeError("Blender export failed:\n" + log_path.read_text(encoding="utf-8", errors="replace")[-5000:])
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if result.get("run_id") != settings["run_id"] or not result.get("success"):
            raise RuntimeError(result.get("error", "Blender returned an invalid completion receipt."))
        path = Path(result["file"])
        if not path.exists() or path.stat().st_size != result["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != result["sha256"]:
            raise RuntimeError("The delivered export does not match the completed job.")
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Export cancelled before delivery. The previous file is preserved.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".gltf":
            payload = json.loads(path.read_text(encoding="utf-8"))
            old_resources = path.stem + "_resources"
            new_resources = destination.stem + "_resources_" + settings["run_id"][:12]
            if (path.parent / old_resources).exists():
                shutil.copytree(path.parent / old_resources, destination.parent / new_resources)
            for item in payload.get("buffers", []) + payload.get("images", []):
                uri = unquote(item.get("uri", ""))
                if uri.startswith(old_resources + "/"):
                    item["uri"] = quote(new_resources + uri[len(old_resources):],safe="/")
            path.write_text(json.dumps(payload), encoding="utf-8")
        delivery = destination.with_name(destination.name + "." + settings["run_id"] + ".tmp")
        shutil.copy2(path, delivery)
        if cancel is not None and cancel.is_set():
            delivery.unlink(missing_ok=True)
            raise RuntimeError("Export cancelled before delivery. The previous file is preserved.")
        delivery.replace(destination)
        result.update(file=str(destination), bytes=destination.stat().st_size,
                      sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
        destination.with_suffix(destination.suffix + ".validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
