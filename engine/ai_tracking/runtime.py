"""Hardware acceleration detection and safe ONNX Runtime CUDA/CPU provider selection."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import onnxruntime as ort


@dataclass
class HardwareDiagnostics:
    os_info: str
    cpu_count: int
    gpu_detected: bool
    gpu_name: str
    driver_version: str
    vram_total_mb: int
    vram_free_mb: int
    ort_version: str
    ort_available_providers: List[str]
    cuda_usable: bool
    selected_device: str  # "cuda" or "cpu"
    selected_providers: List[str]
    warnings: List[str] = field(default_factory=list)


def _query_nvidia_smi() -> tuple[bool, str, str, int, int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if not out:
            return False, "None", "N/A", 0, 0
        line = out.splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        return True, parts[0], parts[1], int(float(parts[2])), int(float(parts[3]))
    except Exception:
        return False, "None", "N/A", 0, 0


def _probe_cuda_ep_safe() -> tuple[bool, Optional[str]]:
    """Safely verify whether ONNX Runtime CUDAExecutionProvider can actually load CUDA/cuDNN DLLs."""
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False, "onnxruntime build does not include CUDAExecutionProvider."
    try:
        # Build minimal 1-node Identity ONNX protobuf bytes without external onnx package
        # Instead, check ort.preload_dlls / ctypes cublas/cudnn availability safely
        import ctypes

        for dll_candidates in [
            ("cublas64_12.dll", "cublas64_11.dll"),
            ("cudnn64_9.dll", "cudnn64_8.dll"),
        ]:
            loaded = False
            for dll in dll_candidates:
                try:
                    ctypes.WinDLL(dll)
                    loaded = True
                    break
                except OSError:
                    continue
            if not loaded:
                return (
                    False,
                    f"Required NVIDIA runtime library not found on PATH ({'/'.join(dll_candidates)}). "
                    "Falling back to CPUExecutionProvider safely.",
                )
        return True, None
    except Exception as exc:
        return False, f"CUDA probe error: {exc}"


def detect_hardware(prefer_gpu: bool = True, min_free_vram_mb: int = 1200) -> HardwareDiagnostics:
    """Detect GPU, VRAM headroom, and ONNX Runtime provider health with safe CPU fallback."""
    import os

    gpu_ok, gpu_name, driver, vram_total, vram_free = _query_nvidia_smi()
    providers = list(ort.get_available_providers())
    warnings: List[str] = []

    cuda_usable = False
    if prefer_gpu and gpu_ok:
        ep_ok, err = _probe_cuda_ep_safe()
        if not ep_ok:
            if err:
                warnings.append(err)
        elif vram_free < min_free_vram_mb:
            warnings.append(
                f"Low free GPU VRAM ({vram_free} MiB free / {vram_total} MiB total; recommend >= {min_free_vram_mb} MiB). "
                "Using CPU fallback to prevent CUDA OOM."
            )
        else:
            cuda_usable = True
    elif prefer_gpu and not gpu_ok:
        warnings.append("No NVIDIA GPU detected via nvidia-smi; running on CPUExecutionProvider.")

    selected_device = "cuda" if cuda_usable else "cpu"
    selected_providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if cuda_usable
        else ["CPUExecutionProvider"]
    )

    return HardwareDiagnostics(
        os_info=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu_count=os.cpu_count() or 1,
        gpu_detected=gpu_ok,
        gpu_name=gpu_name,
        driver_version=driver,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        ort_version=str(ort.__version__),
        ort_available_providers=providers,
        cuda_usable=cuda_usable,
        selected_device=selected_device,
        selected_providers=selected_providers,
        warnings=warnings,
    )
