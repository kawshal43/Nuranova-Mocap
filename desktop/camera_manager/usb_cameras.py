"""USB & Local Connected Camera Discovery, Real-Time Preview Streaming, and Synchronized MP4 Recording.

Supports any UVC USB webcam, HDMI/SDI USB capture card, virtual camera, or integrated PC camera
via OpenCV (DirectShow / MediaFoundation on Windows, V4L2/AVFoundation elsewhere) + QtMultimedia
friendly device naming. Integrates seamlessly with CameraManager so USB cameras work alongside
or instead of Android Wi-Fi phones across Calibration, 1-Cam AI Mode, Live 3D/Finger Preview,
Blender Live Preview, and Synchronized Multi-Cam Recording.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from desktop.camera_manager.manager import CameraManager, ManagedCameraRecord
from desktop.synchronization.sync_engine import ClockModel
from shared_protocol.protocol import (
    CameraConnectionState,
    CameraRole,
    DeviceTelemetry,
    SupportedMode,
    TimestampSource,
)


@dataclass
class DiscoveredUSBCamera:
    index: int
    name: str
    backend_name: str
    width: int
    height: int
    fps: float
    device_id: str
    already_connected: bool = False
    supported_modes: Optional[List[SupportedMode]] = None


def _query_qt_camera_metadata() -> List[Tuple[str, List[SupportedMode]]]:
    """Use PySide6 QtMultimedia (fast, zero-subprocess) to get friendly USB camera names & supported modes."""
    results: List[Tuple[str, List[SupportedMode]]] = []
    try:
        from PySide6.QtMultimedia import QMediaDevices

        for dev in QMediaDevices.videoInputs():
            name = str(dev.description() or "").strip() or "USB Camera"
            modes_map: Dict[Tuple[int, int], float] = {}
            try:
                for fmt in dev.videoFormats():
                    sz = fmt.resolution()
                    w, h = int(sz.width()), int(sz.height())
                    max_fps = float(fmt.maxFrameRate() or 30.0)
                    if w >= 320 and h >= 240:
                        prev = modes_map.get((w, h), 0.0)
                        if max_fps > prev:
                            modes_map[(w, h)] = max_fps
            except Exception:
                pass
            modes = [
                SupportedMode(
                    width=w,
                    height=h,
                    max_fps=round(fps, 1),
                    fps_ranges=[[min(30, int(round(fps))), int(round(fps))]],
                )
                for (w, h), fps in sorted(modes_map.items(), key=lambda kv: (kv[0][0] * kv[0][1], kv[1]))
            ]
            results.append((name, modes))
    except Exception:
        pass
    return results


def _open_cv_capture(index: int) -> Tuple[Optional[cv2.VideoCapture], str]:
    """Open an OpenCV VideoCapture cleanly, preferring DirectShow on Windows for reliable USB webcams."""
    if sys.platform.startswith("win"):
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MediaFoundation"), (cv2.CAP_ANY, "OpenCV")]
    else:
        backends = [(cv2.CAP_ANY, "OpenCV")]

    for api, label in backends:
        cap = cv2.VideoCapture(int(index), api)
        if cap is not None and cap.isOpened():
            return cap, label
        if cap is not None:
            cap.release()
    return None, "Unavailable"


def discover_usb_cameras(
    max_indices: int = 8,
    active_streams: Optional[Dict[int, "USBCameraStream"]] = None,
) -> List[DiscoveredUSBCamera]:
    """Discover all USB / local cameras connected to this PC without interrupting active streams."""
    active_by_idx = active_streams or {}
    qt_meta = _query_qt_camera_metadata()
    discovered: List[DiscoveredUSBCamera] = []

    # When QtMultimedia already enumerated all Windows video inputs, probe exactly those indices
    # (fast <0.1s scan with zero out-of-range DirectShow/MSMF stderr noise); otherwise fallback probe.
    if qt_meta:
        limit = max(len(qt_meta), (max(active_by_idx.keys()) + 1) if active_by_idx else 0)
    else:
        limit = min(max_indices, 10)
    consecutive_misses = 0

    for idx in range(limit):
        # Never double-open a camera that is already actively streaming (DirectShow locks active cams!)
        if idx in active_by_idx:
            st = active_by_idx[idx]
            discovered.append(
                DiscoveredUSBCamera(
                    index=idx,
                    name=st.friendly_name,
                    backend_name=st.backend_name,
                    width=st.width,
                    height=st.height,
                    fps=st.target_fps,
                    device_id=st.device_id,
                    already_connected=True,
                    supported_modes=st.supported_modes,
                )
            )
            consecutive_misses = 0
            continue

        # In headless/offscreen CI tests without physical devices, avoid probing empty indices slowly
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen" and idx >= len(qt_meta):
            break

        cap, backend_name = _open_cv_capture(idx)
        if cap is None:
            consecutive_misses += 1
            if idx >= len(qt_meta) and consecutive_misses >= 2:
                break
            continue

        consecutive_misses = 0
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if w <= 0 or h <= 0:
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
            if w <= 0 or h <= 0:
                w, h = 1280, 720
            if not (5.0 <= fps <= 240.0):
                fps = 30.0
        finally:
            cap.release()

        if idx < len(qt_meta):
            friendly_name, modes = qt_meta[idx]
        else:
            friendly_name, modes = f"USB Camera #{idx}", []

        discovered.append(
            DiscoveredUSBCamera(
                index=idx,
                name=friendly_name,
                backend_name=backend_name,
                width=w,
                height=h,
                fps=fps,
                device_id=f"usb_cam_{idx}",
                already_connected=False,
                supported_modes=modes or None,
            )
        )

    return discovered


class USBCameraStream:
    """Background capture thread for one connected USB camera: live JPEG preview + synchronized MP4 recording."""

    def __init__(
        self,
        index: int,
        device_id: str,
        camera_uid: str,
        friendly_name: str,
        camera_manager: CameraManager,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 30.0,
        supported_modes: Optional[List[SupportedMode]] = None,
        capture_factory: Optional[Callable[[int], Tuple[Optional[object], str]]] = None,
    ) -> None:
        self.index = int(index)
        self.device_id = device_id
        self.camera_uid = camera_uid
        self.friendly_name = friendly_name
        self.camera_manager = camera_manager
        self.width = int(width)
        self.height = int(height)
        self.target_fps = float(target_fps)
        self.backend_name = "OpenCV"
        self.supported_modes = list(supported_modes or [])
        self._capture_factory = capture_factory or _open_cv_capture

        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cfg_lock = threading.Lock()
        self._pending_config: Optional[Tuple[int, int, float]] = None

        # Synchronized MP4 recording state
        self._rec_lock = threading.Lock()
        self._rec_active: bool = False
        self._rec_start_ns: int = 0
        self._rec_output_mp4: Optional[Path] = None
        self._rec_writer: Optional[cv2.VideoWriter] = None
        self._rec_timestamps: List[Dict[str, int]] = []
        self._rec_frame_idx: int = 0
        self._rec_last_finished_path: Optional[Path] = None

        self.running: bool = False
        self.last_error: Optional[str] = None

    def start(self) -> bool:
        if self.running and self._thread is not None and self._thread.is_alive():
            return True
        self._stop_evt.clear()
        self.last_error = None
        ready_evt = threading.Event()
        started_ok = [False]

        def _entry() -> None:
            self._run_loop(ready_evt, started_ok)

        self._thread = threading.Thread(target=_entry, name=f"nuranova-usb-cam-{self.index}", daemon=True)
        self._thread.start()
        ready_evt.wait(timeout=4.0)
        return started_ok[0]

    def stop(self) -> None:
        self.stop_recording()
        self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self._thread = None
        self.running = False

    def reconfigure(self, width: int, height: int, target_fps: float) -> None:
        with self._cfg_lock:
            self._pending_config = (int(width), int(height), float(target_fps))

    def start_recording(self, output_mp4: Path, start_at_monotonic_ns: int = 0) -> None:
        output_mp4 = Path(output_mp4)
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        with self._rec_lock:
            self._finalize_writer_locked()
            self._rec_output_mp4 = output_mp4
            self._rec_start_ns = int(start_at_monotonic_ns)
            self._rec_timestamps = []
            self._rec_frame_idx = 0
            self._rec_last_finished_path = None
            self._rec_active = True

        rec = self.camera_manager.get_camera(self.device_id)
        if rec is not None:
            rec.state = CameraConnectionState.RECORDING

    def stop_recording(self) -> Optional[Path]:
        with self._rec_lock:
            was_active = self._rec_active
            self._rec_active = False
            finished = self._finalize_writer_locked()
        if was_active:
            rec = self.camera_manager.get_camera(self.device_id)
            if rec is not None and rec.state == CameraConnectionState.RECORDING:
                rec.state = CameraConnectionState.CONNECTED
        return finished

    def _finalize_writer_locked(self) -> Optional[Path]:
        out_path = self._rec_output_mp4
        if self._rec_writer is not None:
            try:
                self._rec_writer.release()
            except Exception:
                pass
            self._rec_writer = None

        if out_path is not None and self._rec_timestamps:
            ts_path = out_path.with_name(f"{out_path.stem}_timestamps.json")
            try:
                ts_path.write_text(
                    json.dumps(
                        {
                            "camera_uid": self.camera_uid,
                            "device_id": self.device_id,
                            "timestamp_source": TimestampSource.MONOTONIC.value,
                            "fps": self.target_fps,
                            "frame_count": len(self._rec_timestamps),
                            "frames": self._rec_timestamps,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass
            self._rec_last_finished_path = out_path
            self._rec_output_mp4 = None
            self._rec_timestamps = []
            return out_path

        self._rec_output_mp4 = None
        self._rec_timestamps = []
        return self._rec_last_finished_path

    def _apply_capture_props(self, cap: object, w: int, h: int, fps: float) -> Tuple[int, int, float]:
        try:
            # Prefer MJPG on USB webcams so 720p/1080p runs at full 30-60 FPS instead of low-FPS uncompressed YUY2
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
            cap.set(cv2.CAP_PROP_FPS, float(fps))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or w)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or h)
            actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
            if actual_w <= 0 or actual_h <= 0:
                actual_w, actual_h = w, h
            if not (5.0 <= actual_fps <= 240.0):
                actual_fps = fps
            return actual_w, actual_h, actual_fps
        except Exception:
            return w, h, fps

    def _run_loop(self, ready_evt: threading.Event, started_ok: List[bool]) -> None:
        cap, backend_name = self._capture_factory(self.index)
        if cap is None:
            self.last_error = f"Could not open USB camera #{self.index} ({self.friendly_name}). Check if another app is using it."
            ready_evt.set()
            return

        self.backend_name = backend_name
        self.width, self.height, _ = self._apply_capture_props(cap, self.width, self.height, self.target_fps)
        self.running = True
        started_ok[0] = True
        ready_evt.set()

        # Assign local PC clock sync model (0 ms drift & offset since USB camera uses studio PC clock directly!)
        rec = self.camera_manager.get_camera(self.device_id)
        if rec is not None:
            rec.clock_model = ClockModel(
                camera_uid=self.camera_uid,
                slope=1.0,
                intercept_ns=0.0,
                t0_client_ns=0.0,
                drift_ppm=0.0,
                residual_rmse_ms=0.1,
                min_rtt_ms=0.0,
                median_rtt_ms=0.0,
                sample_count=1,
                meets_5ms_target=True,
            )

        last_frame_t = time.monotonic()
        last_tel_t = 0.0
        ema_fps = float(self.target_fps)
        consecutive_fails = 0

        try:
            while not self._stop_evt.is_set():
                # Apply pending live resolution/FPS reconfiguration safely on capture thread
                pending = None
                with self._cfg_lock:
                    if self._pending_config is not None:
                        pending = self._pending_config
                        self._pending_config = None
                if pending is not None:
                    pw, ph, pfps = pending
                    self.width, self.height, _ = self._apply_capture_props(cap, pw, ph, pfps)
                    self.target_fps = pfps

                ok, frame = cap.read()
                ts_ns = time.monotonic_ns()
                if not ok or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails > 45:
                        self.last_error = f"USB camera '{self.friendly_name}' stopped delivering frames."
                        self.camera_manager.mark_disconnected(self.device_id)
                        break
                    time.sleep(0.015)
                    continue

                consecutive_fails = 0
                fh, fw = frame.shape[:2]
                self.width, self.height = fw, fh

                # Synchronized PC-local MP4 recording
                with self._rec_lock:
                    if self._rec_active and self._rec_output_mp4 is not None and ts_ns >= self._rec_start_ns:
                        if self._rec_writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                            fps_write = max(10.0, min(120.0, self.target_fps))
                            self._rec_writer = cv2.VideoWriter(
                                str(self._rec_output_mp4),
                                fourcc,
                                fps_write,
                                (fw, fh),
                            )
                        if self._rec_writer is not None and self._rec_writer.isOpened():
                            self._rec_writer.write(frame)
                            self._rec_timestamps.append(
                                {
                                    "frame_index": self._rec_frame_idx,
                                    "sensor_timestamp_ns": ts_ns,
                                    "pts_us": ts_ns // 1000,
                                }
                            )
                            self._rec_frame_idx += 1

                # Encode smooth JPEG preview (<= 960px wide @ q=82) for Calibration, Capture Studio & AI Live Preview
                if fw > 960:
                    scale = 960.0 / float(fw)
                    prev_img = cv2.resize(frame, (960, max(1, int(round(fh * scale)))), interpolation=cv2.INTER_AREA)
                else:
                    prev_img = frame
                enc_ok, jpg = cv2.imencode(".jpg", prev_img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                if enc_ok:
                    self.camera_manager.update_preview_frame(self.device_id, jpg.tobytes())

                now = time.monotonic()
                dt = max(now - last_frame_t, 1e-3)
                ema_fps = 0.20 * (1.0 / dt) + 0.80 * ema_fps
                last_frame_t = now

                if now - last_tel_t >= 1.0:
                    last_tel_t = now
                    try:
                        free_mb = shutil.disk_usage(Path.home()).free / (1024.0 * 1024.0)
                    except Exception:
                        free_mb = 0.0
                    is_rec = self._rec_active
                    self.camera_manager.update_telemetry(
                        DeviceTelemetry(
                            device_id=self.device_id,
                            battery_percent=100.0,
                            temperature_celsius=None,
                            available_storage_mb=round(free_mb, 1),
                            measured_fps=round(ema_fps, 1),
                            recording_state="recording" if is_rec else "idle",
                            focus_locked=False,
                            wb_locked=False,
                        )
                    )
        finally:
            with self._rec_lock:
                self._rec_active = False
                self._finalize_writer_locked()
            try:
                cap.release()
            except Exception:
                pass
            self.running = False


class USBCameraManager:
    """Coordinates discovery, live streaming, reconfiguration, and synchronized recording for all USB cameras."""

    def __init__(
        self,
        camera_manager: CameraManager,
        capture_factory: Optional[Callable[[int], Tuple[Optional[object], str]]] = None,
    ) -> None:
        self.camera_manager = camera_manager
        self._capture_factory = capture_factory
        self._lock = threading.RLock()
        self._streams_by_device: Dict[str, USBCameraStream] = {}
        self._streams_by_index: Dict[int, USBCameraStream] = {}

    def scan_cameras(self, max_indices: int = 8) -> List[DiscoveredUSBCamera]:
        with self._lock:
            active_copy = dict(self._streams_by_index)
        return discover_usb_cameras(max_indices=max_indices, active_streams=active_copy)

    def connect_camera(
        self,
        index: int,
        name: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        role: CameraRole = CameraRole.BODY,
        supported_modes: Optional[List[SupportedMode]] = None,
    ) -> ManagedCameraRecord:
        index = int(index)
        device_id = f"usb_cam_{index}"
        with self._lock:
            if device_id in self._streams_by_device and self._streams_by_device[device_id].running:
                existing = self.camera_manager.get_camera(device_id)
                if existing is not None:
                    return existing

        friendly = (name or "").strip() or f"USB Camera #{index}"
        modes = list(supported_modes or [])
        if not modes:
            modes = [
                SupportedMode(width=640, height=480, max_fps=60.0, fps_ranges=[[24, 60]]),
                SupportedMode(width=1280, height=720, max_fps=60.0, fps_ranges=[[24, 60]]),
                SupportedMode(width=1920, height=1080, max_fps=60.0, fps_ranges=[[24, 60]]),
                SupportedMode(width=2560, height=1440, max_fps=30.0, fps_ranges=[[24, 30]]),
                SupportedMode(width=3840, height=2160, max_fps=30.0, fps_ranges=[[24, 30]]),
            ]

        rec = self.camera_manager.register_usb_camera(
            device_id=device_id,
            name=friendly,
            usb_index=index,
            width=width,
            height=height,
            fps=fps,
            role=role,
            supported_modes=modes,
        )

        stream = USBCameraStream(
            index=index,
            device_id=device_id,
            camera_uid=rec.camera_uid,
            friendly_name=friendly,
            camera_manager=self.camera_manager,
            width=width,
            height=height,
            target_fps=fps,
            supported_modes=modes,
            capture_factory=self._capture_factory,
        )
        ok = stream.start()
        if not ok:
            err = stream.last_error or f"Failed to open USB Camera #{index} ({friendly})."
            self.camera_manager.remove_camera(device_id)
            raise RuntimeError(err)

        with self._lock:
            self._streams_by_device[device_id] = stream
            self._streams_by_index[index] = stream

        # Update record config with actual opened resolution
        rec.config = rec.config.model_copy(update={"width": stream.width, "height": stream.height, "target_fps": fps})
        return rec

    def disconnect_camera(self, device_id: str) -> bool:
        with self._lock:
            stream = self._streams_by_device.pop(device_id, None)
            if stream is not None:
                self._streams_by_index.pop(stream.index, None)
        if stream is not None:
            stream.stop()
            return True
        return False

    def reconfigure_camera(self, device_id: str, width: int, height: int, fps: float) -> bool:
        with self._lock:
            stream = self._streams_by_device.get(device_id)
        if stream is None:
            return False
        stream.reconfigure(width, height, fps)
        return True

    def active_device_ids(self) -> List[str]:
        with self._lock:
            return [did for did, st in self._streams_by_device.items() if st.running]

    def start_recording_all(
        self,
        recordings_root: Path,
        take_id: str,
        start_at_monotonic_ns: int = 0,
    ) -> List[str]:
        """Start synchronized local MP4 recording on all enabled & connected USB cameras."""
        started_devices: List[str] = []
        take_dir = Path(recordings_root) / take_id
        take_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            streams = list(self._streams_by_device.values())
        for st in streams:
            rec = self.camera_manager.get_camera(st.device_id)
            if rec is None or not rec.enabled or not st.running:
                continue
            out_mp4 = take_dir / f"{rec.camera_uid}_{take_id}.mp4"
            st.start_recording(out_mp4, start_at_monotonic_ns=start_at_monotonic_ns)
            started_devices.append(st.device_id)
        return started_devices

    def stop_recording_all(self) -> Dict[str, Path]:
        """Stop all active USB camera recordings and return `{camera_uid: mp4_path}`."""
        saved: Dict[str, Path] = {}
        with self._lock:
            streams = list(self._streams_by_device.values())
        for st in streams:
            mp4 = st.stop_recording()
            if mp4 is not None and mp4.exists():
                saved[st.camera_uid] = mp4
        return saved

    def shutdown(self) -> None:
        with self._lock:
            streams = list(self._streams_by_device.values())
            self._streams_by_device.clear()
            self._streams_by_index.clear()
        for st in streams:
            st.stop()
