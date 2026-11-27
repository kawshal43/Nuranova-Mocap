"""Integration & scalability tests for 2-12 Camera Network, Auth, Clock Sync, Audio/Flash Sync,
Resumable SHA-256 Transfers, Project Management, and Background Job Queue.

NOTE: Simulated 12-camera client tests validate network/control/transfer scalability ONLY,
NOT physical smartphone optical tracking performance.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from desktop.camera_manager.manager import CameraManager, validate_recording_config_against_capabilities
from desktop.network.server import StudioNetworkServer
from desktop.project_manager.manager import ProjectManager
from desktop.synchronization.sync_engine import (
    ClockSyncSample,
    align_observations_to_common_timeline,
    detect_flash_sync_frame,
    estimate_audio_sync_offset_s,
    fit_clock_sync_model,
    map_frame_timestamps_to_session_timeline,
)
from desktop.workers.job_queue import JobStatus, ProcessingQueueManager
from shared_protocol.protocol import (
    CameraCapabilities,
    CameraRole,
    DeviceTelemetry,
    FrameTimestampRecord,
    RecordingConfig,
    SupportedMode,
    TimestampSource,
)


def test_12_camera_pairing_auth_telemetry_and_resumable_transfer(tmp_path: Path):
    """SCALABILITY_SIMULATION: Pair 12 simulated camera clients, verify auth enforcement, telemetry, and resumable SHA-256 upload."""
    cm = CameraManager(max_body_cameras=12)
    server = StudioNetworkServer(cm, storage_root=tmp_path / "studio_storage")
    client = TestClient(server.app)

    # Health check public
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    # Unauthenticated camera endpoint rejected (401)
    assert client.get("/api/v1/cameras").status_code == 401

    pin = cm.generate_pairing_pin(ttl_seconds=120)

    # Wrong PIN rejected (403)
    bad_pair = client.post(
        "/api/v1/pair",
        json={
            "device_id": "sim_phone_bad",
            "device_name": "Unauthorized Phone",
            "pin": "000000" if pin != "000000" else "999999",
            "capabilities": CameraCapabilities().model_dump(),
        },
    )
    assert bad_pair.status_code == 403

    # Pair 12 simulated body cameras (mixed resolutions & FPS)
    tokens = {}
    resolutions = [(1920, 1080, 60.0), (1920, 1080, 30.0), (3840, 2160, 60.0), (1280, 720, 120.0)]
    for i in range(12):
        w, h, fps = resolutions[i % len(resolutions)]
        caps = CameraCapabilities(
            manufacturer="Simulated",
            model=f"SimPhone-{i+1:02d}",
            timestamp_source=TimestampSource.REALTIME,
            supported_modes=[SupportedMode(width=w, height=h, max_fps=fps, fps_ranges=[[int(fps), int(fps)]])],
            supports_manual_exposure=True,
            supports_manual_focus=True,
            supports_awb_lock=True,
        )
        resp = client.post(
            "/api/v1/pair",
            json={
                "device_id": f"sim_dev_{i+1:02d}",
                "device_name": f"SimCam {i+1:02d}",
                "pin": pin,
                "capabilities": caps.model_dump(),
                "role": CameraRole.BODY.value,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["paired"] is True
        tokens[f"sim_dev_{i+1:02d}"] = body["auth_token"]

    # 13th body camera exceeds max limit (400)
    resp13 = client.post(
        "/api/v1/pair",
        json={
            "device_id": "sim_dev_13",
            "device_name": "SimCam 13",
            "pin": pin,
            "capabilities": CameraCapabilities().model_dump(),
            "role": CameraRole.BODY.value,
        },
    )
    assert resp13.status_code == 400

    # Verify partial availability: disable 8 of 12 -> still able to record with 4 active cameras
    for i in range(4, 12):
        cm.set_camera_enabled(f"sim_dev_{i+1:02d}", False)
    ok, msg = cm.can_record_multi_view()
    assert ok and "4 active" in msg

    # Verify unsupported config detection (never silently changes!)
    cam2_caps = cm.get_camera("sim_dev_02").capabilities  # supports 1080p @ 30fps only
    ok_cfg, issues, alt = validate_recording_config_against_capabilities(
        RecordingConfig(width=3840, height=2160, target_fps=120.0), cam2_caps
    )
    assert not ok_cfg and alt is not None and alt.height == 1080 and alt.target_fps == 30.0

    # Telemetry & NTP clock sync on sim_dev_01
    hdr = {"Authorization": f"Bearer {tokens['sim_dev_01']}", "X-Device-Id": "sim_dev_01"}
    tel = DeviceTelemetry(
        device_id="sim_dev_01",
        battery_percent=84.0,
        temperature_celsius=34.5,
        available_storage_mb=24500.0,
        measured_fps=59.98,
        focus_locked=True,
        wb_locked=True,
    )
    assert client.post("/api/v1/telemetry", json=tel.model_dump(), headers=hdr).status_code == 200
    assert cm.get_camera("sim_dev_01").telemetry.battery_percent == 84.0

    # Resumable chunked transfer with simulated disconnect mid-upload + SHA-256 verification
    payload = (b"NURANOVA_SYNTHETIC_VIDEO_FRAME_PAYLOAD_" * 4000)  # ~156 KB
    expected_sha = hashlib.sha256(payload).hexdigest()

    init_r = client.post(
        "/api/v1/transfers/init",
        json={
            "device_id": "sim_dev_01",
            "session_id": "sess_01",
            "take_id": "take_01",
            "filename": "cam01_raw.mp4",
            "file_kind": "video",
            "total_size_bytes": len(payload),
            "sha256_hex": expected_sha,
        },
        headers=hdr,
    )
    assert init_r.status_code == 200
    tid = init_r.json()["transfer_id"]

    # Send first 60,000 bytes, then simulate reconnect by calling /init again to resume!
    split = 60_000
    c1 = client.put(
        f"/api/v1/transfers/{tid}/chunk",
        content=payload[:split],
        headers={**hdr, "X-Chunk-Offset": "0"},
    )
    assert c1.status_code == 200 and c1.json()["received_bytes"] == split

    resume_r = client.post(
        "/api/v1/transfers/init",
        json={
            "device_id": "sim_dev_01",
            "session_id": "sess_01",
            "take_id": "take_01",
            "filename": "cam01_raw.mp4",
            "file_kind": "video",
            "total_size_bytes": len(payload),
            "sha256_hex": expected_sha,
        },
        headers=hdr,
    )
    assert resume_r.json()["transfer_id"] == tid
    assert resume_r.json()["received_bytes"] == split

    c2 = client.put(
        f"/api/v1/transfers/{tid}/chunk",
        content=payload[split:],
        headers={**hdr, "X-Chunk-Offset": str(split)},
    )
    assert c2.status_code == 200 and c2.json()["received_bytes"] == len(payload)

    fin = client.post(f"/api/v1/transfers/{tid}/finalize", headers=hdr)
    assert fin.status_code == 200
    fin_body = fin.json()
    assert fin_body["verified"] is True
    assert fin_body["can_delete_local_copy"] is True
    assert fin_body["computed_sha256"] == expected_sha

    # WebSocket preview & control ping test
    with client.websocket_connect(f"/ws/v1/camera/sim_dev_01?token={tokens['sim_dev_01']}") as ws:
        ws.send_bytes(b"\xff\xd8\xff\xe0FAKE_JPEG_PREVIEW")
        ws.send_json({"type": "ping_sync", "seq": 1, "client_t1_ns": 1000000})
        pong = ws.receive_json()
        assert pong["type"] == "pong_sync" and pong["seq"] == 1
    assert cm.get_camera("sim_dev_01").latest_preview_jpeg == b"\xff\xd8\xff\xe0FAKE_JPEG_PREVIEW"


def test_clock_drift_audio_flash_and_mixed_fps_synchronization():
    """SYNTHETIC_GROUND_TRUTH: Verify clock drift regression (<1ms), audio clap sync, flash detection, and mixed 30/60 FPS alignment."""
    # True clock model: studio_ns = 1.000040 * device_ns + 5_000_000_000 (+40 ppm drift, +5.0s offset)
    true_slope = 1.000040
    true_offset = 5_000_000_000.0
    rng = np.random.default_rng(11)
    samples = []
    for seq in range(20):
        t1 = int(10_000_000_000 + seq * 200_000_000)  # 4 seconds total span
        rtt = int(rng.uniform(1_200_000, 2_400_000))  # 1.2 - 2.4 ms LAN RTT
        if seq in (3, 14):
            rtt = 45_000_000  # Inject two 45 ms Wi-Fi spike outliers!
        mid_c = t1 + rtt // 2
        t2 = int(round(true_slope * mid_c + true_offset - 50_000))
        t3 = t2 + 100_000
        t4 = t1 + rtt
        samples.append(ClockSyncSample(seq=seq, t1_client_ns=t1, t2_server_ns=t2, t3_server_ns=t3, t4_client_ns=t4))

    model = fit_clock_sync_model("cam_01", samples)
    assert model.meets_5ms_target
    assert model.residual_rmse_ms < 2.0
    assert abs(model.drift_ppm - 40.0) < 15.0

    # Map Camera2 sensor timestamps (with 1 dropped frame) onto studio timeline
    recs = []
    for i in range(60):
        if i == 25:
            continue  # simulate dropped frame
        ts = int(10_000_000_000 + i * (1e9 / 60.0))
        recs.append(FrameTimestampRecord(frame_index=i, sensor_timestamp_ns=ts, pts_us=ts // 1000, monotonic_receive_ns=ts + 8_000_000, exposure_ns=4_000_000))
    rep = map_frame_timestamps_to_session_timeline("cam_01", recs, model, TimestampSource.REALTIME)
    assert len(rep.missing_frame_indices) == 1
    assert abs(rep.measured_fps - 60.0) < 0.5

    # Audio cross-correlation clap sync (true offset = +18.5 ms at 48 kHz)
    sr = 48_000
    t_arr = np.arange(sr) / sr
    clap_a = np.zeros(sr)
    clap_a[12000:12200] = np.sin(2 * np.pi * 1200 * t_arr[:200]) * np.hanning(200)
    shift_samples = int(round(0.0185 * sr))
    clap_b = np.roll(clap_a, shift_samples) + rng.normal(0, 0.005, size=sr)
    est_offset_s, snr = estimate_audio_sync_offset_s(clap_a, clap_b, sample_rate_hz=sr)
    assert abs(est_offset_s - 0.0185) < 0.0005  # < 0.5 ms accuracy!
    assert snr > 10.0

    # Flash sync detection
    luma = np.full(100, 65.0) + rng.normal(0, 0.8, size=100)
    luma[42] = 210.0
    assert detect_flash_sync_frame(luma) == 42

    # Mixed-FPS temporal resampling (cam_01 @ 60 FPS, cam_02 @ 30 FPS)
    ts_60 = np.arange(0, 1_000_000_000, int(1e9 / 60), dtype=np.int64)
    ts_30 = np.arange(0, 1_000_000_000, int(1e9 / 30), dtype=np.int64)
    obs_60 = np.ones((len(ts_60), 2, 3), dtype=np.float32)
    obs_60[:, 0, 0] = ts_60 * 1e-9 * 100.0
    obs_30 = np.ones((len(ts_30), 2, 3), dtype=np.float32)
    obs_30[:, 0, 0] = ts_30 * 1e-9 * 100.0
    canon_ts, aligned = align_observations_to_common_timeline({"cam_01": ts_60, "cam_02": ts_30}, {"cam_01": obs_60, "cam_02": obs_30}, target_fps=60.0)
    assert aligned["cam_01"].shape == aligned["cam_02"].shape
    assert np.max(np.abs(aligned["cam_01"][:, 0, 0] - aligned["cam_02"][:, 0, 0])) < 1e-3


def test_project_lifecycle_and_background_queue(tmp_path: Path):
    pm = ProjectManager(recent_registry_path=tmp_path / "recent.json")
    proj = pm.create_project(tmp_path / "projects", "Short Film Scene 01")
    assert proj.manifest_path.exists()
    assert proj.db_path.exists()
    assert (proj.root_dir / "corrected_motion").is_dir()

    pm.register_take("take_01", "sess_01", "Hero Walk", fps=60.0, num_frames=120)
    takes = pm.list_takes()
    assert len(takes) == 1 and takes[0]["name"] == "Hero Walk"

    # Backup & restore portable zip to a second "PC" directory
    zip_file = pm.backup_or_export_project_zip(tmp_path / "export_portable.nmocap.zip")
    assert zip_file.exists()
    pm.close_project()

    restored = pm.import_or_restore_project_zip(zip_file, tmp_path / "second_pc_project")
    assert restored.name == "Short Film Scene 01"
    assert len(pm.list_takes()) == 1

    # Duplicate & delete with confirmation
    dup = pm.duplicate_project(tmp_path / "projects", "Short Film Scene 01 Copy")
    assert dup.name == "Short Film Scene 01 Copy"
    pm.delete_project(dup.root_dir, confirmed=True)
    assert not dup.root_dir.exists()

    # Background processing queue execution & cancellation
    q = ProcessingQueueManager(max_workers=2)
    job = q.submit(
        "Test Reconstruction",
        "reconstruct",
        lambda prog, cancel: (prog(0.5, "Halfway"), 42)[1],
        check_path=tmp_path,
    )
    for _ in range(50):
        if job.status == JobStatus.COMPLETED:
            break
        time.sleep(0.02)
    assert job.status == JobStatus.COMPLETED and job.result == 42
    q.shutdown()


def test_paired_camera_token_persistence_and_udp_discovery_beacon(tmp_path: Path):
    """Verify paired camera tokens survive Studio restarts (preventing 4401) and UDP beacon responds to probes."""
    import json
    import socket

    from shared_protocol.protocol import PairRequest

    store = tmp_path / "paired_cameras.json"
    cm1 = CameraManager(persistence_path=store)
    pin = cm1.generate_pairing_pin(ttl_seconds=120)
    resp = cm1.pair_device(
        PairRequest(
            device_id="phone_persist_01",
            device_name="Galaxy S20 FE",
            pin=pin,
            capabilities=CameraCapabilities(manufacturer="Samsung", model="SM-G781B"),
        ),
        ip_address="192.168.8.150",
    )
    assert resp.paired and store.exists()

    # Simulate Desktop Studio restart with fresh CameraManager pointing at same persistence file
    cm2 = CameraManager(persistence_path=store)
    authed = cm2.authenticate_token(resp.auth_token, device_id="phone_persist_01")
    assert authed is not None
    assert authed.camera_uid == resp.assigned_camera_uid

    # Verify UDP discovery probe -> unicast reply works on an ephemeral port
    srv = StudioNetworkServer(cm2, storage_root=tmp_path / "storage", port=8765)
    srv.preferred_lan_ip = "192.168.8.124"
    probe_port = 18766
    srv.start_udp_discovery(udp_port=probe_port)
    try:
        time.sleep(0.25)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2.0)
            s.sendto(b'{"type":"nuranova_discover"}', ("127.0.0.1", probe_port))
            data, _ = s.recvfrom(1024)
            beacon = json.loads(data.decode("utf-8"))
            assert beacon["type"] == "nuranova_studio_beacon"
            assert beacon["host"] == "192.168.8.124"
            assert beacon["port"] == 8765
    finally:
        srv.stop_udp_discovery()


def test_bridge_live_pose_and_finger_stream(tmp_path: Path):
    """Verify GET /api/v1/bridge/live serves real-time body + 30-finger frames to Blender."""
    from desktop.workers.live_preview import LiveMocapPreviewWorker, generate_live_procedural_frame
    from engine.skeleton.rig_definition import ActorProfile

    cm = CameraManager()
    srv = StudioNetworkServer(cm, storage_root=tmp_path / "storage")
    client = TestClient(srv.app)

    # Idle before any live pose is published
    r0 = client.get("/api/v1/bridge/live")
    assert r0.status_code == 200
    assert r0.json()["active"] is False

    # Publish a live procedural frame with body + 30 articulated fingers
    actor = ActorProfile(actor_id="test_actor", name="Test Actor")
    _, _, root_t, lq, l_hand, r_hand = generate_live_procedural_frame(1.25, actor)
    worker = LiveMocapPreviewWorker(cm, get_calibration=lambda: None, get_actor=lambda: actor)
    payload = worker._build_live_blender_dict(
        actor=actor,
        root_t=root_t,
        local_q=lq,
        left_hand=l_hand,
        right_hand=r_hand,
        left_detected=True,
        right_detected=True,
        src_label="Live Phone AI (cam_01) + Fingers",
        fps=30.0,
    )
    srv.update_live_pose(payload)

    r1 = client.get("/api/v1/bridge/live")
    assert r1.status_code == 200
    data = r1.json()
    assert data["active"] is True
    assert data["seq"] >= 1
    assert len(data["skeleton"]["bone_names"]) == 25
    assert len(data["local_rotations_xyzw"]) == 25
    assert data["hands"] is not None
    assert len(data["hands"]["left_local_rotations_xyzw"]) == 16
    assert len(data["hands"]["right_local_rotations_xyzw"]) == 16
    assert data["hands"]["left_detected"] is True
    assert data["hands"]["right_detected"] is True

    # Stop live preview -> endpoint cleanly reports inactive
    srv.clear_live_pose()
    r2 = client.get("/api/v1/bridge/live")
    assert r2.status_code == 200
    assert r2.json()["active"] is False


def test_usb_camera_stream_reconfigure_and_mp4_recording(tmp_path: Path):
    """Verify USBCameraManager streams live JPEG previews, reconfigures live, and records readable MP4 + timestamps."""
    import json
    import cv2

    from desktop.camera_manager.usb_cameras import USBCameraManager

    class _FakeUSBVideoCapture:
        def __init__(self) -> None:
            self.w = 640
            self.h = 480
            self.fps = 30.0
            self.opened = True
            self.released = False
            self._n = 0

        def isOpened(self) -> bool:
            return self.opened

        def set(self, prop: int, val: float) -> bool:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                self.w = int(val)
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                self.h = int(val)
            elif prop == cv2.CAP_PROP_FPS:
                self.fps = float(val)
            return True

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.w)
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.h)
            if prop == cv2.CAP_PROP_FPS:
                return float(self.fps)
            return 0.0

        def read(self):
            time.sleep(0.015)
            self._n += 1
            frame = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
            cv2.putText(frame, f"USB F{self._n}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 200), 2)
            return True, frame

        def release(self) -> None:
            self.opened = False
            self.released = True

    caps_created = []

    def _factory(_idx: int):
        c = _FakeUSBVideoCapture()
        caps_created.append(c)
        return c, "FakeUSB"

    cm = CameraManager()
    usb_mgr = USBCameraManager(cm, capture_factory=_factory)
    try:
        rec = usb_mgr.connect_camera(index=0, name="Logitech Test USB", width=640, height=480, fps=30.0)
        assert rec.adapter_type == "usb_camera"
        assert rec.ip_address == "USB #0"
        assert rec.camera_uid == "cam_01"

        # Wait briefly for preview JPEG frame & local clock sync
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and rec.latest_preview_jpeg is None:
            time.sleep(0.02)
        assert rec.latest_preview_jpeg is not None and rec.latest_preview_jpeg.startswith(b"\xff\xd8")
        assert rec.clock_model is not None and rec.clock_model.residual_rmse_ms <= 1.0

        # Live reconfiguration
        assert usb_mgr.reconfigure_camera(rec.device_id, 800, 600, 30.0) is True
        time.sleep(0.08)

        # Synchronized MP4 recording directly to take folder
        started = usb_mgr.start_recording_all(tmp_path / "recordings", "take_usb_01", start_at_monotonic_ns=0)
        assert rec.device_id in started
        time.sleep(0.25)
        saved = usb_mgr.stop_recording_all()
        assert rec.camera_uid in saved
        mp4_path = saved[rec.camera_uid]
        assert mp4_path.exists() and mp4_path.stat().st_size > 1000

        # Verify MP4 is readable by OpenCV and timestamps JSON exists
        cap_check = cv2.VideoCapture(str(mp4_path))
        ok_read, img = cap_check.read()
        cap_check.release()
        assert ok_read and img is not None

        ts_path = mp4_path.with_name(f"{mp4_path.stem}_timestamps.json")
        assert ts_path.exists()
        ts_data = json.loads(ts_path.read_text(encoding="utf-8"))
        assert ts_data["frame_count"] >= 3
    finally:
        usb_mgr.shutdown()

    assert caps_created and caps_created[0].released is True
