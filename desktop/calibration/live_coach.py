"""Real-Time Live Calibration Coaching, Directional Guidance, Board Mismatch Detection, HUD Drawing & Audio Prompts.

Analyzes live ChArUco corner detections across 2..12 streaming phones and gives immediate,
room-readable coaching instructions:
- Which printed board to show (with automatic A4 7x5 vs A3 9x6 mismatch detection)
- Distance feedback (COME CLOSER / STEP BACK / CLIPPED AT EDGE)
- Position feedback matched to the active pose (MOVE LEFT / MOVE RIGHT / RAISE HIGHER / PLACE LOWER ON FLOOR)
- Visibility feedback naming the exact camera that cannot see the board yet
- Steadiness check & hands-free auto-capture lock-on progress
- Audible chimes + optional non-blocking Windows TTS voice coach prompts
"""

from __future__ import annotations

import base64
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from desktop.calibration.charuco_board import (
    ChArUcoBoardConfig,
    ChArUcoDetection,
    detect_charuco_corners,
)


@dataclass(frozen=True)
class PoseTargetZone:
    """Normalized image target zone (0..1) where the ChArUco board should sit for a given pose."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    hint_short: str


# Matches CALIBRATION_POSE_STEPS indices 0..5 in calibration_guide.py
# Notice generous target zones so two angled cameras (~75-90 deg apart) never give conflicting left/right hints!
POSE_TARGET_ZONES: List[PoseTargetZone] = [
    PoseTargetZone(0.16, 0.84, 0.16, 0.84, "CENTER UPRIGHT (chest height)"),
    PoseTargetZone(0.16, 0.84, 0.16, 0.84, "CENTER TILTED (~25° angle)"),
    PoseTargetZone(0.05, 0.68, 0.15, 0.85, "LEFT SIDE (angled inward)"),
    PoseTargetZone(0.32, 0.95, 0.15, 0.85, "RIGHT SIDE (angled inward)"),
    PoseTargetZone(0.14, 0.86, 0.42, 0.94, "LOW NEAR KNEES (tilted UP)"),
    PoseTargetZone(0.14, 0.86, 0.05, 0.58, "HIGH OVERHEAD (tilted DOWN)"),
]


@dataclass
class CameraCoachFeedback:
    camera_uid: str
    num_corners: int
    status: str  # "missing", "too_far", "too_close", "clipped", "move_left", "move_right", "move_up", "move_down", "moving", "ready"
    short_hint: str
    board_center_norm: Optional[Tuple[float, float]] = None
    board_size_frac: float = 0.0
    bbox_px: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2


@dataclass
class MultiCamCoachDecision:
    ready_count: int
    visible_count: int
    total_cams: int
    all_ready_to_capture: bool
    severity: str  # "ok", "guide", "warn"
    headline: str
    detail: str
    voice_key: str = ""
    voice_prompt: str = ""


def _friendly_cam_name(uid: str) -> str:
    """Convert camera UID like 'cam_02' into natural spoken 'Camera 2'."""
    m = re.match(r"^[a-zA-Z]+[_-]?0*(\d+)$", uid.strip())
    if m:
        return f"Camera {m.group(1)}"
    return uid.replace("_", " ")


def pose_spoken_instruction(pose_index: int, board_label: str = "ChArUco") -> str:
    """Return a clear, natural spoken instruction for each of the 6 guided calibration poses."""
    idx = max(0, min(5, int(pose_index)))
    prompts = [
        f"Pose 1, Center Upright. Stand in the middle and hold your {board_label} board straight upright at chest height facing between both cameras.",
        f"Pose 2, Tilted Angle. Stay in the center and tilt the top of your {board_label} board slightly forward toward the cameras.",
        f"Pose 3, Left Side. Move to the left side of the capture area and angle the {board_label} board slightly inward.",
        f"Pose 4, Right Side. Move to the right side of the capture area and angle the {board_label} board slightly inward.",
        f"Pose 5, Low Angle. Hold the {board_label} board low near your knees and tilt it slightly upward toward the cameras.",
        f"Pose 6, High Angle. Hold the {board_label} board high near your head and tilt it slightly downward toward the cameras.",
    ]
    return prompts[idx]


def analyze_camera_detection(
    camera_uid: str,
    detection: ChArUcoDetection,
    img_w: int,
    img_h: int,
    pose_index: int = 0,
    prev_center_norm: Optional[Tuple[float, float]] = None,
    min_corners: int = 6,
) -> CameraCoachFeedback:
    """Compute actionable distance, placement, and steadiness feedback for one camera feed."""
    n = int(detection.num_corners)
    if n < min_corners or img_w <= 0 or img_h <= 0:
        return CameraCoachFeedback(
            camera_uid=camera_uid,
            num_corners=n,
            status="missing",
            short_hint=f"Show board clearly ({n} corners)",
        )

    pts = np.asarray(detection.corners_2d, dtype=np.float64).reshape(-1, 2)
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    cx_n = (0.5 * (x1 + x2)) / float(img_w)
    cy_n = (0.5 * (y1 + y2)) / float(img_h)
    size_frac = max(bw / float(img_w), bh / float(img_h))
    bbox = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))

    # 1. Edge clipping check (corners jammed against image border)
    margin_x = 0.025 * img_w
    margin_y = 0.025 * img_h
    if x1 < margin_x or x2 > (img_w - margin_x) or y1 < margin_y or y2 > (img_h - margin_y):
        return CameraCoachFeedback(
            camera_uid=camera_uid,
            num_corners=n,
            status="clipped",
            short_hint="STEP BACK — board cut off at edge",
            board_center_norm=(cx_n, cy_n),
            board_size_frac=size_frac,
            bbox_px=bbox,
        )

    # 2. Distance check (too far / too small vs too close / fills frame)
    if size_frac < 0.095:
        return CameraCoachFeedback(
            camera_uid=camera_uid,
            num_corners=n,
            status="too_far",
            short_hint="COME CLOSER — board looks too small",
            board_center_norm=(cx_n, cy_n),
            board_size_frac=size_frac,
            bbox_px=bbox,
        )
    if size_frac > 0.78:
        return CameraCoachFeedback(
            camera_uid=camera_uid,
            num_corners=n,
            status="too_close",
            short_hint="STEP BACK — board is too close",
            board_center_norm=(cx_n, cy_n),
            board_size_frac=size_frac,
            bbox_px=bbox,
        )

    # 3. Pose-specific placement guidance (vertical & horizontal zone)
    zone = POSE_TARGET_ZONES[max(0, min(len(POSE_TARGET_ZONES) - 1, int(pose_index)))]
    if cy_n < zone.y_min:
        return CameraCoachFeedback(
            camera_uid=camera_uid,
            num_corners=n,
            status="move_down",
            short_hint="LOWER BOARD DOWN",
            board_center_norm=(cx_n, cy_n),
            board_size_frac=size_frac,
            bbox_px=bbox,
        )
    if cy_n > zone.y_max:
        return CameraCoachFeedback(
            camera_uid=camera_uid,
            num_corners=n,
            status="move_up",
            short_hint="RAISE BOARD HIGHER",
            board_center_norm=(cx_n, cy_n),
            board_size_frac=size_frac,
            bbox_px=bbox,
        )
    if cx_n < zone.x_min:
        return CameraCoachFeedback(
            camera_uid=camera_uid,
            num_corners=n,
            status="move_right",
            short_hint="MOVE RIGHT into target box",
            board_center_norm=(cx_n, cy_n),
            board_size_frac=size_frac,
            bbox_px=bbox,
        )
    if cx_n > zone.x_max:
        return CameraCoachFeedback(
            camera_uid=camera_uid,
            num_corners=n,
            status="move_left",
            short_hint="MOVE LEFT into target box",
            board_center_norm=(cx_n, cy_n),
            board_size_frac=size_frac,
            bbox_px=bbox,
        )

    # 4. Steadiness check (generous 6.5% tolerance absorbs single-corner flicker while holding board steady)
    if prev_center_norm is not None:
        move_dist = float(np.hypot(cx_n - prev_center_norm[0], cy_n - prev_center_norm[1]))
        if move_dist > 0.065:
            return CameraCoachFeedback(
                camera_uid=camera_uid,
                num_corners=n,
                status="moving",
                short_hint="HOLD STEADY — almost locked...",
                board_center_norm=(cx_n, cy_n),
                board_size_frac=size_frac,
                bbox_px=bbox,
            )

    return CameraCoachFeedback(
        camera_uid=camera_uid,
        num_corners=n,
        status="ready",
        short_hint=f"LOCKED ({n} corners) — HOLD STILL!",
        board_center_norm=(cx_n, cy_n),
        board_size_frac=size_frac,
        bbox_px=bbox,
    )


def evaluate_multi_cam_coaching(
    feedbacks: List[CameraCoachFeedback],
    pose_index: int = 0,
    min_required_cams: int = 2,
) -> MultiCamCoachDecision:
    """Aggregate per-camera feedback into one clear room-readable coaching command + spoken prompt."""
    total = len(feedbacks)
    if total < min_required_cams:
        return MultiCamCoachDecision(
            ready_count=0,
            visible_count=0,
            total_cams=total,
            all_ready_to_capture=False,
            severity="warn",
            headline="Connect at least 2 Android phones with live video streams",
            detail="Open Nuranova Mocap Camera on 2+ phones and make sure both are streaming.",
            voice_key="need_cams",
            voice_prompt="Waiting for at least 2 cameras to stream live video.",
        )

    visible = [f for f in feedbacks if f.status != "missing"]
    ready = [f for f in feedbacks if f.status == "ready"]
    missing = [f for f in feedbacks if f.status == "missing"]
    required = min(total, max(min_required_cams, total if total <= 2 else 2))

    # All required cameras locked and steady!
    if len(ready) >= required:
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=True,
            severity="ok",
            headline="✓ PERFECT! HOLD STILL — CAPTURING POSE...",
            detail=f"{len(ready)}/{total} cameras locked onto the ChArUco board.",
            voice_key="ready",
            voice_prompt="Good! Hold completely still.",
        )

    # Not enough cameras see the board yet
    if len(visible) == 0:
        zone = POSE_TARGET_ZONES[max(0, min(len(POSE_TARGET_ZONES) - 1, pose_index))]
        v_pose = "I cannot see the board yet. Hold the printed board upright facing both cameras, and step closer if needed."
        return MultiCamCoachDecision(
            ready_count=0,
            visible_count=0,
            total_cams=total,
            all_ready_to_capture=False,
            severity="warn",
            headline=f"SHOW CHArUco BOARD TO CAMERAS — {zone.hint_short}",
            detail="Make sure the printed board faces your phones with good lighting and matches the A4/A3 preset above.",
            voice_key=f"missing_all_{pose_index}",
            voice_prompt=v_pose,
        )

    if len(visible) < required:
        miss_names = ", ".join(f.camera_uid for f in missing[:3])
        miss_spoken = " and ".join(_friendly_cam_name(f.camera_uid) for f in missing[:2])
        vis_spoken = _friendly_cam_name(visible[0].camera_uid)
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=False,
            severity="guide",
            headline=f"↻ ANGLE BOARD TOWARD {miss_names} (only {len(visible)}/{total} cams see it)",
            detail=f"{miss_names} cannot see enough corners yet — turn or step slightly so all cameras see the board together.",
            voice_key=f"missing_{miss_names}",
            voice_prompt=f"{vis_spoken} sees the board. Turn or angle the board slightly toward {miss_spoken}.",
        )

    # Multiple cameras see the board — prioritize the most important correction!
    statuses = [f.status for f in visible]
    if "clipped" in statuses or "too_close" in statuses:
        bad = [f.camera_uid for f in visible if f.status in ("clipped", "too_close")]
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=False,
            severity="guide",
            headline="⬇ STEP BACK A LITTLE — Board is too close / near frame edge",
            detail=f"In {', '.join(bad)}, part of the board is too close to the edge. Take half a step back.",
            voice_key="step_back",
            voice_prompt="Step back a little. The board is too close to the edge.",
        )
    if "too_far" in statuses:
        bad = [f.camera_uid for f in visible if f.status == "too_far"]
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=False,
            severity="guide",
            headline="⬆ COME FORWARD / CLOSER — Board looks too small",
            detail=f"In {', '.join(bad)}, the board is far away. Step closer to the cameras so corners detect cleanly.",
            voice_key="come_forward",
            voice_prompt="Come forward, closer to the cameras.",
        )
    if "move_down" in statuses:
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=False,
            severity="guide",
            headline="⬇ LOWER THE BOARD DOWN",
            detail="Move the ChArUco board down into the dashed target area shown on the live video.",
            voice_key=f"move_down_{pose_index}",
            voice_prompt="Lower the board down.",
        )
    if "move_up" in statuses:
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=False,
            severity="guide",
            headline="⬆ RAISE THE BOARD HIGHER",
            detail="Lift the ChArUco board up into the dashed target zone shown on the live video.",
            voice_key="move_up",
            voice_prompt="Raise the board higher.",
        )
    if "move_left" in statuses:
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=False,
            severity="guide",
            headline="⬅ MOVE BOARD TO THE LEFT",
            detail="Shift slightly left into the dashed guide box on screen.",
            voice_key="move_left",
            voice_prompt="Move the board slightly to the left.",
        )
    if "move_right" in statuses:
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=False,
            severity="guide",
            headline="➡ MOVE BOARD TO THE RIGHT",
            detail="Shift slightly right into the dashed guide box on screen.",
            voice_key="move_right",
            voice_prompt="Move the board slightly to the right.",
        )
    if "moving" in statuses:
        return MultiCamCoachDecision(
            ready_count=len(ready),
            visible_count=len(visible),
            total_cams=total,
            all_ready_to_capture=False,
            severity="guide",
            headline="✋ HOLD STEADY — Locking in pose...",
            detail="Keep the board completely still for 1 second to auto-capture cleanly without blur.",
            voice_key="hold_steady",
            voice_prompt="Almost there. Hold the board steady.",
        )

    return MultiCamCoachDecision(
        ready_count=len(ready),
        visible_count=len(visible),
        total_cams=total,
        all_ready_to_capture=False,
        severity="guide",
        headline="Adjust board slightly so all cameras lock in",
        detail="Hold the ChArUco board steady inside the dashed guide zone.",
        voice_key="adjust",
        voice_prompt="Adjust the board slightly so both cameras lock in.",
    )


def detect_wrong_board_preset(
    frame_bgr: np.ndarray,
    active_cfg: ChArUcoBoardConfig,
    current_corners: int,
) -> Optional[str]:
    """Check if the user is holding the other standard printed board (A3 9x6 vs A4 7x5)."""
    if current_corners >= 8:
        return None
    if active_cfg.squares_x == 7 and active_cfg.squares_y == 5:
        alt_cfg = ChArUcoBoardConfig(9, 6, 40.0, 30.0, "DICT_5X5_250")
        alt_label = "A3"
    elif active_cfg.squares_x == 9 and active_cfg.squares_y == 6:
        alt_cfg = ChArUcoBoardConfig(7, 5, 35.0, 26.0, "DICT_5X5_250")
        alt_label = "A4"
    else:
        return None
    alt_det = detect_charuco_corners(frame_bgr, alt_cfg)
    if alt_det.num_corners >= max(8, current_corners + 4):
        return alt_label
    return None


def draw_live_calibration_hud(
    frame_bgr: np.ndarray,
    detection: ChArUcoDetection,
    feedback: CameraCoachFeedback,
    pose_index: int = 0,
    hold_progress: float = 0.0,
    flash_text: str = "",
) -> np.ndarray:
    """Render target placement box, detected ChArUco outline/corners, directional hint & lock-on bar on BGR frame."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    zone = POSE_TARGET_ZONES[max(0, min(len(POSE_TARGET_ZONES) - 1, int(pose_index)))]

    # 1. Draw target placement guide rectangle (subtle dashed/corner brackets)
    zx1 = int(round(zone.x_min * w))
    zy1 = int(round(zone.y_min * h))
    zx2 = int(round(zone.x_max * w))
    zy2 = int(round(zone.y_max * h))

    zone_col = (94, 197, 34) if feedback.status == "ready" else (180, 166, 20)
    corner_len = max(14, min(zx2 - zx1, zy2 - zy1) // 6)
    thick = 2
    # 4 crisp studio viewfinder corner brackets
    for (cx, cy, dx, dy) in [
        (zx1, zy1, 1, 1),
        (zx2, zy1, -1, 1),
        (zx1, zy2, 1, -1),
        (zx2, zy2, -1, -1),
    ]:
        cv2.line(vis, (cx, cy), (cx + dx * corner_len, cy), zone_col, thick, cv2.LINE_AA)
        cv2.line(vis, (cx, cy), (cx, cy + dy * corner_len), zone_col, thick, cv2.LINE_AA)

    # 2. Draw detected ChArUco bounding box & green corners
    if feedback.bbox_px is not None:
        bx1, by1, bx2, by2 = feedback.bbox_px
        box_col = (94, 197, 34) if feedback.status == "ready" else (11, 158, 245)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), box_col, 2, cv2.LINE_AA)

    if detection.num_corners > 0:
        for pt in detection.corners_2d:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(vis, (px, py), 4, (94, 197, 34), -1, cv2.LINE_AA)
            cv2.circle(vis, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Top translucent HUD bar with live guidance text
    bar_h = max(26, int(h * 0.10))
    overlay = vis.copy()
    bar_bg = (22, 101, 52) if feedback.status == "ready" else ((30, 58, 138) if feedback.status != "missing" else (30, 41, 59))
    cv2.rectangle(overlay, (0, 0), (w, bar_h), bar_bg, -1)
    cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)

    font_scale = max(0.45, min(0.75, w / 960.0))
    cv2.putText(
        vis,
        f"{feedback.camera_uid}: {feedback.short_hint}",
        (10, int(bar_h * 0.70)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 4. Bottom lock-on progress bar while holding steady for auto-capture
    if hold_progress > 0.01:
        pb_h = max(8, int(h * 0.03))
        fill_w = int(round(w * min(1.0, max(0.0, hold_progress))))
        cv2.rectangle(vis, (0, h - pb_h), (w, h), (20, 30, 42), -1)
        cv2.rectangle(vis, (0, h - pb_h), (fill_w, h), (94, 197, 34), -1)

    # 5. Brief green capture confirmation flash overlay
    if flash_text:
        fl = vis.copy()
        cv2.rectangle(fl, (0, 0), (w, h), (34, 197, 94), -1)
        cv2.addWeighted(fl, 0.30, vis, 0.70, 0, vis)
        cv2.putText(
            vis,
            flash_text,
            (max(12, w // 2 - 120), h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.65, w / 650.0),
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return vis


def render_board_thumbnail_bgr(config: ChArUcoBoardConfig, width_px: int = 160, height_px: int = 112) -> np.ndarray:
    """Render a crisp miniature BGR preview of the exact ChArUco board the user must show to the cameras."""
    board, _ = config.create_opencv_board()
    gray = board.generateImage((width_px, height_px), marginSize=6, borderBits=1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def play_coach_sound(kind: str = "capture", step_index: int = 0) -> None:
    """Play a crisp non-blocking audio chime (capture / complete / tick / lock_tick) heard across the room."""
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        return

    def _beep() -> None:
        if sys.platform.startswith("win"):
            try:
                import winsound

                if kind == "capture":
                    winsound.Beep(880, 110)
                    winsound.Beep(1320, 180)
                elif kind == "complete":
                    winsound.Beep(784, 110)
                    winsound.Beep(1046, 110)
                    winsound.Beep(1568, 240)
                elif kind == "lock_tick":
                    freq = min(1400, 640 + int(step_index) * 120)
                    winsound.Beep(freq, 55)
                elif kind == "tick":
                    winsound.Beep(660, 60)
            except Exception:
                pass

    threading.Thread(target=_beep, daemon=True).start()


_SAPI_LOOP_PS = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 1
$s.Volume = 100
while (($line = [Console]::In.ReadLine()) -ne $null) {
    if ($line -eq '__QUIT__') { break }
    $parts = $line -split '\\|', 2
    if ($parts.Length -eq 2) {
        $mode = $parts[0]
        $msg = $parts[1]
        if ($mode -eq 'PRIORITY') {
            $s.SpeakAsyncCancelAll()
            $s.SpeakAsync($msg) | Out-Null
        } else {
            if ($s.State -ne 'Speaking') {
                $s.SpeakAsync($msg) | Out-Null
            }
        }
    }
}
$s.SpeakAsyncCancelAll()
$s.Dispose()
"""


class LiveVoiceCoach:
    """Persistent low-latency Windows SAPI voice coach that speaks continuously until each pose is done right.

    - Never overlaps two voices (`NORMAL` prompts are skipped while currently speaking).
    - `PRIORITY` announcements (pose captured, next pose, calibration solved) interrupt immediately.
    - Smart pacing: speaks quickly (~0.95s) when the required correction changes, and repeats
      the current correction every ~2.6s until the user gets into position.
    - 100% no-op in offscreen test mode (`QT_QPA_PLATFORM=offscreen`).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._last_spoken_key: str = ""
        self._last_spoken_t: float = 0.0
        self._last_spoken_text: str = ""

    def _ensure_process(self) -> Optional[subprocess.Popen]:
        if os.environ.get("QT_QPA_PLATFORM") == "offscreen" or not sys.platform.startswith("win"):
            return None
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        try:
            enc = base64.b64encode(_SAPI_LOOP_PS.encode("utf-16-le")).decode("ascii")
            self._proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-EncodedCommand", enc],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return self._proc
        except Exception:
            self._proc = None
            return None

    def _send(self, mode: str, text: str) -> None:
        clean = re.sub(r"[\r\n|]+", " ", text).strip()[:220]
        if not clean:
            return
        self._last_spoken_text = clean
        with self._lock:
            proc = self._ensure_process()
            if proc is None or proc.stdin is None:
                return
            try:
                proc.stdin.write(f"{mode}|{clean}\n")
                proc.stdin.flush()
            except Exception:
                self._proc = None

    def announce(self, text: str, priority: bool = True) -> None:
        """Speak an urgent event immediately (pose captured, start, board switch, calibration result)."""
        now = time.monotonic()
        self._last_spoken_key = "__priority__"
        self._last_spoken_t = now
        self._send("PRIORITY" if priority else "NORMAL", text)

    def coach(
        self,
        voice_key: str,
        text: str,
        now: Optional[float] = None,
        repeat_interval_s: float = 2.6,
        change_interval_s: float = 0.95,
    ) -> bool:
        """Continuously coach the user aloud with smart pacing until they hit the target pose.

        Returns True if a spoken prompt was dispatched on this tick.
        """
        if not text:
            return False
        t = time.monotonic() if now is None else float(now)
        dt = t - self._last_spoken_t
        key_changed = voice_key != self._last_spoken_key

        if key_changed:
            if dt < change_interval_s:
                return False
        else:
            if dt < repeat_interval_s:
                return False

        self._last_spoken_key = voice_key
        self._last_spoken_t = t
        self._send("NORMAL", text)
        return True

    def reset_pacing(self) -> None:
        self._last_spoken_key = ""
        self._last_spoken_t = 0.0

    def stop(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write("__QUIT__\n")
                    proc.stdin.flush()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass


def speak_coach_prompt(text: str) -> None:
    """Backward-compatible helper for one-shot spoken prompt."""
    _DEFAULT_VOICE_COACH.announce(text, priority=True)


_DEFAULT_VOICE_COACH = LiveVoiceCoach()
