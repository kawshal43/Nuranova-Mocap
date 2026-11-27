"""Printable ChArUco calibration board generation (exact physical mm PDF & PNG) and detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ARUCO_DICTIONARIES: Dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass
class ChArUcoBoardConfig:
    squares_x: int = 7
    squares_y: int = 5
    square_length_mm: float = 35.0
    marker_length_mm: float = 26.0
    dictionary_name: str = "DICT_5X5_250"
    measured_square_length_mm: Optional[float] = None  # User's ruler-verified physical measurement

    @property
    def effective_square_length_m(self) -> float:
        val = self.measured_square_length_mm if self.measured_square_length_mm is not None else self.square_length_mm
        return float(val) * 1e-3

    @property
    def effective_marker_length_m(self) -> float:
        scale = (
            (self.measured_square_length_mm / self.square_length_mm)
            if self.measured_square_length_mm is not None
            else 1.0
        )
        return float(self.marker_length_mm * scale) * 1e-3

    def create_opencv_board(self) -> Tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
        if self.dictionary_name not in ARUCO_DICTIONARIES:
            raise ValueError(f"Unsupported ArUco dictionary: {self.dictionary_name}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[self.dictionary_name])
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.effective_square_length_m,
            self.effective_marker_length_m,
            aruco_dict,
        )
        return board, aruco_dict

    def get_chessboard_corners_3d(self) -> np.ndarray:
        """Return (N_corners, 3) 3D coordinates of internal ChArUco corners in board frame (meters)."""
        board, _ = self.create_opencv_board()
        return np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)


def render_charuco_png(
    config: ChArUcoBoardConfig,
    output_path: str | Path,
    dpi: int = 300,
    margin_px: int = 60,
) -> Path:
    """Render high-resolution ChArUco board PNG at exact physical DPI."""
    board, _ = config.create_opencv_board()
    width_mm = config.squares_x * config.square_length_mm
    height_mm = config.squares_y * config.square_length_mm
    px_per_mm = dpi / 25.4
    img_w = max(200, int(round(width_mm * px_per_mm)) + 2 * margin_px)
    img_h = max(200, int(round(height_mm * px_per_mm)) + 2 * margin_px)
    img = board.generateImage((img_w, img_h), marginSize=margin_px, borderBits=1)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return out


def generate_printable_charuco_pdf(
    config: ChArUcoBoardConfig,
    output_pdf: str | Path,
    paper_size: str = "A4",
    dpi: int = 300,
) -> Path:
    """Generate a physical-scale A4 or A3 printable PDF with a 100mm verification ruler."""
    page = landscape(A3) if paper_size.upper() == "A3" else landscape(A4)
    page_w, page_h = page

    board_w_mm = config.squares_x * config.square_length_mm
    board_h_mm = config.squares_y * config.square_length_mm
    if board_w_mm * mm > page_w - 20 * mm or board_h_mm * mm > page_h - 35 * mm:
        raise ValueError(
            f"Board ({board_w_mm:.1f}x{board_h_mm:.1f} mm) exceeds printable area of {paper_size} landscape."
        )

    board, _ = config.create_opencv_board()
    px_per_mm = dpi / 25.4
    raw_w = int(round(board_w_mm * px_per_mm))
    raw_h = int(round(board_h_mm * px_per_mm))
    img = board.generateImage((raw_w, raw_h), marginSize=0, borderBits=1)

    from PIL import Image

    pil_img = Image.fromarray(img)
    out = Path(output_pdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out), pagesize=page)
    c.setTitle(f"Nuranova Mocap ChArUco Calibration Board ({paper_size})")

    x0 = (page_w - board_w_mm * mm) / 2.0
    y0 = (page_h - board_h_mm * mm) / 2.0 + 6 * mm
    c.drawImage(ImageReader(pil_img), x0, y0, width=board_w_mm * mm, height=board_h_mm * mm)

    # Header & verification instructions
    c.setFont("Helvetica-Bold", 11)
    c.drawString(
        15 * mm,
        page_h - 12 * mm,
        f"Nuranova Mocap Studio — ChArUco Calibration Board ({paper_size}, {config.squares_x}x{config.squares_y}, "
        f"{config.dictionary_name})",
    )
    c.setFont("Helvetica", 8.5)
    c.drawString(
        15 * mm,
        page_h - 17 * mm,
        f"Nominal Square: {config.square_length_mm:.2f} mm | Marker: {config.marker_length_mm:.2f} mm | "
        "PRINT AT 100% ACTUAL SIZE (NO FIT-TO-PAGE). Measure the 100 mm bar below and enter actual square size in Calibration Wizard.",
    )

    # 100 mm physical verification bar at bottom
    bar_y = 10 * mm
    bar_x = 15 * mm
    c.setLineWidth(1.2)
    c.line(bar_x, bar_y, bar_x + 100 * mm, bar_y)
    for tick_mm in range(0, 101, 10):
        tx = bar_x + tick_mm * mm
        h = 4 * mm if tick_mm % 50 == 0 else 2.2 * mm
        c.line(tx, bar_y, tx, bar_y + h)
    c.setFont("Helvetica", 8)
    c.drawString(bar_x + 103 * mm, bar_y, "100.0 mm Verification Scale (measure with caliper/ruler)")
    c.showPage()
    c.save()
    return out


@dataclass
class ChArUcoDetection:
    corner_ids: np.ndarray  # (N,) int32
    corners_2d: np.ndarray  # (N, 2) float64
    marker_ids: np.ndarray  # (M,) int32
    marker_corners: list

    @property
    def num_corners(self) -> int:
        return int(self.corner_ids.shape[0])


def _run_charuco_detect_pass(
    gray: np.ndarray,
    board: cv2.aruco.CharucoBoard,
    camera_matrix: Optional[np.ndarray],
    dist_coeffs: Optional[np.ndarray],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    loose_poly: bool = False,
) -> ChArUcoDetection:
    """Run OpenCV CharucoDetector with tuned parameters for full-res, preview, and steeply angled frames."""
    charuco_params = cv2.aruco.CharucoParameters()
    if camera_matrix is not None and scale_x == 1.0 and scale_y == 1.0:
        charuco_params.cameraMatrix = np.asarray(camera_matrix, dtype=np.float64)
    if dist_coeffs is not None and scale_x == 1.0 and scale_y == 1.0:
        charuco_params.distCoeffs = np.asarray(dist_coeffs, dtype=np.float64)

    det_params = cv2.aruco.DetectorParameters()
    det_params.minMarkerPerimeterRate = 0.010 if loose_poly else 0.012
    det_params.adaptiveThreshWinSizeMin = 3
    det_params.adaptiveThreshWinSizeMax = 41 if loose_poly else 35
    det_params.adaptiveThreshWinSizeStep = 4
    det_params.polygonalApproxAccuracyRate = 0.075 if loose_poly else 0.05
    det_params.minDistanceToBorder = 1
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    detector = cv2.aruco.CharucoDetector(board, charuco_params, det_params)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    m_ids = np.zeros((0,), dtype=np.int32) if marker_ids is None else marker_ids.reshape(-1).astype(np.int32)
    m_corners = [] if marker_corners is None else list(marker_corners)
    if (scale_x != 1.0 or scale_y != 1.0) and m_corners:
        scale_vec = np.array([scale_x, scale_y], dtype=np.float32)
        m_corners = [np.asarray(mc, dtype=np.float32) / scale_vec for mc in m_corners]

    if charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0:
        return ChArUcoDetection(
            corner_ids=np.zeros((0,), dtype=np.int32),
            corners_2d=np.zeros((0, 2), dtype=np.float64),
            marker_ids=m_ids,
            marker_corners=m_corners,
        )

    c2d = charuco_corners.reshape(-1, 2).astype(np.float64)
    if scale_x != 1.0 or scale_y != 1.0:
        c2d[:, 0] /= float(scale_x)
        c2d[:, 1] /= float(scale_y)
    return ChArUcoDetection(
        corner_ids=charuco_ids.reshape(-1).astype(np.int32),
        corners_2d=c2d,
        marker_ids=m_ids,
        marker_corners=m_corners,
    )


def detect_charuco_corners(
    image: np.ndarray,
    config: ChArUcoBoardConfig,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> ChArUcoDetection:
    """Detect ChArUco corners using tuned OpenCV CharucoDetector + CLAHE/upscale + oblique floor stretch fallback."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    board, _ = config.create_opencv_board()

    # Pass 1: Fast native-resolution pass
    best = _run_charuco_detect_pass(gray, board, camera_matrix, dist_coeffs, scale_x=1.0, scale_y=1.0)
    if best.num_corners >= 8:
        return best

    # Pass 2: CLAHE local contrast enhancement (+ 2x upscale on preview-sized images)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    h, w = gray.shape[:2]
    if max(h, w) <= 1280:
        up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        cand = _run_charuco_detect_pass(clahe.apply(up), board, None, None, scale_x=2.0, scale_y=2.0)
    else:
        cand = _run_charuco_detect_pass(clahe.apply(gray), board, camera_matrix, dist_coeffs, scale_x=1.0, scale_y=1.0)
    if cand.num_corners > best.num_corners:
        best = cand
    if best.num_corners >= 6:
        return best

    # Pass 3: Anisotropic vertical stretch fallback (un-squashes steeply tilted or flat-on-floor boards!)
    if max(h, w) <= 1280:
        stretched = cv2.resize(gray, None, fx=2.0, fy=3.5, interpolation=cv2.INTER_CUBIC)
        cand3 = _run_charuco_detect_pass(
            clahe.apply(stretched),
            board,
            None,
            None,
            scale_x=2.0,
            scale_y=3.5,
            loose_poly=True,
        )
        if cand3.num_corners > best.num_corners:
            best = cand3

    return best
