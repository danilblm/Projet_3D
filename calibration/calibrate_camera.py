"""
calibrate_camera.py — Interactive chessboard-based camera calibration.

Opens the webcam and asks the user to present a printed chessboard pattern
from different angles.  After enough frames are captured the intrinsic matrix
and distortion coefficients are computed and saved to an .npz file which the
rest of the pipeline loads automatically.

Theory
------
The pinhole camera model relates 3-D world points  **X**  to 2-D image points
**x**  via

    s · x  =  K · [ R | t ] · X

where *K* is the 3×3 intrinsic (camera) matrix and *(R, t)* the extrinsic
pose.  `cv2.calibrateCamera` estimates *K* and the lens-distortion vector *D*
from multiple views of a planar calibration target whose geometry is known
(here a chessboard).

Usage (standalone):
    python -m calibration.calibrate_camera
"""

import os
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def _build_object_points(board_size: tuple[int, int],
                         square_mm: float) -> np.ndarray:
    """
    Build the array of 3-D coordinates for the inner corners of the
    chessboard.  Z = 0 because the pattern is planar.

    Returns shape (N, 3) float32 where N = board_size[0] * board_size[1].
    """
    cols, rows = board_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_mm  # scale to real-world mm
    return objp


def collect_calibration_frames(
    cap: cv2.VideoCapture,
    board_size: tuple[int, int] = config.CHESSBOARD_SIZE,
    square_mm: float = config.CHESSBOARD_SQUARE_MM,
    n_frames: int = config.CALIBRATION_FRAMES_NEEDED,
    delay_sec: float = config.CALIBRATION_DELAY_SEC,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """
    Interactively collect calibration frames from the webcam.

    Shows a live preview. When the chessboard is detected and enough time
    has passed since the last capture, the frame is recorded.

    Returns
    -------
    obj_points : list[np.ndarray]
        3-D reference points for every captured frame.
    img_points : list[np.ndarray]
        Corresponding 2-D corner detections (sub-pixel refined).
    image_size : (width, height)
    """
    objp = _build_object_points(board_size, square_mm)
    criteria_subpix = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                       30, 0.001)

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    last_capture_time = 0.0
    image_size = None

    print(f"\n[calibration] Present a {board_size[0]}×{board_size[1]} "
          f"chessboard to the camera.")
    print(f"[calibration] Need {n_frames} good frames "
          f"(min {delay_sec}s apart).\n")

    while len(obj_points) < n_frames:
        ret, frame = cap.read()
        if not ret:
            continue

        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_FAST_CHECK
            | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        display = frame.copy()

        if found:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria_subpix
            )
            cv2.drawChessboardCorners(display, board_size,
                                      corners_refined, found)

            now = time.time()
            if now - last_capture_time >= delay_sec:
                obj_points.append(objp)
                img_points.append(corners_refined)
                last_capture_time = now
                print(f"  ✓ Captured frame {len(obj_points)}/{n_frames}")

        # HUD
        text = (f"Frames: {len(obj_points)}/{n_frames}  |  "
                f"Press Q to abort")
        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Camera Calibration", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[calibration] Aborted by user.")
            break

    cv2.destroyWindow("Camera Calibration")
    return obj_points, img_points, image_size


def calibrate_and_save(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
    output_path: str = config.CALIBRATION_FILE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run ``cv2.calibrateCamera`` and persist the results.

    Returns
    -------
    camera_matrix : np.ndarray   (3×3)
    dist_coeffs   : np.ndarray   (1×5 or similar)
    """
    print("\n[calibration] Running calibrateCamera …")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    print(f"[calibration] RMS re-projection error: {rms:.4f}")
    print(f"[calibration] Camera matrix:\n{camera_matrix}")
    print(f"[calibration] Distortion coefficients:\n{dist_coeffs.ravel()}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        rms=np.array([rms]),
    )
    print(f"[calibration] Saved → {output_path}")
    return camera_matrix, dist_coeffs


def load_calibration(
    path: str = config.CALIBRATION_FILE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a previously saved calibration.

    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No calibration data at {path}. "
            "Run  python -m calibration.calibrate_camera  first."
        )
    data = np.load(path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    print(f"[calibration] Loaded calibration from {path}")
    return camera_matrix, dist_coeffs


def default_camera_matrix(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a reasonable *approximate* intrinsic matrix when no calibration
    file is available.  Assumes zero distortion and a focal length roughly
    equal to the image width (≈ 60° horizontal FoV).
    """
    fx = fy = float(width)
    cx, cy = width / 2.0, height / 2.0
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    print("[calibration] Using approximate (default) camera matrix.")
    return camera_matrix, dist_coeffs


# ──────────────────────────────────────────────
# Standalone entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        sys.exit("[calibration] ERROR: Cannot open camera.")

    try:
        obj_pts, img_pts, img_sz = collect_calibration_frames(cap)
        if len(obj_pts) >= 5:
            calibrate_and_save(obj_pts, img_pts, img_sz)
        else:
            print("[calibration] Not enough frames collected — "
                  "calibration skipped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
