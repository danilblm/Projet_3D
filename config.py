"""
config.py — Central configuration for the Augmented Reality pipeline.

All tunable parameters are gathered here so that every module
imports a single source of truth.
"""

import os
import cv2

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CALIBRATION_DIR = os.path.join(PROJECT_ROOT, "calibration")
MARKER_DIR = os.path.join(PROJECT_ROOT, "marker")

# Default 3D model path (user places their Sketchfab download here)
MODEL_PATH = os.path.join(MODEL_DIR, "model.obj")

# Camera calibration data
CALIBRATION_FILE = os.path.join(CALIBRATION_DIR, "camera_data.npz")

# Generated ArUco marker image
MARKER_IMAGE_PATH = os.path.join(MARKER_DIR, "aruco_marker.png")

# ──────────────────────────────────────────────
# ArUco marker settings
# ──────────────────────────────────────────────
ARUCO_DICT_ID = cv2.aruco.DICT_5X5_100
MARKER_ID = 0          # Which marker to generate / detect
MARKER_SIZE_PX = 700   # Pixel size of the generated marker image
MARKER_LENGTH_M = 0.05 # Physical side length in metres (for pose estimation)

# ──────────────────────────────────────────────
# Camera / capture
# ──────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# ──────────────────────────────────────────────
# Chessboard calibration
# ──────────────────────────────────────────────
CHESSBOARD_SIZE = (9, 6)       # inner corners (cols, rows)
CHESSBOARD_SQUARE_MM = 25.0    # side length of one square in mm
CALIBRATION_FRAMES_NEEDED = 20 # how many good frames to collect
CALIBRATION_DELAY_SEC = 1.5    # min seconds between captured frames

# ──────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────
MODEL_SCALE = 1.0              # uniform scale applied to the 3D model
MODEL_ROTATION_DEG = (0, 0, 0) # (rx, ry, rz) extra rotation — auto-orient handles alignment
RENDER_WIDTH = CAMERA_WIDTH
RENDER_HEIGHT = CAMERA_HEIGHT
NEAR_PLANE = 0.01
FAR_PLANE = 100.0
MAX_FACES = 200_000            # auto-decimate meshes above this; 0 = off

# ──────────────────────────────────────────────
# Pose smoothing
# ──────────────────────────────────────────────
POSE_SMOOTH_ALPHA = 0.20       # 0.15 = very smooth, 0.7 = responsive
TRACKING_PERSIST_FRAMES = 30   # keep last pose visible for N frames after losing marker

# ──────────────────────────────────────────────
# Auto-rotation (turntable)
# ──────────────────────────────────────────────
TURNTABLE_SPEED = 45.0         # degrees per second

# ──────────────────────────────────────────────
# Screenshot
# ──────────────────────────────────────────────
SCREENSHOT_DIR = os.path.join(PROJECT_ROOT, "screenshots")

# ──────────────────────────────────────────────
# Keyboard interaction (runtime controls)
# ──────────────────────────────────────────────
SCALE_STEP = 0.1               # +/- per key press
ROTATION_STEP = 10.0           # degrees per key press

# ──────────────────────────────────────────────
# Video recording
# ──────────────────────────────────────────────
RECORD_PATH = os.path.join(PROJECT_ROOT, "output_ar.mp4")
RECORD_FPS = 25.0

# ──────────────────────────────────────────────
# Display
# ──────────────────────────────────────────────
WINDOW_NAME = "AR Pipeline — Press H for help"
