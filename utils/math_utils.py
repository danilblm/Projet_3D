"""
math_utils.py — Linear-algebra helpers for the AR pipeline.

Functions here convert between the different rotation / transformation
representations used by OpenCV (Rodrigues vectors, 3×3 rotation matrices)
and OpenGL (4×4 model-view and projection matrices).

Coordinate-system conventions
-----------------------------
* **OpenCV camera frame** :  X → right, Y → down, Z → forward (into scene).
* **OpenGL camera frame** :  X → right, Y → up,   Z → backward (out of screen).

The conversion is a 180° rotation around the X-axis, usually expressed as
``diag(1, -1, -1)`` applied to the rotation and translation.
"""

import math
import cv2
import numpy as np


# ──────────────────────────────────────────────
# Rotation helpers
# ──────────────────────────────────────────────

def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues vector (3,1) to a 3×3 rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R


def build_model_view_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Build a 4×4 OpenGL-style model-view matrix from OpenCV pose output.

    OpenCV  →  OpenGL coordinate flip is handled here.

    Parameters
    ----------
    rvec : (3, 1) Rodrigues rotation vector
    tvec : (3, 1) translation vector (metres)

    Returns
    -------
    mv : (4, 4) float64
        Column-major compatible model-view matrix.
    """
    R = rodrigues_to_matrix(rvec)

    # Flip Y and Z axes to go from OpenCV coords to OpenGL coords
    # OpenCV: X right, Y down, Z forward
    # OpenGL: X right, Y up,   Z backward
    cv_to_gl = np.diag([1.0, -1.0, -1.0])
    R_gl = cv_to_gl @ R
    t_gl = cv_to_gl @ tvec.reshape(3)

    mv = np.eye(4, dtype=np.float64)
    mv[:3, :3] = R_gl
    mv[:3, 3] = t_gl
    return mv


def build_projection_matrix(
    camera_matrix: np.ndarray,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 100.0,
) -> np.ndarray:
    """
    Convert the OpenCV 3×3 intrinsic camera matrix into an OpenGL 4×4
    projection matrix.

    The derivation maps the OpenCV projection

        [fx  0  cx]       [X]
        [ 0 fy  cy]   ·   [Y]   =  s · [u, v, 1]^T
        [ 0  0   1]       [Z]

    to the NDC cube  [-1, 1]³  expected by OpenGL.

    Parameters
    ----------
    camera_matrix : (3, 3) float64
    width, height : image dimensions in pixels
    near, far     : clipping planes in metres

    Returns
    -------
    proj : (4, 4) float64   (row-major)
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    proj = np.zeros((4, 4), dtype=np.float64)

    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height
    proj[0, 2] = 1.0 - 2.0 * cx / width
    proj[1, 2] = -(1.0 - 2.0 * cy / height)  # flip for GL Y-up
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0

    return proj


def euler_to_rotation_matrix(
    rx_deg: float, ry_deg: float, rz_deg: float,
) -> np.ndarray:
    """
    Build a 3×3 rotation matrix from Euler angles (degrees).

    Rotation order: Rz · Ry · Rx  (extrinsic XYZ).
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx),  math.cos(rx)],
    ])
    Ry = np.array([
        [ math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)],
    ])
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz),  math.cos(rz), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


def apply_extra_rotation(mv: np.ndarray,
                         rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Apply an extra Euler rotation (degrees) to an existing 4×4 model-view
    matrix.  Useful for re-orienting a model that was exported with a
    different "up" convention.
    """
    R_extra = euler_to_rotation_matrix(rx, ry, rz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_extra
    return mv @ T


def apply_scale(mv: np.ndarray, s: float) -> np.ndarray:
    """Uniformly scale the model by factor *s* within the model-view matrix."""
    S = np.eye(4, dtype=np.float64)
    S[0, 0] = S[1, 1] = S[2, 2] = s
    return mv @ S


# ──────────────────────────────────────────────
# Temporal pose smoother
# ──────────────────────────────────────────────

class PoseSmoother:
    """
    Double-EMA ("twin-alpha") pose smoother.

    Uses a responsive alpha when the pose jumps significantly,
    and a heavier alpha for steady tracking — giving both stability
    and fast recovery after re-detection.
    """

    def __init__(self, alpha: float = 0.4):
        self.alpha_normal = np.clip(alpha, 0.01, 1.0)
        self.alpha_fast   = min(alpha * 3.0, 1.0)  # faster catch-up
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None
        self._jump_thresh = 0.02  # translation jump (metres) to trigger fast alpha

    def smooth(
        self, rvec: np.ndarray, tvec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Feed a new raw (rvec, tvec) and return the smoothed version.
        """
        if self._rvec is None:
            self._rvec = rvec.copy()
            self._tvec = tvec.copy()
        else:
            # Pick alpha based on how far the new pose jumped
            dt = np.linalg.norm(tvec.ravel() - self._tvec.ravel())
            alpha = self.alpha_fast if dt > self._jump_thresh else self.alpha_normal
            self._rvec = alpha * rvec + (1.0 - alpha) * self._rvec
            self._tvec = alpha * tvec + (1.0 - alpha) * self._tvec
        return self._rvec.copy(), self._tvec.copy()

    def reset(self):
        """Clear the filter state (e.g. when tracking is lost)."""
        self._rvec = None
        self._tvec = None
