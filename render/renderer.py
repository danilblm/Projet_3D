"""
renderer.py — High-level AR compositor with runtime controls.

Bridges the OpenGL renderer and the camera frame:

1. Takes the camera image and the estimated marker pose.
2. Applies temporal pose smoothing (EMA filter).
3. Computes the OpenGL projection and model-view matrices.
4. Calls ``OpenGLRenderer.render()`` to produce an RGBA overlay.
5. Alpha-composites the overlay onto the camera image.

Runtime controls (scale & rotation) are mutable and driven from main.py.
"""

from __future__ import annotations

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.math_utils import (
    build_model_view_matrix,
    build_projection_matrix,
    apply_extra_rotation,
    apply_scale,
    PoseSmoother,
)
from render.opengl_renderer import OpenGLRenderer, MeshData


class ARRenderer:
    """
    Manages the full render-and-composite cycle with runtime controls.
    """

    def __init__(
        self,
        mesh_data: MeshData,
        camera_matrix: np.ndarray,
        width: int = config.RENDER_WIDTH,
        height: int = config.RENDER_HEIGHT,
    ):
        self.width = width
        self.height = height

        # Pre-compute the GL projection matrix (constant for a given camera)
        self.projection = build_projection_matrix(
            camera_matrix, width, height,
            near=config.NEAR_PLANE, far=config.FAR_PLANE,
        )

        # Instantiate the off-screen OpenGL renderer
        self.gl_renderer = OpenGLRenderer(mesh_data, width, height)

        # Temporal pose smoother
        self.smoother = PoseSmoother(alpha=config.POSE_SMOOTH_ALPHA)

        # ── Runtime-mutable controls ──
        self.scale = config.MODEL_SCALE
        self.extra_rotation = list(config.MODEL_ROTATION_DEG)  # [rx, ry, rz]

    def swap_model(self, mesh_data: MeshData):
        """Instantly swap the displayed 3D model (GPU upload only)."""
        self.gl_renderer.swap_model(mesh_data)
        self.smoother.reset()

    def set_wireframe(self, enabled: bool):
        self.gl_renderer.set_wireframe(enabled)

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> np.ndarray:
        """
        Render the 3D model at the given pose and composite onto *frame*.
        """
        # 0. Temporal smoothing
        rvec, tvec = self.smoother.smooth(rvec, tvec)

        # 1. Build model-view matrix  (OpenCV → OpenGL coords)
        mv = build_model_view_matrix(rvec, tvec)

        # 2. Apply runtime extra rotation
        rx, ry, rz = self.extra_rotation
        if rx != 0 or ry != 0 or rz != 0:
            mv = apply_extra_rotation(mv, rx, ry, rz)

        # 3. Apply runtime scale
        if self.scale != 1.0:
            mv = apply_scale(mv, self.scale)

        # 4. Off-screen render  →  RGBA overlay
        rgba = self.gl_renderer.render(self.projection, mv)

        # 5. Alpha-composite onto the camera frame
        return self._composite(frame, rgba)

    def on_tracking_lost(self):
        """Call when the marker is no longer visible."""
        self.smoother.reset()

    def release(self):
        """Free GPU resources."""
        self.gl_renderer.release()

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _composite(bg: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
        """
        Alpha-blend an RGBA overlay on top of a BGR background.

        Uses the standard "over" compositing operator:
            C_out = α · C_fg  +  (1 − α) · C_bg
        """
        alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        fg_bgr = overlay_rgba[:, :, :3][:, :, ::-1]  # RGBA → BGR

        out = bg.copy().astype(np.float32)
        fg = fg_bgr.astype(np.float32)

        out = alpha * fg + (1.0 - alpha) * out
        return np.clip(out, 0, 255).astype(np.uint8)
