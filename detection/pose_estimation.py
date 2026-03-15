"""
pose_estimation.py — 6-DoF marker pose estimation.

Given the detected 2-D corners and the camera intrinsics this module
recovers the rotation and translation that map the marker's local
coordinate frame into the camera frame.

Mathematical background
-----------------------
The Perspective-n-Point (PnP) problem finds *(R, t)* such that

    s · [u, v, 1]ᵀ  =  K · (R · Xₘ + t)

where **Xₘ** are the known 3-D coordinates of the marker corners
(a square lying in the Z = 0 plane) and **(u, v)** are the observed
pixel positions.

OpenCV's ``solvePnP`` solves this with the iterative Levenberg–Marquardt
method (flag ``SOLVEPNP_ITERATIVE``) initialised via the DLT.
"""

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class PoseEstimator:
    """
    Estimates the pose of a square planar marker in camera space.

    The marker's local frame has:
        * origin at the marker centre
        * X → right, Y → down, Z → into the marker (right-hand rule)
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        marker_length: float = config.MARKER_LENGTH_M,
    ):
        """
        Parameters
        ----------
        camera_matrix : (3, 3) float64
        dist_coeffs   : (N, 1) float64
        marker_length : float
            Physical side length of the marker in metres.
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.marker_length = marker_length

        # 3-D coordinates of the four corners in the marker's local frame
        half = marker_length / 2.0
        self.object_points = np.array([
            [-half,  half, 0],  # top-left
            [ half,  half, 0],  # top-right
            [ half, -half, 0],  # bottom-right
            [-half, -half, 0],  # bottom-left
        ], dtype=np.float32)

    def estimate(
        self,
        corners: tuple[np.ndarray, ...],
        ids: np.ndarray | None,
        target_id: int | None = None,
    ) -> list[dict]:
        """
        Estimate 6-DoF pose for each detected marker.

        Parameters
        ----------
        corners : tuple of (1, 4, 2) arrays
        ids     : (N, 1) int array or None
        target_id : int | None
            If given, only estimate poses for this marker ID.

        Returns
        -------
        results : list[dict]
            Each dict contains:
                ``id``    – marker ID
                ``rvec``  – (3, 1) Rodrigues rotation vector
                ``tvec``  – (3, 1) translation vector  (metres)
                ``corners`` – (4, 2) image corners
        """
        if ids is None or len(corners) == 0:
            return []

        results: list[dict] = []
        for i, marker_id in enumerate(ids.flatten()):
            if target_id is not None and marker_id != target_id:
                continue

            img_pts = corners[i].reshape(-1, 2).astype(np.float64)

            # Use IPPE_SQUARE — specialised solver for square planar markers
            # (more stable than generic iterative PnP for coplanar points)
            success, rvec, tvec = cv2.solvePnP(
                self.object_points,
                img_pts,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue

            # Refine with iterative LM using IPPE result as initial guess
            success2, rvec, tvec = cv2.solvePnP(
                self.object_points,
                img_pts,
                self.camera_matrix,
                self.dist_coeffs,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            results.append({
                "id": int(marker_id),
                "rvec": rvec,
                "tvec": tvec,
                "corners": img_pts,
            })

        return results

    def draw_axes(
        self,
        frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        length: float | None = None,
    ) -> np.ndarray:
        """
        Draw XYZ axes on the frame at the estimated pose.
        """
        if length is None:
            length = self.marker_length * 0.5
        cv2.drawFrameAxes(
            frame, self.camera_matrix, self.dist_coeffs,
            rvec, tvec, length,
        )
        return frame
