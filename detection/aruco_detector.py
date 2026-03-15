"""
aruco_detector.py — ArUco marker detection wrapper.

Encapsulates the OpenCV ArUco detection pipeline (dictionary look-up,
parameter tuning, corner refinement) behind a clean class interface
used by the rest of the project.

Key concepts
------------
* **Dictionary** – a predefined set of binary patterns (e.g. 5×5, 100 ids).
* **DetectorParameters** – knobs for adaptive thresholding, corner refinement,
  error-correction bits, etc.
* Each call to ``detect()`` returns the list of detected marker corners
  (in image-pixel coordinates) and their integer IDs.
"""

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ArucoDetector:
    """
    Robust ArUco detector with CLAHE pre-processing and tuned parameters
    for maximum detection rate under varying lighting conditions.
    """

    def __init__(
        self,
        dictionary_id: int = config.ARUCO_DICT_ID,
        refine_strategy: int = cv2.aruco.CORNER_REFINE_SUBPIX,
    ):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        p = cv2.aruco.DetectorParameters()

        # Corner refinement
        p.cornerRefinementMethod = refine_strategy
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 50
        p.cornerRefinementMinAccuracy = 0.01

        # Adaptive thresholding — wider window range catches more lighting
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 53
        p.adaptiveThreshWinSizeStep = 4
        p.adaptiveThreshConstant = 7

        # Be more permissive with marker candidates
        p.minMarkerPerimeterRate = 0.01     # detect smaller markers
        p.maxMarkerPerimeterRate = 4.0
        p.polygonalApproxAccuracyRate = 0.05
        p.minCornerDistanceRate = 0.02
        p.minMarkerDistanceRate = 0.02

        # Error correction — tolerate up to 1-bit errors
        p.errorCorrectionRate = 0.6

        # Perspective removal
        p.perspectiveRemovePixelPerCell = 4
        p.perspectiveRemoveIgnoredMarginPerCell = 0.13

        self.parameters = p
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.parameters
        )

        # CLAHE for adaptive contrast enhancement
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(
        self, frame: np.ndarray
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray | None]:
        """
        Detect ArUco markers with CLAHE-enhanced contrast.
        Uses half-resolution internally for speed, then scales corners back.
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Downscale for faster detection
        h, w = gray.shape[:2]
        small = cv2.resize(gray, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        # CLAHE on smaller image (much faster)
        enhanced = self._clahe.apply(small)
        corners, ids, _rejected = self.detector.detectMarkers(enhanced)

        # Scale corners back to original resolution
        if ids is not None and len(corners) > 0:
            corners = tuple(c * 2.0 for c in corners)

        return corners, ids

    def draw_detections(
        self, frame: np.ndarray,
        corners: tuple[np.ndarray, ...],
        ids: np.ndarray | None,
    ) -> np.ndarray:
        """
        Draw detected marker boundaries and IDs on the image (in-place).
        """
        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        return frame
