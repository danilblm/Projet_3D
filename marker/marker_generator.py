"""
marker_generator.py — Generate and save ArUco marker images.

Uses the OpenCV ArUco module to produce a single marker from the dictionary
defined in config.py.  The marker is saved as a high-resolution PNG that the
user can print and place in the scene.

Usage (standalone):
    python -m marker.marker_generator
"""

import os
import sys
import cv2
import numpy as np

# Allow running both as a module and standalone script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def generate_marker(
    dictionary_id: int = config.ARUCO_DICT_ID,
    marker_id: int = config.MARKER_ID,
    size_px: int = config.MARKER_SIZE_PX,
    output_path: str = config.MARKER_IMAGE_PATH,
    border_bits: int = 1,
) -> np.ndarray:
    """
    Generate a single ArUco marker and save it to disk.

    Parameters
    ----------
    dictionary_id : int
        OpenCV enum for the ArUco dictionary (e.g. cv2.aruco.DICT_5X5_100).
    marker_id : int
        ID of the marker inside the chosen dictionary.
    size_px : int
        Side length of the output image in pixels.
    output_path : str
        File path where the PNG will be written.
    border_bits : int
        Width of the white border around the marker (in marker-bit units).

    Returns
    -------
    marker_img : np.ndarray
        The generated marker image (grayscale, uint8).
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)

    # generateImageMarker is the modern API (OpenCV ≥ 4.7)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_px)

    # Add a white border so the marker is easier to detect on paper
    border_px = int(size_px * 0.1)
    marker_with_border = cv2.copyMakeBorder(
        marker_img,
        border_px, border_px, border_px, border_px,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, marker_with_border)
    print(f"[marker_generator] Marker ID {marker_id} saved → {output_path}  "
          f"({marker_with_border.shape[1]}×{marker_with_border.shape[0]} px)")
    return marker_with_border


# ──────────────────────────────────────────────
# Standalone entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    img = generate_marker()
    cv2.imshow("Generated ArUco Marker", img)
    print("Press any key to close…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
