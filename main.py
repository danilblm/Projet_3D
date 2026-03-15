"""
main.py — Entry-point for the Augmented Reality pipeline.

Features:
    • ArUco marker detection + solvePnP pose estimation
    • Real-time 3D model rendering (off-screen OpenGL)
    • Instant model switching  (N / P)
    • Auto-rotation turntable  (T)
    • Wireframe mode            (F)
    • Screenshot capture        (C)
    • Video recording           (R)
    • Tracking persistence      (keeps last pose for N frames)
    • Multi-marker support      (each marker ID → different model)
    • Fullscreen toggle         (F11)
    • Pose smoothing (EMA)
    • HUD overlay with all info

Usage:
    python main.py                       # default model path from config
    python main.py models/car.glb        # custom model path
    python main.py --calibrate           # run calibration first
    python main.py --generate-marker     # generate the ArUco marker image
"""

from __future__ import annotations

import sys
import os
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import config
from marker.marker_generator import generate_marker
from calibration.calibrate_camera import (
    load_calibration,
    default_camera_matrix,
    collect_calibration_frames,
    calibrate_and_save,
)
from detection.aruco_detector import ArucoDetector
from detection.pose_estimation import PoseEstimator
from render.renderer import ARRenderer
from render.opengl_renderer import preload_mesh


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

MODEL_EXTENSIONS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".fbx", ".dae"}


def _parse_args() -> dict:
    args = {
        "model_path": config.MODEL_PATH,
        "calibrate": False,
        "generate_marker": False,
    }
    for a in sys.argv[1:]:
        if a in ("--calibrate", "-c"):
            args["calibrate"] = True
        elif a in ("--generate-marker", "-m"):
            args["generate_marker"] = True
        elif not a.startswith("-"):
            args["model_path"] = a
    return args


def _scan_models(models_dir: str) -> list[str]:
    if not os.path.isdir(models_dir):
        return []
    files = []
    for f in sorted(os.listdir(models_dir)):
        if os.path.splitext(f)[1].lower() in MODEL_EXTENSIONS:
            files.append(os.path.join(models_dir, f))
    return files


def _take_screenshot(frame: np.ndarray) -> str:
    d = config.SCREENSHOT_DIR
    os.makedirs(d, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(d, f"screenshot_{ts}.png")
    cv2.imwrite(path, frame)
    return path


# ──────────────────────────────────────────────
# HUD
# ──────────────────────────────────────────────

HELP_TEXT = [
    "CONTROLS:",
    "  Q          Quit",
    "  H          Toggle this help",
    "  N / P      Next / Previous model",
    "  M          Toggle multi-marker mode",
    "  +/-        Scale model up/down",
    "  W/S        Rotate X +/-",
    "  A/D        Rotate Y +/-",
    "  Z/X        Rotate Z +/-",
    "  T          Toggle auto-rotation",
    "  F          Toggle wireframe",
    "  C          Screenshot",
    "  R          Toggle recording",
    "  F11        Fullscreen",
    "  0          Reset scale & rotation",
]


def _draw_hud(frame, fps, scale, rot, tracking, recording, show_help,
              model_name="", model_idx=0, model_count=0,
              turntable=False, wireframe=False):
    h, fw = frame.shape[:2]

    # ── Top bar ──
    status = "TRACKING" if tracking else "NO MARKER"
    extras = ""
    if turntable:
        extras += "  [SPIN]"
    if wireframe:
        extras += "  [WIRE]"
    if recording:
        extras += "  [REC]"
    info = (f"FPS: {fps:.0f}  |  Scale: {scale:.2f}  |  "
            f"Rot: ({rot[0]:.0f},{rot[1]:.0f},{rot[2]:.0f})  |  "
            f"{status}{extras}")

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, 38), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, info, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1,
                cv2.LINE_AA)

    # Tracking dot
    color = (0, 220, 0) if tracking else (0, 0, 220)
    cv2.circle(frame, (fw - 20, 20), 8, color, -1, cv2.LINE_AA)

    # Recording dot
    if recording:
        cv2.circle(frame, (fw - 50, 20), 8, (0, 0, 255), -1, cv2.LINE_AA)

    # ── Bottom bar — model name ──
    if model_count > 0:
        model_info = f"Model [{model_idx+1}/{model_count}]: {model_name}"
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (0, h - 34), (fw, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay3, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, model_info, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 200, 255), 1,
                    cv2.LINE_AA)

    # ── Help panel ──
    if show_help:
        panel_h = 22 * len(HELP_TEXT) + 20
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (8, 44), (320, 44 + panel_h),
                       (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
        for i, line in enumerate(HELP_TEXT):
            cv2.putText(frame, line, (16, 66 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)

    return frame


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def run_ar_pipeline(model_path: str, camera_matrix: np.ndarray,
                    dist_coeffs: np.ndarray) -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    if not cap.isOpened():
        sys.exit("[main] ERROR: Cannot open camera.")

    ret, test_frame = cap.read()
    if not ret:
        sys.exit("[main] ERROR: Cannot read from camera.")
    h, w = test_frame.shape[:2]
    print(f"[main] Camera resolution: {w}x{h}")

    # ── Scan & preload models ──
    models_dir = os.path.join(PROJECT_ROOT, "models")
    model_list = _scan_models(models_dir)
    # Normalise paths so CLI arg (relative) matches scan result (absolute)
    model_path_abs = os.path.abspath(model_path)
    abs_set = {os.path.abspath(p) for p in model_list}
    if model_path_abs not in abs_set:
        if os.path.isfile(model_path):
            model_list.insert(0, model_path_abs)
    # Ensure all paths are absolute (deduplicated)
    model_list = [os.path.abspath(p) for p in model_list]
    if not model_list:
        model_list = [model_path_abs]
    model_idx = 0
    if model_path_abs in model_list:
        model_idx = model_list.index(model_path_abs)

    print(f"[main] Pre-loading {len(model_list)} model(s)...")
    mesh_cache = [preload_mesh(mp) for mp in model_list]
    print(f"[main] All models pre-loaded!\n")

    # ── Pipeline ──
    detector = ArucoDetector()
    pose_est = PoseEstimator(camera_matrix, dist_coeffs)
    ar_renderer = ARRenderer(mesh_cache[model_idx], camera_matrix, w, h)

    # ── State ──
    fps_start = time.time()
    fps_count = 0
    fps_display = 0.0
    show_help = True
    recording = False
    writer = None
    was_tracking = False
    fullscreen = False

    # Turntable
    turntable = False
    turntable_angle = 0.0
    last_time = time.time()

    # Wireframe
    wireframe = False

    # Tracking persistence
    persist_max = config.TRACKING_PERSIST_FRAMES
    frames_since_lost = persist_max + 1
    last_rvec = None
    last_tvec = None

    # Multi-marker: map marker_id -> model_idx (ID 0 -> model 0, ID 1 -> model 1, etc.)
    # Disabled by default; user can enable with M key if they have multiple markers.
    multi_marker_active = False

    # Frame counter for detection skip
    frame_num = 0
    detect_every = 2  # detect every N frames, reuse last pose otherwise
    cached_corners = None
    cached_ids = None

    print(f"[main] AR pipeline running.  Press H for help, Q to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()
            dt = now - last_time
            last_time = now
            frame_num += 1

            # Turntable auto-rotation
            if turntable:
                turntable_angle += config.TURNTABLE_SPEED * dt
                ar_renderer.extra_rotation[1] = turntable_angle % 360.0

            tracking = False

            # 1. Detect markers (skip some frames for speed)
            if frame_num % detect_every == 0:
                cached_corners, cached_ids = detector.detect(frame)
            corners, ids = cached_corners, cached_ids

            # 2. Pose estimation + 3D rendering
            if ids is not None:
                # Multi-marker: switch model based on first detected ID
                if multi_marker_active and len(ids) > 0:
                    detected_id = int(ids[0][0])
                    target_model_idx = detected_id % len(mesh_cache)
                    if target_model_idx != model_idx:
                        model_idx = target_model_idx
                        ar_renderer.swap_model(mesh_cache[model_idx])
                        print(f"  Marker #{detected_id} -> {mesh_cache[model_idx].name}")

                poses = pose_est.estimate(corners, ids,
                                          target_id=None)  # accept any marker
                for pose in poses:
                    tracking = True
                    last_rvec = pose["rvec"]
                    last_tvec = pose["tvec"]
                    frames_since_lost = 0
                    frame = ar_renderer.draw(frame, last_rvec, last_tvec)

            # 3. Tracking persistence — use last pose for a few frames
            if not tracking and last_rvec is not None and frames_since_lost < persist_max:
                frame = ar_renderer.draw(frame, last_rvec, last_tvec)
                frames_since_lost += 1
                tracking = True  # show as tracking in HUD

            # Reset smoother only when persistence has fully expired
            if was_tracking and not tracking and frames_since_lost >= persist_max:
                ar_renderer.on_tracking_lost()
                last_rvec = None
                last_tvec = None
            was_tracking = tracking

            # 4. Flip for mirror view
            frame = cv2.flip(frame, 1)

            # 5. FPS
            fps_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count = 0
                fps_start = time.time()

            # 6. HUD
            frame = _draw_hud(
                frame, fps_display,
                ar_renderer.scale, ar_renderer.extra_rotation,
                tracking, recording, show_help,
                model_name=mesh_cache[model_idx].name,
                model_idx=model_idx,
                model_count=len(model_list),
                turntable=turntable,
                wireframe=wireframe,
            )

            # 7. Recording
            if recording and writer is not None:
                writer.write(frame)

            cv2.imshow(config.WINDOW_NAME, frame)

            # 8. Keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("h"):
                show_help = not show_help

            # Scale
            elif key in (ord("+"), ord("=")):
                ar_renderer.scale += config.SCALE_STEP
                print(f"  Scale -> {ar_renderer.scale:.2f}")
            elif key == ord("-"):
                ar_renderer.scale = max(0.1, ar_renderer.scale - config.SCALE_STEP)
                print(f"  Scale -> {ar_renderer.scale:.2f}")

            # Rotation
            elif key == ord("w"):
                ar_renderer.extra_rotation[0] += config.ROTATION_STEP
                print(f"  Rotation -> {ar_renderer.extra_rotation}")
            elif key == ord("s"):
                ar_renderer.extra_rotation[0] -= config.ROTATION_STEP
                print(f"  Rotation -> {ar_renderer.extra_rotation}")
            elif key == ord("a"):
                ar_renderer.extra_rotation[1] += config.ROTATION_STEP
                print(f"  Rotation -> {ar_renderer.extra_rotation}")
            elif key == ord("d"):
                ar_renderer.extra_rotation[1] -= config.ROTATION_STEP
                print(f"  Rotation -> {ar_renderer.extra_rotation}")
            elif key == ord("z"):
                ar_renderer.extra_rotation[2] += config.ROTATION_STEP
                print(f"  Rotation -> {ar_renderer.extra_rotation}")
            elif key == ord("x"):
                ar_renderer.extra_rotation[2] -= config.ROTATION_STEP
                print(f"  Rotation -> {ar_renderer.extra_rotation}")

            # Reset
            elif key == ord("0"):
                ar_renderer.scale = 1.0
                ar_renderer.extra_rotation = [0.0, 0.0, 0.0]
                turntable_angle = 0.0
                print("  Reset scale & rotation")

            # Model switch (manual — disables multi-marker auto-switch)
            elif key == ord("n"):
                if len(model_list) > 1:
                    multi_marker_active = False
                    model_idx = (model_idx + 1) % len(model_list)
                    ar_renderer.swap_model(mesh_cache[model_idx])
                    print(f"  Model -> {mesh_cache[model_idx].name}")
            elif key == ord("p"):
                if len(model_list) > 1:
                    multi_marker_active = False
                    model_idx = (model_idx - 1) % len(model_list)
                    ar_renderer.swap_model(mesh_cache[model_idx])
                    print(f"  Model -> {mesh_cache[model_idx].name}")

            # Multi-marker toggle
            elif key == ord("m"):
                multi_marker_active = not multi_marker_active
                print(f"  Multi-marker {'ON' if multi_marker_active else 'OFF'}")

            # Turntable
            elif key == ord("t"):
                turntable = not turntable
                if not turntable:
                    turntable_angle = 0.0
                    ar_renderer.extra_rotation[1] = 0.0
                print(f"  Turntable {'ON' if turntable else 'OFF'}")

            # Wireframe
            elif key == ord("f"):
                wireframe = not wireframe
                ar_renderer.set_wireframe(wireframe)
                print(f"  Wireframe {'ON' if wireframe else 'OFF'}")

            # Screenshot
            elif key == ord("c"):
                path = _take_screenshot(frame)
                print(f"  Screenshot -> {path}")

            # Recording
            elif key == ord("r"):
                if not recording:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        config.RECORD_PATH, fourcc,
                        config.RECORD_FPS, (w, h),
                    )
                    recording = True
                    print(f"  Recording started -> {config.RECORD_PATH}")
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print("  Recording stopped.")

            # Fullscreen (F11 = key code varies; use 0x7A on Windows)
            elif key == 122:  # F11 on many systems
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(config.WINDOW_NAME,
                                          cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(config.WINDOW_NAME,
                                          cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)
                print(f"  Fullscreen {'ON' if fullscreen else 'OFF'}")

    finally:
        if writer is not None:
            writer.release()
        ar_renderer.release()
        cap.release()
        cv2.destroyAllWindows()
        print("[main] Pipeline shut down.")


# ──────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args["generate_marker"]:
        generate_marker()
        print()

    if not os.path.isfile(config.MARKER_IMAGE_PATH):
        print("[main] No marker image found — generating one now.")
        generate_marker()
        print()

    if args["calibrate"]:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            sys.exit("[main] ERROR: Cannot open camera for calibration.")
        try:
            obj_pts, img_pts, img_sz = collect_calibration_frames(cap)
            if len(obj_pts) >= 5:
                calibrate_and_save(obj_pts, img_pts, img_sz)
            else:
                print("[main] Not enough frames — skipping calibration.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
        print()

    try:
        camera_matrix, dist_coeffs = load_calibration()
    except FileNotFoundError:
        print("[main] No calibration file found — using approximate intrinsics.")
        camera_matrix, dist_coeffs = default_camera_matrix(
            config.CAMERA_WIDTH, config.CAMERA_HEIGHT
        )

    model_path = args["model_path"]
    if not os.path.isfile(model_path):
        # Try first model in models/ folder
        models_dir = os.path.join(PROJECT_ROOT, "models")
        candidates = _scan_models(models_dir)
        if candidates:
            model_path = candidates[0]
            print(f"[main] Using first available model: {os.path.basename(model_path)}")
        else:
            sys.exit(
                f"[main] ERROR: No 3D model found.\n"
                f"  Place .glb/.obj/.ply files in the 'models/' folder."
            )

    run_ar_pipeline(model_path, camera_matrix, dist_coeffs)


if __name__ == "__main__":
    main()
