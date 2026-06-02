# Augmented Reality Pipeline — ArUco Markers + Real-Time 3D Rendering

A complete, modular augmented reality system that detects ArUco markers through a webcam, estimates their 6-DoF pose, and renders a 3D model on top of the marker in real time with correct perspective.

---

## Architecture

```
main.py                 ← Entry-point & real-time loop
config.py               ← All tuneable parameters

marker/
  marker_generator.py   ← Generates & saves ArUco marker PNG

calibration/
  calibrate_camera.py   ← Chessboard calibration + save/load
  camera_data.npz       ← (generated) intrinsic matrix & distortion

detection/
  aruco_detector.py     ← ArUco detection wrapper
  pose_estimation.py    ← solvePnP-based 6-DoF pose

render/
  opengl_renderer.py    ← Off-screen moderngl mesh renderer
  renderer.py           ← Projection maths + alpha compositing

utils/
  math_utils.py         ← OpenCV ↔ OpenGL coordinate transforms

models/
  (place your 3D model here)
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 + |
| OS | Windows 10 / 11 |
| Webcam | Any USB / built-in |
| OpenGL | 3.3 core (virtually any GPU) |

### Python packages

```
opencv-python >= 4.7
opencv-contrib-python >= 4.7
numpy >= 1.24
trimesh >= 4.0
moderngl >= 5.8
Pillow >= 9.0
scipy >= 1.10
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your 3D model

> **Note:** No 3D models are bundled with this repository — they are kept out of git to keep the repo lightweight and to avoid redistributing third-party assets. Download your own and drop it in `models/`.

Download a model from [Sketchfab](https://sketchfab.com/) (`.obj`, `.glb`, or `.ply`) and place it in the `models/` folder:

```
models/model.obj        # default path
```

Or pass a custom path when running:

```bash
python main.py models/dragon.glb
```

### 3. Generate the ArUco marker

```bash
python main.py --generate-marker
```

This creates `marker/aruco_marker.png`. **Print it** (or display it on a screen) — the detector needs a physical marker to track.

### 4. (Optional) Calibrate your camera

For the best accuracy, run the chessboard calibration:

```bash
python main.py --calibrate
```

Hold a **9 × 6 inner-corner** chessboard in front of the camera from various angles. The script collects 20 frames automatically and saves the intrinsics to `calibration/camera_data.npz`.

> If you skip this step the pipeline will use an approximate intrinsic matrix that assumes ~60° horizontal FoV. It works, but pose accuracy may suffer.

### 5. Run the AR pipeline

```bash
python main.py
```

Point the webcam at the printed ArUco marker. The 3D model will appear on top of the marker with correct perspective, rotation and scale.

Press **Q** to quit.

---

## Command-line options

| Flag | Short | Description |
|---|---|---|
| `--generate-marker` | `-m` | Generate the ArUco marker image |
| `--calibrate` | `-c` | Run interactive camera calibration |
| *(positional)* | | Path to 3D model file |

You can combine flags:

```bash
python main.py --calibrate --generate-marker models/robot.obj
```

---

## Configuration

All tuneable parameters live in **`config.py`**:

| Parameter | Default | Description |
|---|---|---|
| `ARUCO_DICT_ID` | `DICT_5X5_100` | ArUco dictionary |
| `MARKER_ID` | `0` | Which marker ID to generate / track |
| `MARKER_LENGTH_M` | `0.05` | Physical marker side (metres) |
| `MODEL_SCALE` | `1.0` | Extra uniform scale for the model |
| `MODEL_ROTATION_DEG` | `(0, 0, 0)` | Extra Euler rotation (degrees) |
| `CAMERA_INDEX` | `0` | OpenCV camera device index |
| `CAMERA_WIDTH / HEIGHT` | `1280 × 720` | Requested capture resolution |

---

## Pipeline in Detail

### Step 1 — Marker Generation

`marker/marker_generator.py` uses `cv2.aruco.generateImageMarker` from the `DICT_5X5_100` dictionary to produce a high-resolution PNG with a white border.

### Step 2 — Camera Calibration

`calibration/calibrate_camera.py` implements the classical Zhang method via `cv2.calibrateCamera`. The user presents a chessboard; the script:
1. Detects inner corners with sub-pixel refinement (`cornerSubPix`).
2. Collects ≥ 20 frames from different viewpoints.
3. Solves for the 3 × 3 intrinsic matrix **K** and the distortion vector **D**.
4. Saves them to `camera_data.npz`.

### Step 3 — ArUco Detection

`detection/aruco_detector.py` wraps the modern `cv2.aruco.ArucoDetector` API with adaptive-threshold tuning and sub-pixel corner refinement.

### Step 4 — Pose Estimation

`detection/pose_estimation.py` calls `cv2.solvePnP` (iterative LM) with the known 3-D marker corners and the detected 2-D image corners, yielding a Rodrigues rotation vector **rvec** and translation **tvec**.

### Step 5 — 3D Rendering

`render/opengl_renderer.py` creates a headless OpenGL 3.3 context via **moderngl**, loads the mesh with **trimesh**, and renders it off-screen into an RGBA framebuffer. The shaders apply Phong-like lighting (ambient + diffuse + specular).

### Step 6 — Compositing

`render/renderer.py` converts the OpenCV camera intrinsics into an OpenGL projection matrix, flips the coordinate axes (OpenCV Y-down → OpenGL Y-up), and alpha-blends the rendered overlay onto the camera frame.

---

## Supported 3D Formats

| Format | Extension | Notes |
|---|---|---|
| Wavefront OBJ | `.obj` | Most common on Sketchfab |
| glTF Binary | `.glb` | Compact single-file format |
| Stanford PLY | `.ply` | Point-cloud / mesh |
| STL | `.stl` | Also supported via trimesh |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Cannot open camera" | Check `CAMERA_INDEX` in config.py (try 0, 1, 2…) |
| Marker not detected | Print the marker larger; ensure good lighting; avoid glare |
| Model appears upside-down | Adjust `MODEL_ROTATION_DEG` in config.py, e.g. `(180, 0, 0)` |
| Model too big / too small | Adjust `MODEL_SCALE` in config.py |
| Low FPS | Reduce `CAMERA_WIDTH / HEIGHT`; use a simpler model |
| OpenGL errors | Update your GPU drivers; ensure OpenGL 3.3+ support |

---

## Standalone Module Usage

Each module can be run independently:

```bash
# Generate marker only
python -m marker.marker_generator

# Calibrate camera only
python -m calibration.calibrate_camera
```

---

## License

Academic / educational use. Model files from Sketchfab are subject to their own licences.
