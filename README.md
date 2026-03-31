<div align="center">

<img src="assets/logo.png" alt="Badminton-VisionAI Logo" width="480"/>

##### AI-Powered Badminton Performance Analysis System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF?style=flat-square)](https://ultralytics.com)
[![Roboflow](https://img.shields.io/badge/Annotations-Roboflow-6706CE?style=flat-square&logo=roboflow&logoColor=white)](https://roboflow.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![Version](https://img.shields.io/badge/Version-2.1.0-blue?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE.txt)
[![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)]()

*Computer vision analytics for professional badminton performance*

**[Features](#-features) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [Installation](#-installation) · [Usage](#-usage) · [Dashboard](#-dashboard) · [Roadmap](#-roadmap)**

</div>

---

## 🎯 Overview

**Badminton-VisionAI** is an end-to-end AI pipeline that transforms raw match footage into actionable performance intelligence. Built on state-of-the-art computer vision models, it delivers frame-accurate player tracking,  shuttlecock trajectory reconstruction, intelligent shot classification, and coach-ready analytics — all packaged in an interactive dashboard.

<div align="center">

### Tracking
<img src="demos/badminton-visionAI_01.gif" width="900">

Player & shuttle tracking · Court mapping · Shot detection

### Analytics
<img src="demos/badminton-visionAI_02.gif" width="900">

Heatmap · Players & Shots statistics · Coach report

</div>

---

## ✨ Features

### 🎥 Computer Vision Pipeline
- **Player Detection & Tracking** — Player detection using YOLO, integrated with ByteTrack for robust multi-player tracking and persistent IDs across frames
- **Shuttlecock Tracking** — TrackNet-powered trajectory prediction with InpaintNet for occluded frame recovery
- **Court Detection** — Automatic ResNet-based keypoint detection and RANSAC homography estimation for court-plane normalisation
- **Segment Anything (SAM)** — Precise player segmentation for fine-grained movement analysis
- **Side Court Visualisation** — Real-time mini-court overlay projected via homography, rendered on every output frame

<p align="center">• • •</p>

### 📐 Spatial Analysis
- **Homography Projection** — Transforms camera-perspective player and ball positions onto a canonical 2D mini-court view
- **Heatmap Generation** — Positional frequency maps revealing court coverage patterns per player
- **Trajectory Reconstruction** — Smoothed shuttlecock path with InpaintNet interpolation for occluded frames
- **Ball Trail Rendering** — Configurable trail length and colour overlaid on output video

<p align="center">• • •</p>

### 🏸 Shot Intelligence
- **Shot Type Classification** — Detects smash, drop, clear, drive, net kill, and serve automatically
- **Power Estimation** — Velocity-based shuttlecock speed computation at moment of impact
- **Rally Segmentation** — Automatic rally start/end detection and per-rally statistics

<p align="center">• • •</p>

### 📊 Performance Dashboard
- **Interactive Streamlit UI** — Run the analysis pipeline, and explore results directly in the browser
- **Live Video Metrics** — Annotated output video with per-frame player stats, shot type, and speed overlays
- **Per-Player Statistics** — Shot counts, movement distances, court coverage, rally durations, and shuttlecock analytics
- **Coach-Ready Reports** — Exportable PDF summaries with performance insights and tactical recommendations

<p align="center">• • •</p>

### 🐳 Production-Ready
- **Dual Dockerfile setup** — Separate optimized containers for the CV pipeline and the web dashboard
- **Modular architecture** — Cleanly decoupled modules for tracking, analysis, detection, and visualization
- **Smart caching** — Pipeline skips re-processing if tracking JSON files already exist on disk
- **Conda + pip support** — Both `environment.yml` and `requirements*.txt` provided

---

## 🏗️ System Architecture

<div align="center">
<img src="assets/system-architecture-transparentbg.png">
</div>

---

### Project Structure

```
Badminton-visionAI/
├── tracking/                     # Tracking modules
│   ├── court_detection/          # Keypoint detection & homography
│   ├── players_tracking/         # YOLO-based player tracking
│   ├── players_poses/            # Pose estimation (YOLO Pose)
│   └── shuttle_tracking/         # TrackNet ball tracking (models, predict, dataset)
├── shot_detection/               # Shot classification, power estimation, stabilizer
├── analysis/                     # Metrics, heatmaps, side-court projection, dashboard
├── webapp/                       # Streamlit dashboard
│   ├── pages/                    # match_replay, court_explorer, coach_report
│   ├── tabs/                     # positioning, shot_profile
│   └── reports/                  # PDF report helpers and page builders
├── utils/                        # Shared utilities (I/O, video, logger, conversions)
├── constants/                    # Court dimensions, player heights
├── config/                       # config.yaml, streamlitconfig.yaml
├── models/                       # Pretrained weights
│   ├── players_tracking/         # yolov8m.pt
│   ├── players_poses/            # yolo_poses_model.pt
│   ├── sam_model/                # sam.pth
│   └── shuttle_ball_tracking/    # TrackNet_best.pt, InpaintNet_best.pt
├── notebooks/                    # data_exploration.ipynb, training_models.ipynb
├── data/
│   ├── Input_videos/             # Drop your match videos here
│   ├── json/                     # Pipeline output JSON (read by dashboard)
│   └── images/                   # Sample court/player images
├── outputs/
│   └── tracking_results/         # Annotated output videos
├── logs/                         # Runtime logs per module
├── docs/                         # Extended documentation
├── app.py                        # ← Run this FIRST (CV pipeline entry point)
├── web.py                        # ← Run this SECOND (Streamlit dashboard)
├── Dockerfile.pipeline           # CV pipeline container
├── Dockerfile.web                # Web dashboard container
├── environment.yml               # Conda environment
├── requirements.pipeline.txt
└── requirements.web.txt
```

---

## 🚀 Quick Start

Badminton-VisionAI runs in **two sequential steps**: the CV pipeline processes your video and writes all output data first, then the web dashboard reads those outputs for interactive exploration.

### Step 1 — Run the CV Pipeline

```bash
# Clone the repository
git clone https://github.com/amramer/Badminton-visionAI.git
cd Badminton-visionAI

# Build and run the pipeline container
docker build -f Dockerfile.pipeline -t badminton-pipeline .
docker run --gpus all \
  -v $(pwd)/data/Input_videos:/app/data/Input_videos \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data/json:/app/data/json \
  badminton-pipeline \
  --input data/Input_videos/your_match.mp4 --output outputs/
```

This runs all 9 pipeline stages and produces the annotated video and all JSON data files consumed by the dashboard.

### Step 2 — Launch the Web Dashboard

```bash
# Build and run the dashboard container (after the pipeline has finished)
docker build -f Dockerfile.web -t badminton-web .
docker run -p 8501:8501 \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data/json:/app/data/json \
  badminton-web

# Open your browser at http://localhost:8501
```

---


## 🛠️ Installation

### Option A — Conda (Recommended for full stack)

```bash
git clone https://github.com/amramer/Badminton-visionAI.git
cd Badminton-visionAI

conda env create -f environment.yml
conda activate badminton-visionai
```

### Option B — pip (CV pipeline only)

```bash
pip install -r requirements.pipeline.txt
```

### Option C — pip (Web dashboard only)

```bash
pip install -r requirements.web.txt
```

### Option D — Docker (Production, two containers)

```bash
# Container 1 — CV pipeline (run first)
docker build -f Dockerfile.pipeline -t badminton-pipeline .

# Container 2 — Web dashboard (run after pipeline completes)
docker build -f Dockerfile.web -t badminton-web .
```

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 (CUDA recommended) |
| OpenCV | ≥ 4.8 |
| Streamlit | ≥ 1.32 |
| Supervision | latest |
| CUDA (optional) | ≥ 11.8 for GPU inference |

---

## 📦 Model Setup

This repository does not include pre-trained model weights. Download and place the following models in their respective directories before running the pipeline:

```bash
# Download all models (run from project root)
mkdir -p models/{players_tracking,sam_model,players_poses,shuttle_ball_tracking}
wget -O models/players_tracking/yolov8m.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt
wget -O models/sam_model/sam.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -O models/players_poses/yolo_poses_model.pt https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt
```

### For TrackNet and InpaintNet:
- 1. Download from https://github.com/ChgygLin/TrackNet-pytorch/releases
- 2. Place both files in models/shuttle_ball_tracking/

### Verify installation:

```bash
ls -l models/{players_tracking,sam_model,players_poses,shuttle_ball_tracking}/
```

### Required directory structure:

```bash
models/
├── players_tracking/yolov8m.pt
├── sam_model/sam.pth
├── players_poses/yolo_poses_model.pt
└── shuttle_ball_tracking/
    ├── TrackNet_best.pt
    └── InpaintNet_best.pt
```
---
## 📖 Usage

### 1. Run the Full Pipeline on a Video

Place your video in `data/Input_videos/`, update `video.input_video` in `config/config.yaml`, then run:

```bash
python app.py
```

The pipeline executes 9 sequential stages — initialisation, config loading, video decoding, court detection, player tracking, ball tracking, shot detection, side-court visualisation, and dashboard generation — writing all results to `data/json/` and the annotated video to `outputs/tracking_results/`.

> **Smart caching:** If `data/json/players_tracking.json` or `data/json/ball_tracking.json` already exist from a previous run, those tracking stages are skipped automatically. Delete the relevant JSON files to force re-processing.

**Key `config/config.yaml` settings:**

| Key | Default | Description |
|---|---|---|
| `video.input_video` | `data/Input_videos/badminton_Japan_Open_2024.mp4` | Path to input video |
| `video.final_output` | `outputs/tracking_results/final_analysis.mp4` | Path for annotated output video |
| `video.max_frames` | `null` | Limit frames processed (`null` = all) |
| `video.show_progress` | `true` | Show tqdm progress bars |
| `players.players_tracker_batch_size` | `4` | YOLO inference batch size |
| `ball.tracker_batch_size` | `4` | TrackNet inference batch size |
| `shot_detection.api_key` | — | Your Roboflow API key |
| `shot_detection.conf_threshold` | `0.3` | Shot detection confidence threshold |
| `court_keypoints.use_fixed_keypoints` | `true` | Load keypoints from JSON instead of running model |

### 2. Launch the Interactive Dashboard

```bash
# Only after app.py has finished successfully
streamlit run web.py
```

Navigate to `http://localhost:8501`. The dashboard reads `data/json/` and `outputs/tracking_results/` — it does **not** re-run the pipeline.

### 3. Run via Python API

```python
from tracking.players_tracking.player_tracking import PlayerTracker
from tracking.shuttle_tracking.ball_tracking import BallTracker
from analysis.sidecourt import SideCourt
from config import load_config

config = load_config()

# Initialize components
player_tracker = PlayerTracker(
    model_path=config["players"]["players_tracker_model"],
    batch_size=config["players"].get("players_tracker_batch_size", 4),
    frame_rate=fps,
)

ball_tracker = BallTracker(
    tracking_model_path=config["ball"]["tracker_model"],
    inpaint_model_path=config["ball"]["inpaint_model"],
    batch_size=config["ball"].get("tracker_batch_size", 4),
    frame_rate=fps
)
```

---

## 📊 Dashboard

The Streamlit dashboard provides an end-to-end interactive experience for exploring pipeline results:

| Panel | Description |
|---|---|
| 🎯 **Tracking View** | Annotated output video with bounding boxes, SAM masks, player IDs, ball trail, and shot overlays |
| 🗺️ **Court Explorer** | Interactive homographic court projection with player and ball positions |
| 🔥 **Positioning Tab** | Per-player heatmaps showing court coverage and movement patterns |
| 📈 **Shot Profile Tab** | Shot type breakdown, speed distribution, and per-player shot history |
| 🎬 **Match Replay** | Frame-by-frame playback with all tracking overlays |
| 📄 **Coach Report** | Exportable PDF with performance insights and tactical recommendations |

---

## 🧠 Models & Methods

| Component | Model / Method | Weight file |
|---|---|---|
| Player Detection | YOLOv8m (Ultralytics) | `models/players_tracking/yolov8m.pt` |
| Player Segmentation | SAM ViT-H (Segment Anything) | `models/sam_model/sam.pth` |
| Player Poses | YOLO Pose | `models/players_pose/yolo_poses_model.pt` |
| Shuttle Tracking | TrackNet | `models/shuttle_ball_tracking/TrackNet_best.pt` |
| Occlusion Recovery | InpaintNet | `models/shuttle_ball_tracking/InpaintNet_best.pt` |
| Court Detection | ResNet keypoints + RANSAC homography | Fixed keypoints via `data/json/court_keypoints.json` |
| Shot Classification | Roboflow-hosted model (skeleton features) | Configured via `shot_detection` block in config |
| Tracking Algorithm | SORT / ByteTrack-style ID assignment | Via `supervision` library |

---

## ⚙️ Configuration

All pipeline parameters live in `config/config.yaml`. A separate `config/streamlitconfig.yaml` governs dashboard display settings. Below is an annotated summary of the key sections:

```yaml
# Video paths
video:
  input_video: "data/Input_videos/badminton_Japan_Open_2024.mp4"
  court_detection_output: "outputs/tracking_results/court_detection.mp4"
  final_output: "outputs/tracking_results/final_analysis.mp4"
  max_frames: null        # null = process entire video
  show_progress: true

# Court keypoint detection
court_keypoints:
  court_keypoints_path: "data/json/court_keypoints.json"
  use_fixed_keypoints: true   # true = load from JSON, false = run ResNet model
  model_type: "resnet"

# Player tracking & annotation
players:
  players_tracker_model: "models/players_tracking/yolov8m.pt"
  sam_model_type: "vit_h"
  sam_model: "models/sam_model/sam.pth"
  players_tracking_path: "data/json/players_tracking.json"
  players_tracker_batch_size: 4
  annotation:
    generate_masks: true
    mask_color: "RED"
    label_color: "BLUE"
    ellipse_color: "BLUE"
    show_confidence: true

# Player name mapping (update to match your video)
player_mapping:
  player_1:
    id: 1
    name: "CHOU T.C."
  player_2:
    id: 2
    name: "LANIER"

# Shuttlecock tracking
ball:
  tracker_model: "models/shuttle_ball_tracking/TrackNet_best.pt"
  inpaint_model: "models/shuttle_ball_tracking/InpaintNet_best.pt"
  tracking_path: "data/json/ball_tracking.json"
  tracker_batch_size: 4
  trail_length: 8         # number of frames to show in ball trail
  ball_color: "YELLOW"
  trail_color: "CYAN"
  ball_radius: 6

# Shot detection via Roboflow
shot_detection:
  api_key: "YOUR_ROBOFLOW_API_KEY"
  project_name: "your-roboflow-project"
  version: 1
  shot_events_path: "data/json/shot_events.json"
  player_shots_path: "data/json/player_shot_history.json"
  conf_threshold: 0.3

# Mini-court overlay
side_court:
  position: "top_right"   # "top_right" or "bottom_left"
  scale_factor: 1.0
  alpha: 0.36

# Dashboard overlay panels
dashboard:
  dashboard_output_dir: "data/json"
  metrics_save_interval: 1   # save metrics every N frames (1 = every frame)
```

---

## 📁 Output Files

After `app.py` completes, outputs are written to two locations that `web.py` reads directly:

```
data/json/                           # Structured results (read by web.py)
├── players_tracking.json            # Frame-by-frame player positions & IDs
├── players_final_metrics.json       # Per-player aggregated statistics
├── player_shot_history.json         # Full shot timeline per player
├── ball_tracking.json               # Raw shuttlecock trajectory per frame
├── ball_final_metrics.json          # Shuttlecock speed & flight metrics
├── shot_events.json                 # Detected shot events with type & frame index
├── final_shots_stats.json           # Aggregated shot type statistics
└── court_keypoints.json             # Detected court keypoints for homography

outputs/tracking_results/            # Video outputs (read by web.py)
├── final_analysis.mp4               # Fully annotated video (all overlays)
└── court_detection.mp4              # Court keypoint detection visualisation
```

> **Caching:** If `players_tracking.json` or `ball_tracking.json` already exist, those pipeline stages are skipped on the next run. Delete these files to force a fresh tracking pass.

---

## 🗺️ Roadmap

- [x] Player detection and tracking (YOLOv8m + SAM ViT-H)
- [x] Shuttlecock tracking (TrackNet + InpaintNet)
- [x] Court keypoint detection and homography projection
- [x] Shot type classification and power estimation
- [x] Side-court mini overlay on output video
- [x] Streamlit interactive dashboard (replay, heatmaps, shot profile, coach report)
- [x] Docker containerisation (pipeline + dashboard)
- [x] Smart JSON caching to skip re-processing
- [ ] Multi-camera support and view stitching
- [ ] Real-time inference mode (RTSP / webcam input)
- [ ] Pose estimation deeper integration (MediaPipe / ViTPose)
- [ ] 3D trajectory reconstruction
- [ ] REST API endpoint for external integrations
- [ ] Mobile-friendly dashboard (PWA)
- [ ] Automated highlight reel generation

---

## 🤝 Contributing

Contributions are very welcome! Here's how to get involved:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add: brief description of change'`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request — describe what you did and why

Please check open [Issues](https://github.com/amramer/Badminton-visionAI/issues) before starting work to avoid duplication.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE.txt](LICENSE.txt) for details. You are free to use, modify, and distribute with attribution.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 
- [TrackNet](https://github.com/ChgygLin/TrackNet-pytorch) 
- [InpaintNet](https://github.com/ChgygLin/TrackNet-pytorch) 
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [Roboflow](https://roboflow.com)
- [Supervision](https://github.com/roboflow/supervision)
- [Streamlit](https://streamlit.io) 
- [OpenCV](https://opencv.org)

---

<div align="center">

Made with ❤️ by [Amr Amer](https://github.com/amramer)

⭐ **If this project helps you, please give it a star!** ⭐

[![GitHub stars](https://img.shields.io/github/stars/amramer/Badminton-visionAI?style=social)](https://github.com/amramer/Badminton-visionAI)

</div>
