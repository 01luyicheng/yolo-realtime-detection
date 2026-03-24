# YOLO Real-time Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A real-time object detection application using YOLOv8 with a full GUI.

## Features

- Real-time object detection from webcam
- Support for detection and segmentation models
- Adjustable confidence and IOU thresholds
- Class filtering
- Video recording
- Screenshot capture
- FPS monitoring
- Fullscreen mode

## Installation

### Requirements

- Python 3.8+
- Webcam

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install ultralytics opencv-python numpy
```

## Usage

### Quick Start

Run the launch script:

```bash
./run_yolo.sh
```

Or run directly with Python:

```bash
python3 yolo_realtime.py
```

### Command Line Options

```bash
python3 yolo_realtime.py --model yolov8n.pt
```

## Hotkeys

| Key | Action |
|-----|--------|
| `Q` / `ESC` | Quit program |
| `B` | Toggle bounding boxes |
| `M` | Toggle masks (segmentation models only) |
| `[` / `]` | Decrease/Increase confidence threshold |
| `,` / `.` | Decrease/Increase IOU threshold |
| `;` | Adjust skip frames |
| `+` / `-` | Adjust text size |
| `R` | Start/Stop recording |
| `S` | Take screenshot |
| `H` | Show/Hide help |
| `F` | Toggle fullscreen |
| `C` | Clear statistics |
| `1-9` | Select/Cancel class filter |
| `A` | Select all classes |
| `N` | Clear class filter |

## Model

The default model is YOLOv8n (nano), which is optimized for speed.

You can replace it with other YOLOv8 models:

| Model | Speed | Accuracy |
|-------|-------|----------|
| `yolov8n.pt` (nano) | Fastest | Lowest |
| `yolov8s.pt` (small) | Fast | Low |
| `yolov8m.pt` (medium) | Medium | Medium |
| `yolov8l.pt` (large) | Slow | High |
| `yolov8x.pt` (xlarge) | Slowest | Highest |

## Project Structure

```
.
├── yolo_realtime.py    # Main program
├── yolov8n.pt          # YOLO model file (YOLOv8 nano)
├── run_yolo.sh         # Launch script
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # This file
```

## Notes

- The application will use CPU by default
- For GPU acceleration, ensure CUDA is properly installed
- Segmentation masks are only available with segmentation models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
