YOLO Real-time Detection
========================

A real-time object detection application using YOLOv8 with a full GUI.

Files
-----
- yolo_realtime.py  : Main program
- yolov8n.pt        : YOLO model file (YOLOv8 nano)
- run_yolo.sh       : Launch script

Requirements
------------
- Python 3.8+
- ultralytics
- opencv-python
- numpy

Installation
------------
pip install ultralytics opencv-python numpy

Usage
-----
Run the launch script:
    ./run_yolo.sh

Or run directly with Python:
    python3 yolo_realtime.py

Hotkeys
-------
Q/ESC - Quit program
B     - Toggle bounding boxes
M     - Toggle masks (segmentation models only)
[     - Decrease confidence threshold
]     - Increase confidence threshold
,     - Decrease IOU threshold
.     - Increase IOU threshold
;     - Adjust skip frames
+/-   - Adjust text size
R     - Start/Stop recording
S     - Take screenshot
H     - Show/Hide help
F     - Toggle fullscreen
C     - Clear statistics
1-9   - Select/Cancel class filter
A     - Select all classes
N     - Clear class filter

Features
--------
- Real-time object detection from webcam
- Support for detection and segmentation models
- Adjustable confidence and IOU thresholds
- Class filtering
- Video recording
- Screenshot capture
- FPS monitoring
- Fullscreen mode

Model
-----
The default model is YOLOv8n (nano), which is optimized for speed.
You can replace it with other YOLOv8 models:
- yolov8n.pt (nano) - fastest, lowest accuracy
- yolov8s.pt (small) - balanced
- yolov8m.pt (medium) - higher accuracy
- yolov8l.pt (large) - highest accuracy, slower
- yolov8x.pt (xlarge) - maximum accuracy, slowest

Notes
-----
- The application will use CPU by default
- For GPU acceleration, ensure CUDA is properly installed
- Segmentation masks are only available with segmentation models
