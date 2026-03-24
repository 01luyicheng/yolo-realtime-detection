#!/usr/bin/env python3
"""
YOLO Real-time Object Detection - Full GUI Version
All settings can be adjusted via GUI or hotkeys
"""
import os
import argparse

from ultralytics import YOLO
import cv2
import time
import sys
import numpy as np
import logging
from datetime import datetime
from collections import defaultdict

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 类别颜色映射
CLASS_COLORS = {
    'person': (0, 255, 0),
    'car': (255, 0, 0),
    'truck': (0, 0, 255),
    'bus': (255, 128, 0),
    'motorcycle': (255, 255, 0),
    'bicycle': (0, 255, 255),
    'dog': (255, 0, 255),
    'cat': (128, 0, 255),
    'bird': (0, 128, 255),
}

def get_class_color(name):
    """获取类别颜色"""
    if name in CLASS_COLORS:
        return CLASS_COLORS[name]
    hash_val = hash(name)
    return (hash_val & 0xFF, (hash_val >> 8) & 0xFF, (hash_val >> 16) & 0xFF)

class YOLODetector:
    """YOLO检测器类"""
    
    def __init__(self, model_path='yolov8n.pt'):
        """初始化检测器"""
        self.model_path = model_path
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.model = None
        self.supports_segmentation = False  # 是否支持分割
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            logger.info(f"Loading YOLO model: {self.model_path}")
            logger.info("Using device: CPU")
            self.model = YOLO(self.model_path)
            
            # Check if model supports segmentation
            self.supports_segmentation = hasattr(self.model, 'model') and hasattr(self.model.model, 'task') and self.model.model.task == 'segment'
            
            if self.supports_segmentation:
                logger.info("✅ YOLO model loaded (segmentation supported)")
            else:
                logger.info("✅ YOLO model loaded (detection only)")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, frame):
        """Perform object detection"""
        return self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )

class SettingsPanel:
    """Settings panel class"""
    
    def __init__(self, window_name, detector):
        self.window_name = window_name
        self.detector = detector
        self.class_names = list(detector.model.names.values())
        
        # Settings values
        self.conf_threshold = 25  # 0-100
        self.iou_threshold = 45   # 0-100
        self.text_size = 50       # 0-200
        self.skip_frames = 2      # 0-10
        self.camera_id = 0        # 0-3
        
        # Toggle states
        self.show_boxes = True
        self.show_masks = False
        self.show_stats = True
        self.show_help = True
        self.fullscreen = False
        self.recording = False
        
        # Class filtering
        self.filter_classes = set()
        self.selected_class = None
        
        # Performance stats
        self.fps_history = []
        self.max_fps_history = 30
        self.class_counts = defaultdict(int)
        
        # Use hotkey controls instead of trackbars for better compatibility
        self.detector = detector  # Keep detector reference
    
    def increase_conf(self):
        """Increase confidence threshold"""
        self.conf_threshold = min(self.conf_threshold + 5, 100)
        self.detector.conf_threshold = self.conf_threshold / 100.0
    
    def decrease_conf(self):
        """Decrease confidence threshold"""
        self.conf_threshold = max(self.conf_threshold - 5, 0)
        self.detector.conf_threshold = self.conf_threshold / 100.0
    
    def increase_iou(self):
        """Increase IOU threshold"""
        self.iou_threshold = min(self.iou_threshold + 5, 100)
        self.detector.iou_threshold = self.iou_threshold / 100.0
    
    def decrease_iou(self):
        """Decrease IOU threshold"""
        self.iou_threshold = max(self.iou_threshold - 5, 0)
        self.detector.iou_threshold = self.iou_threshold / 100.0
    
    def increase_text_size(self):
        """Increase text size"""
        self.text_size = min(self.text_size + 10, 200)
    
    def decrease_text_size(self):
        """Decrease text size"""
        self.text_size = max(self.text_size - 10, 0)
    
    def increase_skip(self):
        """Increase frame skip"""
        self.skip_frames = min(self.skip_frames + 1, 10)
    
    def decrease_skip(self):
        """Decrease frame skip"""
        self.skip_frames = max(self.skip_frames - 1, 0)
    
    def get_text_size(self):
        """Get actual text size"""
        return 0.3 + (self.text_size / 200.0) * 1.7
    
    def draw_control_panel(self):
        """Draw control panel"""
        control_window = "Control Panel"
        panel = np.zeros((600, 320, 3), dtype=np.uint8)
        
        # Background gradient
        for i in range(320):
            alpha = 1 - (i / 320) * 0.3
            panel[:, i] = (30 * alpha, 30 * alpha, 40 * alpha)
        
        y = 30
        
        # Title
        cv2.putText(panel, "YOLO Control Panel", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 50
        
        # Performance info
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        cv2.putText(panel, f"FPS: {avg_fps:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 30
        
        # Model type info
        model_type = "Segmentation" if self.detector.supports_segmentation else "Detection"
        model_color = (0, 255, 0) if self.detector.supports_segmentation else (255, 255, 0)
        cv2.putText(panel, f"Model: {model_type}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_color, 1)
        y += 30
        
        # Class statistics
        if self.class_counts:
            cv2.putText(panel, "Detection Stats:", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 30
            
            sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
            for cls_name, count in sorted_classes[:6]:
                color = get_class_color(cls_name)
                cv2.putText(panel, f"  {cls_name}: {count}", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y += 20
        
        y += 10
        
        # Display settings
        cv2.putText(panel, "Display Settings:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 30
        
        settings = [
            ("Boxes", self.show_boxes),
        ]
        
        # Show Masks option only for segmentation models
        if self.detector.supports_segmentation:
            settings.append(("Masks", self.show_masks))
        
        settings.extend([
            ("Stats", self.show_stats),
            ("Help", self.show_help),
            ("Fullscreen", self.fullscreen),
        ])
        
        for name, status in settings:
            status_str = "✓" if status else "✗"
            color = (0, 255, 0) if status else (100, 100, 100)
            cv2.putText(panel, f"{status_str} {name}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 20
        
        y += 10
        
        # Recording status
        if self.recording:
            cv2.rectangle(panel, (20, y), (300, y + 40), (0, 0, 255), -1)
            cv2.putText(panel, "● RECORDING", (80, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.rectangle(panel, (20, y), (300, y + 40), (100, 100, 100), -1)
            cv2.putText(panel, "○ Not Recording", (80, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y += 60
        
        # Current parameters
        cv2.putText(panel, "Current Params:", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 30
        
        params = [
            f"Conf: {self.conf_threshold}%",
            f"IOU: {self.iou_threshold}%",
            f"Text Size: {self.text_size}%",
            f"Skip Frames: {self.skip_frames}",
            f"Camera ID: {self.camera_id}",
        ]
        
        for param in params:
            cv2.putText(panel, param, (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y += 20
        
        y += 10
        
        # Class filter hint
        if self.filter_classes:
            cv2.putText(panel, f"Filter Classes: {len(self.filter_classes)}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            y += 20
            for cls_name in list(self.filter_classes)[:3]:
                cv2.putText(panel, f"  - {cls_name}", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
                y += 18
        
        # Hotkey hints
        y = 560
        shortcuts = [
            "Params:",
            "[/] Conf  [.] IOU  [;] Skip  +/- Text",
        ]
        
        # Show different shortcuts based on model type
        if self.detector.supports_segmentation:
            shortcuts.append("Display: B Box M Mask F Full H Help C Clear")
        else:
            shortcuts.append("Display: B Box F Full H Help C Clear")
        
        shortcuts.extend([
            "Actions: R Record S Screenshot 1-9 Class A All N None",
            "Q Quit"
        ])
        
        for shortcut in shortcuts:
            cv2.putText(panel, shortcut, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.28, (150, 150, 150), 1)
            y += 16
        
        cv2.imshow(control_window, panel)
    
    def draw_fps_bar(self, frame):
        """Draw FPS bar chart"""
        if not self.fps_history:
            return frame
        
        bar_height = 60
        bar_width = 200
        x = frame.shape[1] - bar_width - 10
        y = 10
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), 1)
        
        # Draw FPS curve
        if len(self.fps_history) > 1:
            points = []
            max_fps = max(self.fps_history) if self.fps_history else 1
            for i, fps in enumerate(self.fps_history):
                px = x + int((i / len(self.fps_history)) * bar_width)
                py = y + bar_height - int((fps / max_fps) * (bar_height - 10)) - 5
                points.append((px, py))
            
            if len(points) > 1:
                cv2.polylines(frame, [np.array(points)], False, (0, 255, 0), 2)
        
        # Current FPS
        current_fps = self.fps_history[-1] if self.fps_history else 0
        cv2.putText(frame, f"{current_fps:.1f} FPS", (x + 5, y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def draw_help_overlay(self, frame):
        """Draw help overlay"""
        if not self.show_help:
            return frame
        
        help_text = [
            "Hotkeys:",
            "Q/ESC - Quit",
            "B - Boxes",
        ]
        
        # Show Masks hotkey only for segmentation models
        if self.detector.supports_segmentation:
            help_text.append("M - Masks")
        
        help_text.extend([
            "[/] - Confidence",
            "[,]. - IOU",
            "; - Skip Frames",
            "+/- - Text Size",
            "R - Record",
            "S - Screenshot",
            "H - Help",
            "F - Fullscreen",
            "C - Clear Stats",
            "1-9 - Select Class",
            "A - Select All",
            "N - Clear Filter",
        ])
        
        overlay = frame.copy()
        y = 30
        for text in help_text:
            cv2.putText(overlay, text, (frame.shape[1] - 180, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            y += 18
        
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

def main():
    # Initialize detector
    try:
        detector = YOLODetector()
    except Exception as e:
        logger.error(f"Detector initialization failed: {e}")
        input("Press Enter to exit...")
        return

    # Open camera
    logger.info("Opening camera...")
    camera_id = 0
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        logger.error("❌ Failed to open camera")
        input("Press Enter to exit...")
        return
    
    logger.info("✅ Camera opened")

    # Set camera parameters
    PROCESS_WIDTH = 640
    PROCESS_HEIGHT = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    # Get actual parameters
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")

    # Create settings panel
    settings = SettingsPanel("YOLO Real-time Detection", detector)

    # Create main window
    window_name = "YOLO Real-time Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, PROCESS_WIDTH, PROCESS_HEIGHT)

    logger.info("\nReal-time YOLO detection started!")
    logger.info("All settings can be adjusted via hotkeys\n")

    frame_count = 0
    processed_count = 0
    total_time = 0
    last_result = None
    last_print_time = time.time()
    global_start_time = time.time()
    last_boxes = []
    
    video_writer = None
    screenshot_count = 0

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                break

            frame_count += 1

            # Print status every 5 seconds
            if time.time() - last_print_time > 5:
                avg_time = total_time / processed_count if processed_count > 0 else 0
                fps = 1 / avg_time if avg_time > 0 else 0
                logger.info(f"[Status] Processed {processed_count} frames, FPS: {fps:.1f}, Total: {frame_count}")
                last_print_time = time.time()

            # Frame skipping
            if frame_count % (settings.skip_frames + 1) == 0:
                # YOLO detection
                start_time = time.time()
                results = detector.detect(frame)
                inference_time = time.time() - start_time

                # Update FPS history
                fps = 1 / inference_time if inference_time > 0 else 0
                settings.fps_history.append(fps)
                if len(settings.fps_history) > settings.max_fps_history:
                    settings.fps_history.pop(0)

                # Draw results
                result = results[0]
                last_result = result

                # Update class statistics
                settings.class_counts.clear()
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    name = detector.model.names[cls_id]
                    settings.class_counts[name] += 1

                annotated_frame = frame.copy()
                
                # Draw masks (only for segmentation models)
                if settings.show_masks and detector.supports_segmentation and hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    for mask in masks:
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                        mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(annotated_frame, contours, -1, (0, 255, 255), 2)

                # Draw bounding boxes
                if settings.show_boxes:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        name = detector.model.names[cls_id]
                        
                        # Class filtering
                        if settings.filter_classes and name not in settings.filter_classes:
                            continue
                        
                        # Selected class filtering
                        if settings.selected_class is not None and cls_id != settings.selected_class:
                            continue
                            
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        color = get_class_color(name)
                        
                        # Boundary check: ensure coordinates are within image bounds
                        x1 = max(0, min(x1, frame.shape[1] - 1))
                        y1 = max(0, min(y1, frame.shape[0] - 1))
                        x2 = max(0, min(x2, frame.shape[1] - 1))
                        y2 = max(0, min(y2, frame.shape[0] - 1))
                        
                        # Draw box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        text_size = settings.get_text_size()
                        label = f"{name} {conf:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)
                        
                        # Label position boundary check
                        label_y = y1 - 10
                        if label_y - label_size[1] < 0:
                            label_y = y2 + label_size[1] + 10
                        
                        cv2.rectangle(annotated_frame, (x1, label_y - label_size[1]),
                                    (x1 + label_size[0], label_y), color, -1)
                        cv2.putText(annotated_frame, label, (x1, label_y - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 2)

                # Save detection boxes
                last_boxes = []
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    name = detector.model.names[cls_id]
                    if settings.filter_classes and name not in settings.filter_classes:
                        continue
                    last_boxes.append({
                        'cls': cls_id,
                        'conf': float(box.conf[0]),
                        'xyxy': box.xyxy[0].cpu().numpy().astype(int),
                        'name': name
                    })

                processed_count += 1
                total_time += inference_time
            else:
                # Use previous frame results
                annotated_frame = frame.copy()
                if last_boxes:
                    text_size = settings.get_text_size()
                    for box_info in last_boxes:
                        # Selected class filtering
                        if settings.selected_class is not None and box_info['cls'] != settings.selected_class:
                            continue
                            
                        x1, y1, x2, y2 = box_info['xyxy']
                        
                        # Boundary check: ensure coordinates are within image bounds
                        x1 = max(0, min(x1, frame.shape[1] - 1))
                        y1 = max(0, min(y1, frame.shape[0] - 1))
                        x2 = max(0, min(x2, frame.shape[1] - 1))
                        y2 = max(0, min(y2, frame.shape[0] - 1))
                        
                        if settings.show_boxes:
                            color = get_class_color(box_info['name'])
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{box_info['name']} {box_info['conf']:.2f}"
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)
                            
                            # Label position boundary check
                            label_y = y1 - 10
                            if label_y - label_size[1] < 0:
                                label_y = y2 + label_size[1] + 10
                            
                            cv2.rectangle(annotated_frame, (x1, label_y - label_size[1]),
                                        (x1 + label_size[0], label_y), color, -1)
                            cv2.putText(annotated_frame, label, (x1, label_y - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 2)

            # 添加FPS进度条
            if settings.show_stats:
                annotated_frame = settings.draw_fps_bar(annotated_frame)
            
            # 添加帮助信息
            annotated_frame = settings.draw_help_overlay(annotated_frame)

            # 显示
            cv2.imshow(window_name, annotated_frame)
            
            # 绘制控制面板
            settings.draw_control_panel()

            # 录制
            if settings.recording and video_writer:
                video_writer.write(annotated_frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Quit
                break
            elif key == ord('b'):  # Toggle boxes
                settings.show_boxes = not settings.show_boxes
                logger.info(f"[Settings] Boxes: {'ON' if settings.show_boxes else 'OFF'}")
            elif key == ord('m'):  # Toggle masks
                if detector.supports_segmentation:
                    settings.show_masks = not settings.show_masks
                    logger.info(f"[Settings] Masks: {'ON' if settings.show_masks else 'OFF'}")
                else:
                    logger.info("[Settings] Current model does not support segmentation")
            elif key == ord('['):  # Decrease confidence
                settings.decrease_conf()
                logger.info(f"[Settings] Confidence: {settings.conf_threshold}%")
            elif key == ord(']'):  # Increase confidence
                settings.increase_conf()
                logger.info(f"[Settings] Confidence: {settings.conf_threshold}%")
            elif key == ord(','):  # Decrease IOU
                settings.decrease_iou()
                logger.info(f"[Settings] IOU: {settings.iou_threshold}%")
            elif key == ord('.'):  # Increase IOU
                settings.increase_iou()
                logger.info(f"[Settings] IOU: {settings.iou_threshold}%")
            elif key == ord(';'):  # Toggle frame skip
                if settings.skip_frames > 0:
                    settings.decrease_skip()
                else:
                    settings.increase_skip()
                logger.info(f"[Settings] Skip frames: {settings.skip_frames}")
            elif key == ord('+') or key == ord('='):  # Increase text size
                settings.increase_text_size()
                logger.info(f"[Settings] Text size: {settings.text_size}%")
            elif key == ord('-'):  # Decrease text size
                settings.decrease_text_size()
                logger.info(f"[Settings] Text size: {settings.text_size}%")
            elif key == ord('r'):  # Toggle recording
                settings.recording = not settings.recording
                if settings.recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = f"yolo_detection_{timestamp}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(output_path, fourcc, actual_fps, (actual_width, actual_height))
                    logger.info(f"[Record] Started: {output_path}")
                else:
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    logger.info("[Record] Stopped")
            elif key == ord('s'):  # Screenshot
                screenshot_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"yolo_screenshot_{timestamp}_{screenshot_count}.jpg"
                cv2.imwrite(screenshot_path, annotated_frame)
                logger.info(f"[Screenshot] Saved: {screenshot_path}")
            elif key == ord('h'):  # Toggle help
                settings.show_help = not settings.show_help
                logger.info(f"[Settings] Help: {'SHOW' if settings.show_help else 'HIDE'}")
            elif key == ord('f'):  # Toggle fullscreen
                settings.fullscreen = not settings.fullscreen
                if settings.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                logger.info(f"[Settings] Fullscreen: {'ON' if settings.fullscreen else 'OFF'}")
            elif key == ord('c'):  # Clear stats
                settings.class_counts.clear()
                settings.fps_history.clear()
                logger.info("[Settings] Statistics cleared")
            elif ord('1') <= key <= ord('9'):  # Select class
                class_idx = key - ord('1')
                if class_idx < len(detector.model.names):
                    cls_name = detector.model.names[class_idx]
                    if cls_name in settings.filter_classes:
                        settings.filter_classes.remove(cls_name)
                        logger.info(f"[Settings] Removed filter: {cls_name}")
                    else:
                        settings.filter_classes.add(cls_name)
                        logger.info(f"[Settings] Added filter: {cls_name}")
            elif key == ord('a'):  # Select all classes
                settings.filter_classes = set(detector.model.names.values())
                logger.info(f"[Settings] All classes selected")
            elif key == ord('n'):  # Clear filter
                settings.filter_classes.clear()
                logger.info("[Settings] Class filter cleared")

    except KeyboardInterrupt:
        logger.info("\n\nDetection stopped")
    except Exception as e:
        logger.error(f"Runtime error: {e}", exc_info=True)
    finally:
        # Release resources
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        
        # Statistics
        avg_time = total_time / processed_count if processed_count > 0 else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        logger.info(f"\nStatistics:")
        logger.info(f"  Total frames: {frame_count}")
        logger.info(f"  Processed frames: {processed_count}")
        logger.info(f"  Avg inference time: {avg_time*1000:.1f}ms")
        logger.info(f"  Detection FPS: {fps:.1f}")
        logger.info(f"  Runtime: {time.time() - global_start_time:.1f}s")
        logger.info(f"  Screenshots: {screenshot_count}")
        
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()