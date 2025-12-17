"""
Camera Handler Module for Real-Time Video Capture
Manages camera operations and frame processing for the MSI system
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple
from queue import Queue


class CameraHandler:
    """Handles real-time camera capture and frame processing"""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)  # Buffer for latest frames
        self.capture_thread = None
        self.lock = threading.Lock()
        
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
    def initialize_camera(self) -> bool:
        """Initialize the camera and set properties"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera with index {self.camera_index}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Allow camera to warm up
            time.sleep(0.5)
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise RuntimeError("Cannot read frames from camera")
            
            print(f"Camera initialized successfully: {self.width}x{self.height} @ {self.fps} FPS")
            return True
            
        except Exception as e:
            print(f"Camera initialization failed: {str(e)}")
            self.release_camera()
            return False
    
    def start_capture(self) -> bool:
        """Start the camera capture thread"""
        if self.is_running:
            return True
            
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        print("Camera capture started")
        return True
    
    def stop_capture(self):
        """Stop the camera capture thread"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        self.release_camera()
        print("Camera capture stopped")
    
    def _capture_frames(self):
        """Internal method to capture frames in a separate thread"""
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Update FPS counter
                    self._update_fps()
                    
                    # Add frame to queue (overwrite oldest if full)
                    with self.lock:
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except:
                                pass
                        self.frame_queue.put(frame)
                        self.frame_count += 1
                else:
                    print("Warning: Failed to capture frame")
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Frame capture error: {str(e)}")
                time.sleep(0.01)
    
    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff >= 1.0:
            self.fps_counter = self.frame_count
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        with self.lock:
            try:
                # Get the latest frame without blocking
                frame = None
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                return frame
            except:
                return None
    
    def get_frame_with_retry(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get a frame with retry mechanism"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            frame = self.get_latest_frame()
            if frame is not None:
                return frame
            time.sleep(0.01)
        
        return None
    
    def get_current_fps(self) -> int:
        """Get current FPS"""
        return self.fps_counter
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry"""
        self.start_capture()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_capture()


class FrameProcessor:
    """Processes camera frames for classification"""
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess frame for feature extraction"""
        # Resize frame
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        return blurred
    
    @staticmethod
    def draw_roi(frame: np.ndarray, roi_size: Tuple[int, int] = (300, 300)) -> np.ndarray:
        """Draw Region of Interest (ROI) on frame"""
        frame_height, frame_width = frame.shape[:2]
        roi_width, roi_height = roi_size
        
        # Calculate ROI coordinates (center of frame)
        x1 = (frame_width - roi_width) // 2
        y1 = (frame_height - roi_height) // 2
        x2 = x1 + roi_width
        y2 = y1 + roi_height
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text
        cv2.putText(frame, "Place object here", (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame, (x1, y1, x2, y2)
    
    @staticmethod
    def extract_roi(frame: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract ROI from frame"""
        x1, y1, x2, y2 = roi_coords
        return frame[y1:y2, x1:x2]
    
    @staticmethod
    def add_info_overlay(frame: np.ndarray, classification: str, confidence: float, fps: int) -> np.ndarray:
        """Add classification info overlay to frame in top-right corner"""
        # Get frame dimensions
        height, width = frame.shape[:2]

        # Prepare text lines to display
        lines = [f"FPS: {fps}"]

        # Text rendering parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        line_spacing = 6
        pad_x, pad_y = 10, 8
        margin = 10

        # Compute text sizes and required box size
        max_w = 0
        total_h = 0
        sizes = []
        for text in lines:
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            sizes.append((tw, th))
            if tw > max_w:
                max_w = tw
            total_h += th + line_spacing

        total_h -= line_spacing  # remove extra spacing after last line

        box_w = max_w + 2 * pad_x
        box_h = total_h + 2 * pad_y

        # Position box at top-right, but keep inside frame
        x_end = width - margin
        x_start = max(x_end - box_w, margin)
        y_start = margin
        y_end = y_start + box_h

        # Create overlay and draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw each text line inside the box
        text_x = int(x_start + pad_x)
        y = int(y_start + pad_y + sizes[0][1])
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (text_x, y), font, font_scale, (200, 200, 200), font_thickness, cv2.LINE_AA)
            if i + 1 < len(sizes):
                y += sizes[i + 1][1] + line_spacing

        return frame
    
    @staticmethod
    def draw_prediction_box(frame: np.ndarray, classification: str, confidence: float, 
                          color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw prediction box around classified object"""
        frame_height, frame_width = frame.shape[:2]
        
        # Draw border
        border_thickness = 5
        cv2.rectangle(frame, (0, 0), (frame_width - 1, frame_height - 1), 
                     color, border_thickness)
        
        # Add classification label at bottom
        label_text = f"{classification} ({confidence:.1%})"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = frame_height - 20
        
        # Draw text background
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), color, -1)
        
        # Draw text
        cv2.putText(frame, label_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        return frame