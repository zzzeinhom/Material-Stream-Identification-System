"""
GUI Application for Material Stream Identification System
Modified to use CNN-based feature extraction
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
from typing import Optional
from PIL import Image, ImageTk
import os

from camera_handler import CameraHandler, FrameProcessor
from cnn_feature_extraction import CNNFeatureExtractor  # Import CNN feature extractor
from classifier import UnifiedMaterialClassifier
import yaml


class MSIApplication:
    """Main GUI application for Material Stream Identification"""
    
    # Feature dimension will be set based on CNN model
    EXPECTED_FEATURES = None
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.camera_handler = None
        self.cnn_extractor = None  # CNN feature extractor
        self.classifier = None
        self.frame_processor = FrameProcessor()
        
        # Application state
        self.is_running = False
        self.current_frame = None
        self.current_prediction = "Unknown"
        self.current_confidence = 0.0
        self.current_fps = 0
        self.individual_results = {}
        
        # GUI elements
        self.root = None
        self.video_label = None
        self.info_frame = None
        self.status_label = None
        self.confidence_bar = None
        
        # Threading
        self.update_thread = None
        self.stop_event = threading.Event()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return {
                'camera': {'index': 0, 'width': 640, 'height': 480, 'fps': 30},
                'classification': {'confidence_threshold': 0.6, 'unknown_threshold': 0.4, 'display_confidence': True},
                'models': {
                    'svm_model': "models/svm_classifier.pkl",
                    'knn_model': "models/knn_classifier.pkl"
                },
                'cnn': {
                    'model_name': 'ResNet50',  # Options: VGG16, ResNet50, MobileNetV2
                    'input_size': [224, 224]
                },
                'gui': {'window_title': "Material Stream Identification System", 'update_interval': 100}
            }
    
    def initialize_system(self) -> bool:
        """Initialize all system components"""
        try:
            # Initialize camera
            self.camera_handler = CameraHandler(
                camera_index=self.config['camera']['index'],
                width=self.config['camera']['width'],
                height=self.config['camera']['height'],
                fps=self.config['camera']['fps']
            )
            
            # Initialize CNN feature extractor
            cnn_config = self.config.get('cnn', {})
            model_name = cnn_config.get('model_name', 'ResNet50')
            input_size = tuple(cnn_config.get('input_size', [224, 224]))
            
            print(f"Initializing CNN feature extractor: {model_name}")
            self.cnn_extractor = CNNFeatureExtractor(
                model_name=model_name,
                input_size=input_size
            )
            
            # Set expected features based on CNN model
            self.EXPECTED_FEATURES = self.cnn_extractor.feature_dim
            print(f"Expected feature dimension: {self.EXPECTED_FEATURES}")
            
            # Initialize classifier
            self.classifier = UnifiedMaterialClassifier()
            
            models_loaded = self.classifier.load_models(
                svm_model_path=self.config['models']['svm_model'],
                knn_model_path=self.config['models']['knn_model']
            )
            
            if not models_loaded:
                print("Failed to load classifier models")
                return False
            
            print("System initialized successfully")
            return True
            
        except Exception as e:
            print(f"System initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_gui(self):
        """Create the main GUI window with modern styling"""
        self.root = tk.Tk()
        self.root.title(self.config['gui']['window_title'])
        self.root.geometry("1200x700")
        
        # Configure modern style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Define color scheme
        bg_color = '#1e1e1e'
        sidebar_color = '#252526'
        accent_color = '#007acc'
        text_color = '#cccccc'
        
        self.root.configure(bg=bg_color)
        
        # Configure styles
        style.configure('Main.TFrame', background=bg_color)
        style.configure('Sidebar.TFrame', background=sidebar_color)
        style.configure('Card.TLabelframe', background=sidebar_color, foreground=text_color, 
                       borderwidth=0, relief='flat')
        style.configure('Card.TLabelframe.Label', background=sidebar_color, foreground=text_color,
                       font=('Segoe UI', 10, 'bold'))
        style.configure('TLabel', background=sidebar_color, foreground=text_color, 
                       font=('Segoe UI', 9))
        style.configure('Header.TLabel', background=sidebar_color, foreground=text_color,
                       font=('Segoe UI', 11, 'bold'))
        style.configure('Value.TLabel', background=sidebar_color, foreground='#4ec9b0',
                       font=('Segoe UI', 10, 'bold'))
        style.configure('Accent.TButton', background=accent_color, foreground='white',
                       font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.map('Accent.TButton', background=[('active', '#005a9e')])
        
        # Main container
        main_container = ttk.Frame(self.root, style='Main.TFrame')
        main_container.pack(fill='both', expand=True)
        
        # Left side - Video feed
        video_container = ttk.Frame(main_container, style='Main.TFrame')
        video_container.pack(side='left', fill='both', expand=True)
        
        # Video display with border
        video_frame = tk.Frame(video_container, bg="#000000", highlightbackground=accent_color,
                              highlightthickness=2)
        video_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.video_label = tk.Label(video_frame, bg='#000000')
        self.video_label.pack(fill='both', expand=True)
        
        # Right side - Control panel
        self._create_right_panel(main_container, sidebar_color, accent_color, text_color)
        
        # Bottom status bar
        self._create_modern_status_bar(main_container, bg_color, text_color)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _create_right_panel(self, parent, sidebar_color, accent_color, text_color):
        """Create modern right panel with controls and info"""
        right_panel = ttk.Frame(parent, style='Sidebar.TFrame', width=350)
        right_panel.pack(side='right', fill='both', padx=(0, 10), pady=10)
        right_panel.pack_propagate(False)
        
        # Header
        header_frame = ttk.Frame(right_panel, style='Sidebar.TFrame')
        header_frame.pack(fill='x', padx=15, pady=(15, 10))
        
        title_label = ttk.Label(header_frame, text="CLASSIFICATION", 
                               style='Header.TLabel', font=('Segoe UI', 12, 'bold'))
        title_label.pack()
        
        # CNN Model info
        cnn_info = ttk.Label(header_frame, 
                            text=f"CNN: {self.config.get('cnn', {}).get('model_name', 'ResNet50')}", 
                            font=('Segoe UI', 8), foreground='#808080')
        cnn_info.pack()
        
        # Control buttons section
        control_frame = ttk.Frame(right_panel, style='Sidebar.TFrame')
        control_frame.pack(fill='x', padx=15, pady=10)
        
        self.start_button = tk.Button(control_frame, text="START CLASSIFICATION",
                                     command=self.toggle_classification,
                                     bg=accent_color, fg='white', font=('Segoe UI', 10, 'bold'),
                                     relief='flat', cursor='hand2', height=2)
        self.start_button.pack(fill='x', pady=(0, 10))
        self.start_button.bind('<Enter>', lambda e: self.start_button.config(bg='#005a9e'))
        self.start_button.bind('<Leave>', lambda e: self.start_button.config(bg=accent_color))
        
        # Classifier selection
        classifier_frame = ttk.Frame(control_frame, style='Sidebar.TFrame')
        classifier_frame.pack(fill='x', pady=5)
        
        ttk.Label(classifier_frame, text="Classifier Method:", 
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 5))
        
        self.classifier_var = tk.StringVar(value='ensemble')
        classifier_combo = ttk.Combobox(classifier_frame, textvariable=self.classifier_var,
                                       values=['ensemble', 'svm', 'knn'], state='readonly',
                                       font=('Segoe UI', 9))
        classifier_combo.pack(fill='x')
        
        # Separator
        separator = ttk.Separator(right_panel, orient='horizontal')
        separator.pack(fill='x', padx=15, pady=15)
        
        # Results section
        results_frame = ttk.Frame(right_panel, style='Sidebar.TFrame')
        results_frame.pack(fill='both', expand=True, padx=15)
        
        # Current material
        ttk.Label(results_frame, text="Detected Material", 
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 5))
        
        self.prediction_label = ttk.Label(results_frame, text="Unknown",
                                         style='Value.TLabel',
                                         font=('Segoe UI', 16, 'bold'))
        self.prediction_label.pack(anchor='w', pady=(0, 15))
        
        # Confidence
        confidence_container = ttk.Frame(results_frame, style='Sidebar.TFrame')
        confidence_container.pack(fill='x', pady=(0, 5))
        
        ttk.Label(confidence_container, text="Confidence", 
                 font=('Segoe UI', 9)).pack(side='left')
        
        self.confidence_label = ttk.Label(confidence_container, text="0.0%",
                                         font=('Segoe UI', 9, 'bold'),
                                         foreground='#4ec9b0')
        self.confidence_label.pack(side='right')
        
        # Progress bar
        self.confidence_bar = ttk.Progressbar(results_frame, length=300, mode='determinate')
        self.confidence_bar.pack(fill='x', pady=(0, 20))
        
        # Individual classifier results
        classifiers_frame = ttk.Frame(results_frame, style='Sidebar.TFrame')
        classifiers_frame.pack(fill='x', pady=(0, 15))
        
        ttk.Label(classifiers_frame, text="Model Predictions", 
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(0, 8))
        
        # SVM result
        svm_container = ttk.Frame(classifiers_frame, style='Sidebar.TFrame')
        svm_container.pack(fill='x', pady=3)
        ttk.Label(svm_container, text="SVM:", font=('Segoe UI', 8)).pack(side='left')
        self.svm_label = ttk.Label(svm_container, text="-", font=('Segoe UI', 8, 'bold'),
                                   foreground='#ce9178')
        self.svm_label.pack(side='right')
        
        # KNN result
        knn_container = ttk.Frame(classifiers_frame, style='Sidebar.TFrame')
        knn_container.pack(fill='x', pady=3)
        ttk.Label(knn_container, text="KNN:", font=('Segoe UI', 8)).pack(side='left')
        self.knn_label = ttk.Label(knn_container, text="-", font=('Segoe UI', 8, 'bold'),
                                   foreground='#ce9178')
        self.knn_label.pack(side='right')
        
        # Material info
        ttk.Label(results_frame, text="Material Information", 
                 font=('Segoe UI', 9)).pack(anchor='w', pady=(15, 5))
        
        info_container = tk.Frame(results_frame, bg='#1e1e1e', highlightbackground='#3c3c3c',
                                 highlightthickness=1)
        info_container.pack(fill='both', expand=True)
        
        self.info_text = tk.Text(info_container, height=8, wrap=tk.WORD,
                                bg='#1e1e1e', fg=text_color, font=('Segoe UI', 8),
                                relief='flat', padx=10, pady=10)
        self.info_text.pack(fill='both', expand=True)
        self.info_text.config(state=tk.DISABLED)
    
    def _create_modern_status_bar(self, parent, bg_color, text_color):
        """Create modern status bar"""
        status_frame = tk.Frame(parent, bg='#252526', height=30)
        status_frame.pack(side='bottom', fill='x')
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="● Ready", 
                                     bg='#252526', fg='#4ec9b0',
                                     font=('Segoe UI', 9), anchor='w')
        self.status_label.pack(side='left', padx=15)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0", 
                                 bg='#252526', fg=text_color,
                                 font=('Segoe UI', 9))
        self.fps_label.pack(side='right', padx=15)
    
    def toggle_classification(self):
        """Start or stop classification"""
        if not self.is_running:
            self.start_classification()
        else:
            self.stop_classification()
    
    def start_classification(self):
        """Start the classification process"""
        try:
            if not self.camera_handler.start_capture():
                messagebox.showerror("Error", "Failed to start camera")
                return
            
            self.is_running = True
            self.start_button.config(text="STOP CLASSIFICATION", bg='#c82333')
            self.start_button.bind('<Enter>', lambda e: self.start_button.config(bg='#a71d2a'))
            self.start_button.bind('<Leave>', lambda e: self.start_button.config(bg='#c82333'))
            self.status_label.config(text="● Running", fg='#4ec9b0')
            
            self.stop_event.clear()
            self.update_thread = threading.Thread(target=self.classification_loop, daemon=True)
            self.update_thread.start()
            
            print("Classification started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start classification: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def stop_classification(self):
        """Stop the classification process"""
        self.is_running = False
        self.stop_event.set()
        
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        
        if self.camera_handler:
            self.camera_handler.stop_capture()
        
        self.start_button.config(text="START CLASSIFICATION", bg='#007acc')
        self.start_button.bind('<Enter>', lambda e: self.start_button.config(bg='#005a9e'))
        self.start_button.bind('<Leave>', lambda e: self.start_button.config(bg='#007acc'))
        self.status_label.config(text="● Stopped", fg='#f48771')
        
        print("Classification stopped")
    
    def classification_loop(self):
        """Main classification loop running in separate thread"""
        while self.is_running and not self.stop_event.is_set():
            try:
                frame = self.camera_handler.get_latest_frame()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                processed_frame, classification, confidence, individual_results = self.process_frame(frame)
                
                self.current_frame = processed_frame
                self.current_prediction = classification
                self.current_confidence = confidence
                self.individual_results = individual_results
                self.current_fps = self.camera_handler.get_current_fps()
                
                self.root.after(0, self.update_gui)
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"Classification error: {str(e)}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame using CNN feature extraction"""
        try:
            # Draw ROI on frame
            frame_with_roi, roi_coords = self.frame_processor.draw_roi(frame.copy())
            
            # Extract ROI
            roi_frame = self.frame_processor.extract_roi(frame, roi_coords)
            
            # Preprocess for display (optional, for visualization)
            processed_roi = self.frame_processor.preprocess_frame(roi_frame)
            
            # Extract features using CNN
            features = self.cnn_extractor.extract_features_from_image(roi_frame)
            
            # Verify feature dimensions
            if features.shape[0] != self.EXPECTED_FEATURES:
                raise ValueError(
                    f"Feature dimension mismatch! Expected {self.EXPECTED_FEATURES}, "
                    f"got {features.shape[0]}. Check CNN model configuration."
                )
            
            # Classify using selected method
            classifier_method = self.classifier_var.get()
            pred_class, confidence, individual_results = self.classifier.predict(
                features, method=classifier_method
            )
            
            # Add info overlay to frame
            frame_with_info = self.frame_processor.add_info_overlay(
                frame_with_roi, pred_class, confidence, self.current_fps
            )
            
            # Draw prediction box if confidence is high enough
            if confidence > self.config['classification']['confidence_threshold']:
                class_info = self.classifier.get_class_info(pred_class)
                frame_with_info = self.frame_processor.draw_prediction_box(
                    frame_with_info, pred_class, confidence, class_info['color']
                )
            
            return frame_with_info, pred_class, confidence, individual_results
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return frame, "Unknown", 0.0, {}
    
    def update_gui(self):
        """Update GUI elements with current data"""
        try:
            if self.current_frame is not None:
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                display_width = self.video_label.winfo_width()
                display_height = self.video_label.winfo_height()
                
                if display_width > 1 and display_height > 1:
                    image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.video_label.config(image=photo)
                self.video_label.image = photo
            
            self.prediction_label.config(text=self.current_prediction)
            
            confidence_percent = self.current_confidence * 100
            self.confidence_label.config(text=f"{confidence_percent:.1f}%")
            self.confidence_bar['value'] = confidence_percent
            
            if self.individual_results:
                if 'svm' in self.individual_results:
                    svm_class = self.individual_results['svm']['class']
                    svm_conf = self.individual_results['svm']['confidence'] * 100
                    self.svm_label.config(text=f"{svm_class} ({svm_conf:.1f}%)")
                
                if 'knn' in self.individual_results:
                    knn_class = self.individual_results['knn']['class']
                    knn_conf = self.individual_results['knn']['confidence'] * 100
                    self.knn_label.config(text=f"{knn_class} ({knn_conf:.1f}%)")
            
            class_info = self.classifier.get_class_info(self.current_prediction)
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"{class_info['description']}\n\n")
            self.info_text.insert(tk.END, f"Recycling: {class_info['recycling_info']}")
            self.info_text.config(state=tk.DISABLED)
            
            self.fps_label.config(text=f"FPS: {self.current_fps:.1f}")
            
        except Exception as e:
            print(f"GUI update error: {str(e)}")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_running:
            self.stop_classification()
        
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        """Run the application"""
        print("Initializing system...")
        if not self.initialize_system():
            messagebox.showerror("Error", "Failed to initialize system. Please check if model files exist.")
            return
        
        print("Creating GUI...")
        self.create_gui()
        
        try:
            print("Starting application...")
            self.root.mainloop()
        except KeyboardInterrupt:
            print("Application interrupted by user")
        finally:
            self.on_closing()


def main():
    """Main entry point"""
    print("=" * 60)
    print("Material Stream Identification System")
    print("Using CNN-based Feature Extraction")
    print("=" * 60)
    app = MSIApplication()
    app.run()


if __name__ == "__main__":
    main()