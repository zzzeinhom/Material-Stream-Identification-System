# Material Stream Identification (MSI) System

A real-time computer vision application for classifying waste materials using classical machine learning techniques.

## Project Overview

The MSI system is a **local desktop application** that processes live video frames from a webcam to classify waste items into predefined material categories. The system uses feature-based machine learning (not deep learning) and demonstrates the complete ML pipeline from data processing to real-time deployment.

## Features

### Core Functionality
- **Real-time video capture** from webcam
- **Feature-based classification** using classical ML algorithms
- **Dual classifier system**: Support Vector Machine (SVM) and k-Nearest Neighbors (k-NN)
- **Unknown class rejection** for uncertain inputs
- **Interactive GUI** with live classification display

### Material Categories
- Glass
- Paper
- Cardboard
- Plastic
- Metal
- Trash
- Unknown (rejection class)

### Technical Features
- Modular architecture with clear separation of concerns
- Multi-threaded processing for smooth real-time performance
- Configurable parameters via YAML configuration
- Feature extraction using color, texture, and shape descriptors
- Confidence-based classification with threshold controls

## System Architecture

```
msi_system/
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml           # System configuration
├── src/
│   ├── __init__.py
│   ├── gui_application.py    # Main GUI application
│   ├── camera_handler.py     # Camera capture and frame processing
│   ├── feature_extractor.py  # Feature extraction from images
│   └── classifier.py         # ML classifiers with unknown handling
├── models/                   # Trained ML models (created at runtime)
├── data/                     # Data storage
├── utils/                    # Utility functions
└── logs/                     # Log files
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies**
   Create venv
   ```bash
   pip install -r requirements.txt
   ```

3. **Create sample models** (for testing without training data)
   ```bash
   python main.py --create-sample-data
   ```

## Usage
