#!/usr/bin/env python3
"""
Material Stream Identification (MSI) System
Main entry point for the real-time classification application

This is a local desktop application that uses computer vision and machine learning
to classify waste materials in real-time using a webcam.
"""

import sys
import os
import argparse
import warnings
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from gui_application import MSIApplication


def setup_environment():
    """Setup the application environment"""
    # Create necessary directories
    directories = ['models', 'data', 'logs', 'config']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/msi_system.log'),
            logging.StreamHandler()
        ]
    )
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'cv2', 'numpy', 'sklearn', 'skimage', 'PIL', 
        'yaml', 'tkinter', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Material Stream Identification System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                    # Run with default settings
  python main.py --config custom_config.yaml  # Use custom config file
  python main.py --camera 1         # Use camera index 1
  python main.py --no-gui           # Run in headless mode (future feature)
        '''
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--camera', '-cam',
        type=int,
        default=0,
        help='Camera index to use (default: 0)'
    )
    
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run in headless mode without GUI (future feature)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with verbose logging'
    )
    
    parser.add_argument(
        '--create-sample-data',
        action='store_true',
        help='Create sample models for testing'
    )
    
    return parser.parse_args()


def print_welcome_message():
    """Print welcome message and system information"""
    print("=" * 60)
    print("  Material Stream Identification (MSI) System")
    print("  Real-time Waste Classification Application")
    print("=" * 60)
    print()
    print("System Features:")
    print("  ✓ Real-time camera capture")
    print("  ✓ Feature-based machine learning")
    print("  ✓ SVM and k-NN classifiers")
    print("  ✓ Unknown class rejection")
    print("  ✓ Interactive GUI")
    print()
    print("Material Classes:")
    print("  • Glass")
    print("  • Paper")
    print("  • Cardboard")
    print("  • Plastic")
    print("  • Metal")
    print("  • Trash")
    print("  • Unknown")
    print()
    print("=" * 60)
    print()


def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Print welcome message
    print_welcome_message()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Handle special commands
    if args.create_sample_data:
        print("Creating sample models for testing...")
        from src.classifier import create_sample_models
        create_sample_models()
        print("Sample models created successfully!")
        return
    
    # Create application instance
    try:
        app = MSIApplication(config_path=args.config)
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"\nApplication error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\nApplication terminated successfully")


if __name__ == "__main__":
    main()