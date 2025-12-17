"""
Shared Model Definition Module
Re-exports SVMClassifier and KNNClassifier for pickle compatibility.
This centralized location ensures both training and inference use the same class definitions.
"""

# Import from root-level classifier files
import sys
from pathlib import Path

# Add parent directory to path to import from root
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Import the classifier classes from their source files
from SVM_classifier import SVMClassifier
from KNN_classifier import KNNClassifier

# Re-export for use in other modules
__all__ = ['SVMClassifier', 'KNNClassifier']
