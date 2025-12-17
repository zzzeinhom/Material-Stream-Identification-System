"""
Unified Classifier that uses standalone SVM and KNN implementations
Compatible with gui_application.py
"""

import numpy as np
import joblib
from typing import Tuple, Optional, Dict
import pandas as pd
import sys

# Import classifier definitions from shared module
from model_definition import SVMClassifier, KNNClassifier



class UnifiedMaterialClassifier:
    """Combined classifier using standalone implementations with unknown class rejection"""
    
    def __init__(self, confidence_threshold: float = 0.6, unknown_threshold: float = 0.4):
        self.svm_classifier = None
        self.knn_classifier = None
        self.class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
        
        self.confidence_threshold = confidence_threshold
        self.unknown_threshold = unknown_threshold
        self.class_map = {
            'glass': 'Glass',
            'paper': 'Paper', 
            'cardboard': 'Cardboard',
            'plastic': 'Plastic',
            'metal': 'Metal',
            'trash': 'Trash',
            'unknown': 'Unknown'
        }
        
    def load_models(self, svm_model_path: str, knn_model_path: str, 
                   svm_scaler_path: str = None, knn_scaler_path: str = None) -> bool:
        """Load trained standalone models"""
        try:
            # Ensure class definitions are available in __main__ for pickle unpickling
            # by importing them into the __main__ namespace
            sys.modules['__main__'].SVMClassifier = SVMClassifier
            sys.modules['__main__'].KNNClassifier = KNNClassifier
            
            # Load SVM classifier (entire SVMClassifier object)
            self.svm_classifier = joblib.load(svm_model_path)
            print(f"Loaded SVM model from {svm_model_path}")
            
            # Load KNN classifier (entire KNNClassifier object)
            self.knn_classifier = joblib.load(knn_model_path)
            print(f"Loaded KNN model from {knn_model_path}")
            
            # Verify models are trained
            if not self.svm_classifier.is_trained:
                print("Warning: SVM model is not trained")
                return False
            
            if not self.knn_classifier.is_trained:
                print("Warning: KNN model is not trained")
                return False
            
            # Set class names from one of the classifiers
            if hasattr(self.svm_classifier, 'class_names') and self.svm_classifier.class_names:
                self.class_names = self.svm_classifier.class_names
            elif hasattr(self.knn_classifier, 'class_names') and self.knn_classifier.class_names:
                self.class_names = self.knn_classifier.class_names
                
            print(f"Class names: {self.class_names}")
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_svm(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict using standalone SVM classifier"""
        if self.svm_classifier is None or not self.svm_classifier.is_trained:
            return "Unknown", 0.0
        
        try:
            # Ensure features are 1D array
            if len(features.shape) > 1:
                features = features.flatten()
            
            # Reshape to 2D for sklearn compatibility
            features_2d = features.reshape(1, -1)
            
            # Apply scaling
            features_scaled = self.svm_classifier.scaler.transform(features_2d)
            
            # Apply PCA transformation
            features_pca = self.svm_classifier.pca.transform(features_scaled)
            
            # Get prediction
            prediction = self.svm_classifier.svm_model.predict(features_pca)[0]
            probabilities = self.svm_classifier.svm_model.predict_proba(features_pca)[0]
            
            # Get class name and confidence
            class_name = self.class_names[prediction] if self.class_names else str(prediction)
            confidence = float(probabilities[prediction])
            
            # Map to standard class names
            mapped_class = self.class_map.get(class_name.lower(), class_name)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return "Unknown", confidence
            
            return mapped_class, confidence
            
        except Exception as e:
            print(f"SVM prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Unknown", 0.0
    
    def predict_knn(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict using standalone KNN classifier"""
        if self.knn_classifier is None or not self.knn_classifier.is_trained:
            return "Unknown", 0.0
        
        try:
            # Ensure features are 1D array
            if len(features.shape) > 1:
                features = features.flatten()
            
            # Use the standalone KNN's predict_features method
            # This handles scaling and NCA transformation internally
            class_name, confidence = self.knn_classifier.predict_features(features)
            
            # Map to standard class names
            mapped_class = self.class_map.get(class_name.lower(), class_name)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                return "Unknown", confidence
            
            return mapped_class, float(confidence)
            
        except Exception as e:
            print(f"KNN prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Unknown", 0.0
    
    def predict_ensemble(self, features: np.ndarray) -> Tuple[str, float, dict]:
        """Predict using ensemble of both classifiers"""
        # Get predictions from both classifiers
        svm_class, svm_conf = self.predict_svm(features)
        knn_class, knn_conf = self.predict_knn(features)
        
        # Store individual results
        individual_results = {
            'svm': {'class': svm_class, 'confidence': float(svm_conf)},
            'knn': {'class': knn_class, 'confidence': float(knn_conf)}
        }
        
        print(f"SVM: {svm_class} ({svm_conf:.3f}), KNN: {knn_class} ({knn_conf:.3f})")
        
        # If both classifiers agree and have high confidence
        if svm_class == knn_class and svm_conf > self.confidence_threshold and knn_conf > self.confidence_threshold:
            # Average the confidences
            final_confidence = (svm_conf + knn_conf) / 2
            return svm_class, final_confidence, individual_results
        
        # If classifiers disagree, use the one with higher confidence
        if svm_conf > knn_conf:
            primary_class = svm_class
            primary_conf = svm_conf
        else:
            primary_class = knn_class
            primary_conf = knn_conf
        
        # Check if primary classifier has sufficient confidence
        if primary_conf >= self.confidence_threshold:
            return primary_class, primary_conf, individual_results
        
        # If both have low confidence, return Unknown
        return "Unknown", min(svm_conf, knn_conf), individual_results
    
    def predict_with_unknown_handling(self, features: np.ndarray, method: str) -> Tuple[str, float, dict]:
        """Main prediction method compatible with gui_application.py"""
        try:
            # Ensure features are numpy array
            if isinstance(features, pd.DataFrame):
                features = features.values
            
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            
            # Get prediction based on method
            if method == 'svm':
                pred_class, confidence = self.predict_svm(features)
                individual_results = {'svm': {'class': pred_class, 'confidence': float(confidence)}}
            elif method == 'knn':
                pred_class, confidence = self.predict_knn(features)
                individual_results = {'knn': {'class': pred_class, 'confidence': float(confidence)}}
            elif method == 'ensemble':
                pred_class, confidence, individual_results = self.predict_ensemble(features)
            else:
                raise ValueError(f"Unknown prediction method: {method}")
            
            # Additional unknown class handling
            if confidence < self.unknown_threshold:
                pred_class = "Unknown"
            
            return pred_class, float(confidence), individual_results
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Unknown", 0.0, {}
    
    def get_class_info(self, class_name: str) -> dict:
        """Get information about a material class"""
        class_info = {
            'Glass': {
                'description': 'Glass bottles, jars, and containers',
                'recycling_info': 'Rinse and remove lids. Most glass is recyclable.',
                'color': (0, 255, 0)  # Green
            },
            'Paper': {
                'description': 'Paper sheets, newspapers, magazines',
                'recycling_info': 'Keep dry and clean. Avoid soiled paper.',
                'color': (255, 255, 0)  # Yellow
            },
            'Cardboard': {
                'description': 'Cardboard boxes and packaging',
                'recycling_info': 'Flatten boxes. Remove tape and labels.',
                'color': (255, 165, 0)  # Orange
            },
            'Plastic': {
                'description': 'Plastic bottles, containers, packaging',
                'recycling_info': 'Check recycling symbols. Rinse containers.',
                'color': (0, 0, 255)  # Red
            },
            'Metal': {
                'description': 'Metal cans, foil, and containers',
                'recycling_info': 'Rinse containers. Most metals are recyclable.',
                'color': (128, 128, 128)  # Gray
            },
            'Trash': {
                'description': 'Non-recyclable waste',
                'recycling_info': 'Dispose in regular trash bin.',
                'color': (128, 0, 128)  # Purple
            },
            'Unknown': {
                'description': 'Unable to identify material',
                'recycling_info': 'Please consult local recycling guidelines.',
                'color': (255, 255, 255)  # White
            }
        }
        
        # Handle lowercase class names
        class_name_cap = class_name.capitalize()
        return class_info.get(class_name_cap, class_info['Unknown'])


# Utility function to train and save standalone models
def train_and_save_standalone_models():
    """Train both classifiers and save them in standalone format"""
    # Load data
    X = pd.read_csv("data/features/features.csv", header=None)
    y = pd.read_csv("data/features/labels.csv", header=None).values.ravel()
    
    # Train SVM classifier
    print("Training SVM classifier...")
    svm_classifier = SVMClassifier(kernel='rbf', C=1000, gamma='scale', n_components=100)
    svm_classifier.class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
    svm_classifier.train(X, y, test_size=0.2)
    
    # Save SVM model
    svm_model_path = "models/svm_classifier.pkl"
    joblib.dump(svm_classifier, svm_model_path)
    print(f"Saved SVM model to {svm_model_path}")
    
    # Train KNN classifier
    print("\nTraining KNN classifier...")
    knn_classifier = KNNClassifier(n_neighbors=3, weights='distance', n_components=70)
    knn_classifier.class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
    knn_classifier.train(X, y, test_size=0.2)
    
    # Save KNN model
    knn_model_path = "models/knn_classifier.pkl"
    joblib.dump(knn_classifier, knn_model_path)
    print(f"Saved KNN model to {knn_model_path}")
    
    return svm_model_path, knn_model_path


if __name__ == "__main__":
    # Example usage
    print("Training and saving models...")
    train_and_save_standalone_models()
    
    print("\nTesting unified classifier...")
    classifier = UnifiedMaterialClassifier()
    
    if classifier.load_models("models/svm_classifier.pkl", "models/knn_classifier.pkl"):
        print("Models loaded successfully!")
        print("Ready for predictions!")