import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from tqdm import tqdm


class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0, gamma='scale'):
        """
        Initialize SVM Classifier

        Args:
            kernel: SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        # why probability=True?
        self.svm_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.scaler = StandardScaler()
        self.class_names = None
        self.is_trained = False

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the SVM classifier

        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
        """
        print("=" * 60)
        print("TRAINING SVM CLASSIFIER")
        print("=" * 60)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Feature dimension: {X_train.shape[1]}")

        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train SVM
        print("Training SVM model...")
        self.svm_model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.svm_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60 + "\n")

        return X_test_scaled, y_test

    def predict(self, image_path):
        """
        Predict class for an image

        Args:
            image_path: Path to the image file

        Returns:
            class_name: Predicted class name
            confidence: Prediction confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Extract features from image
        features = self.extract_features_from_image(image_path)
        if features is None:
            raise ValueError(f"Could not extract features from {image_path}")

        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict
        prediction = self.svm_model.predict(features_scaled)[0]
        probabilities = self.svm_model.predict_proba(features_scaled)[0]

        class_name = self.class_names[prediction]
        confidence = probabilities[prediction]

        return class_name, confidence


def main():
    # Configuration
    DATASET_DIR = "data/features"  # Output from data_augmentation.py
    MODEL_SAVE_PATH = "data/models/svm_classifier"

    # Initialize classifier
    classifier = SVMClassifier(kernel='rbf', C=.1, gamma='scale')
    SVMClassifier.class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
    # Load and extract features from augmented dataset
    print("Loading and extracting features from augmented dataset...")
    X, y = pd.read_csv("data/features/features.csv"), pd.read_csv("data/features/labels.csv").values.ravel()
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Classes: {SVMClassifier.class_names}\n")

    # Train model
    classifier.train(X, y, test_size=0.2)



if __name__ == "__main__":
    main()