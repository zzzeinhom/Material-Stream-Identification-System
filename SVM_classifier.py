import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from tqdm import tqdm


class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', n_components=200):
        """
        Initialize SVM Classifier

        Args:
            kernel: SVM kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
        """

        self.svm_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash','unknown']
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

        print("Applying PCA dimensionality reduction...")  # <--- ADDED
        # Fit PCA on training data and transform it
        X_train_pca = self.pca.fit_transform(X_train_scaled)  # <--- ADDED
        # Transform test data (DO NOT fit PCA on test data)
        X_test_pca = self.pca.transform(X_test_scaled)  # <--- ADDED

        print(f"Reduced Feature dimension: {X_train_pca.shape[1]}")  # <--- ADDED

        # Train SVM
        print("Training SVM model...")
        self.svm_model.fit(X_train_pca, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.svm_model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        #
        # print("\nConfusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))

        print("=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60 + "\n")

        return X_test_scaled, y_test

    def predict(self, X, threshold=0.54):
        """
        Returns:
            class_name: Predicted class name
            confidence: Prediction confidence
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

          # Reshape if single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        # Predict
        prediction_index = self.svm_model.predict(X_pca)[0]  # predict returns an array
        probabilities = self.svm_model.predict_proba(X_pca)[0]

        class_name = self.class_names[prediction_index]
        confidence = probabilities[prediction_index]

        if confidence < threshold:
            return "unknown", confidence

        return class_name, confidence


def main():
    # Initialize classifier
    classifier = SVMClassifier(kernel='rbf', C = 10, gamma='scale')
    # Load and extract features from augmented dataset
    print("Loading and extracting features from augmented dataset...")
    X, y = pd.read_csv("data/features/features.csv", header=None), pd.read_csv("data/features/labels.csv", header=None).values.ravel()
    # print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"Classes: {classifier.class_names}\n")
    classifier.train(X.values, y)
    
    os.makedirs("models", exist_ok=True)
    import joblib
    joblib.dump(classifier, "models/svm_classifier.pkl")
    print("Saved SVM model to models/svm_classifier.pkl")


if __name__ == "__main__":
    main()