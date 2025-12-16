import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class KNNClassifier:
    def __init__(self, n_neighbors=5, weights='distance', n_components=100):
        """
        Initialize k-NN Classifier
        
        Args:
            n_neighbors (int): Number of neighbors to use for queries.
            weights (str): Weight function used in prediction.
                           'uniform': All points in each neighborhood are weighted equally.
                           'distance': Weight points by the inverse of their distance.
            n_components (int): Number of components for nca dimensionality reduction.
        """
        
        self.knn_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            n_jobs=-1,
            p=1  # Manhattan distance 
            #p=2  # Euclidean distance
              
                                              )
        
        self.scaler = StandardScaler()
        #self.nca = nca(n_components=n_components)
        self.nca=NeighborhoodComponentsAnalysis(n_components=n_components)
        self.class_names = None
        self.is_trained = False

    def train(self, X, y, test_size=0.2, random_state=82):
        """
        Train the k-NN classifier
        
        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of test set
            random_state: Random seed
        """
        print("=" * 60)
        print(f"TRAINING k-NN CLASSIFIER (Weights: {self.knn_model.weights})")
        print("=" * 60)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")

        # 1. Scale features (Crucial for k-NN because it is distance-based)
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 2. Apply nca (Neighborhood Components Analysis)
        print("Applying Neighborhood Components Analysis...")
        X_train_nca = self.nca.fit_transform(X_train_scaled, y_train)
        X_test_nca = self.nca.transform(X_test_scaled)
        
        print(f"Reduced Feature dimension: {X_train_nca.shape[1]}")

        # 3. Fit k-NN
        print("Fitting k-NN model...")
        self.knn_model.fit(X_train_nca, y_train)
        self.is_trained = True

        # 4. Evaluate
        y_pred = self.knn_model.predict(X_test_nca)
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

    def predict_features(self, features_vector):
        """
        Predict class for a raw feature vector (already extracted)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Reshape if single sample
        if len(features_vector.shape) == 1:
            features_vector = features_vector.reshape(1, -1)

        # 1. Scale
        features_scaled = self.scaler.transform(features_vector)
        
        # 2. nca
        features_nca = self.nca.transform(features_scaled)

        # 3. Predict
        prediction = self.knn_model.predict(features_nca)[0]
        probabilities = self.knn_model.predict_proba(features_nca)[0]

        class_name = self.class_names[prediction] if self.class_names else str(prediction)
        confidence = probabilities[prediction]

        return class_name, confidence

def main():
    # --- Configuration ---
    # Assuming CSVs are in the same location as your SVM script
    FEATURE_FILE = "data/features/features.csv"
    LABEL_FILE = "data/features/labels.csv"

    # --- Load Data ---
    print("Loading features from CSV...")
    try:
        X = pd.read_csv(FEATURE_FILE, header=None).values
        y = pd.read_csv(LABEL_FILE, header=None).values.ravel()
        print(f"Loaded {len(X)} samples.")
    except FileNotFoundError:
        print("Error: Feature files not found. Please ensure data/features/features.csv exists.")
        return

    # --- Initialize k-NN Classifier ---
    # Requirements met here: 
    # 1. Accepts feature input (X)
    # 2. Accepts weighting scheme ('distance' or 'uniform')
    
    knn = KNNClassifier(
        n_neighbors=3,      # k=3
        weights='distance', # Weight points by inverse of their distance
        n_components=70   # nca components
    )
    
    knn.class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']

    # --- Train ---
    knn.train(X, y, test_size=0.2)

    
if __name__ == "__main__":
    main()