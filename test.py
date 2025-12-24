

import numpy as np
import cv2
import os
from pathlib import Path
import joblib
import pandas as pd
from tqdm import tqdm

try:
    from cnn_feature_extraction import CNNFeatureExtractor
    from SVM_classifier import SVMClassifier
    from KNN_classifier import KNNClassifier
    USING_PROJECT_IMPORTS = True
except ImportError:
    USING_PROJECT_IMPORTS = False
    from keras.applications import ResNet50
    from keras.applications.resnet50 import preprocess_input
    from keras.models import Model
    import tensorflow as tf
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    class CNNFeatureExtractor:
        def __init__(self, model_name='ResNet50', input_size=(224, 224)):
            self.model_name = model_name
            self.input_size = input_size
            self.model = None
            self.preprocess_func = None
            self._load_model()
            
        def _load_model(self):
            base_model = ResNet50(
                weights='imagenet', 
                include_top=False, 
                input_shape=(*self.input_size, 3), 
                pooling='avg'
            )
            self.model = base_model
            self.preprocess_func = preprocess_input
            self.feature_dim = 2048
        
        def extract_features_from_image(self, img):
            img_resized = cv2.resize(img, self.input_size)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_batch = np.expand_dims(img_rgb, axis=0)
            img_preprocessed = self.preprocess_func(img_batch)
            features = self.model.predict(img_preprocessed, verbose=0)
            return features.flatten()
        
        def extract_features_from_path(self, img_path):
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    return None
                return self.extract_features_from_image(img)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                return None


def predict(dataFilePath, bestModelPath, outputCSVPath=None):
    print(f"Starting prediction...")
    print(f"Data path: {dataFilePath}")
    print(f"Model path: {bestModelPath}")
    
    try:
        classifier = joblib.load(bestModelPath)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return []
    
    try:
        feature_extractor = CNNFeatureExtractor(model_name='ResNet50', input_size=(224, 224))
        print("Feature extractor initialized")
    except Exception as e:
        print(f"Error initializing feature extractor: {e}")
        return []
    
    data_path = Path(dataFilePath)
    if not data_path.exists():
        print(f"Data path does not exist: {dataFilePath}")
        return []
    else:
        print(f"Data path exists")
    
    # Find images
    image_files = sorted(list(data_path.glob('*.jpg')) + list(data_path.glob('*.png')))
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("✗ No images found in directory")
        return []
    
    
    
    predictions = []
    image_names = []
    
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
    
    for img_path in tqdm(image_files, desc="Processing"):
        image_names.append(img_path.stem)
        
        try:
            features = feature_extractor.extract_features_from_path(img_path)
            
            if features is None:
                predictions.append(6)
                continue
            
            features = features.reshape(1, -1)
            
            class_name, confidence = classifier.predict(features)
            
            prediction_id = class_names.index(class_name)
            
            predictions.append(prediction_id)
            
        except Exception as e:
            predictions.append(6)
            continue
    
    if outputCSVPath is not None:
        predicted_labels = [class_names[pred] for pred in predictions]
        
        df = pd.DataFrame({
            'ImageName': image_names,
            'predictedlabel': predicted_labels
        })
        
        df.to_csv(outputCSVPath, index=False)
        print(f"\n Predictions saved to: {outputCSVPath}")
    
    return predictions


if __name__ == "__main__":
    TEST_DATA_PATH = "data/sample" 
    MODEL_PATH = "models/svm_classifier.pkl"
    OUTPUT_CSV = "data/predictions.csv"
    
    print("="*50)
    print("Starting prediction pipeline...")
    print("="*50)
    
    predictions = predict(
        dataFilePath=TEST_DATA_PATH,
        bestModelPath=MODEL_PATH,
        outputCSVPath=OUTPUT_CSV 
    )
    
    if predictions:
        print(f"\n✓ Returned {len(predictions)} predictions")
        print(f"First 10 predictions: {predictions[:10]}")
    else:
        print("\n✗ No predictions returned - check error messages above")