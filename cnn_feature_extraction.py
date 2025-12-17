import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from keras.applications import VGG16, ResNet50, MobileNetV2
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from keras.models import Model
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNNFeatureExtractor:
    
    
    def __init__(self, model_name='VGG16', input_size=(224, 224)):
        """
        Initialize the CNN Feature Extractor
        
        Args:
            model_name (str): Name of the pre-trained model to use
                            Options: 'VGG16', 'ResNet50', 'MobileNetV2'
            input_size (tuple): Input image size for the model (height, width)
        """
        self.model_name = model_name
        self.input_size = input_size
        self.model = None
        self.preprocess_func = None
        
        print("=" * 60)
        print(f"INITIALIZING CNN FEATURE EXTRACTOR: {model_name}")
        print("=" * 60)
        
        self._load_model()
        
    def _load_model(self):
        """
        Load pre-trained CNN model and remove classification layers
        to use it as a feature extractor
        """
        print(f"Loading pre-trained {self.model_name} model...")
        
        if self.model_name == 'VGG16':
            # VGG16: Deep model with excellent feature extraction capabilities
            # Using 'fc2' layer gives 4096-dimensional features
            base_model = VGG16(weights='imagenet', include_top=True, input_shape=(*self.input_size, 3))
            self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
            self.preprocess_func = vgg_preprocess
            self.feature_dim = 4096
            
        elif self.model_name == 'ResNet50':
            # ResNet50: Uses residual connections, good for deeper features
            # Using 'avg_pool' layer gives 2048-dimensional features
            base_model = ResNet50(weights='imagenet', include_top=False, 
                                 input_shape=(*self.input_size, 3), pooling='avg')
            self.model = base_model
            self.preprocess_func = resnet_preprocess
            self.feature_dim = 2048
            
        elif self.model_name == 'MobileNetV2':
            # MobileNetV2: Lightweight and fast, good for deployment
            # Using global average pooling gives 1280-dimensional features
            base_model = MobileNetV2(weights='imagenet', include_top=False,
                                    input_shape=(*self.input_size, 3), pooling='avg')
            self.model = base_model
            self.preprocess_func = mobilenet_preprocess
            self.feature_dim = 1280
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        print(f"Model loaded successfully!")
        print(f"Feature dimension: {self.feature_dim}")
        print("=" * 60 + "\n")
    
    def extract_features_from_image(self, img):
        """
        Extract features from a single image
        
        Args:
            img: OpenCV image (BGR format)
            
        Returns:
            features: 1D numpy array of features
        """
        # Resize image to model input size
        img_resized = cv2.resize(img, self.input_size)
        
        # Convert BGR to RGB (OpenCV uses BGR, but models expect RGB)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension and preprocess
        img_batch = np.expand_dims(img_rgb, axis=0)
        img_preprocessed = self.preprocess_func(img_batch)
        
        # Extract features
        features = self.model.predict(img_preprocessed, verbose=0)
        
        # Flatten to 1D vector
        features = features.flatten()
        
        return features
    
    def extract_features_from_path(self, img_path):
        """
        Extract features from an image file path
        
        Args:
            img_path: Path to image file
            
        Returns:
            features: 1D numpy array of features, or None if loading fails
        """
        try:
            img = cv2.imread(str(img_path))
            
            if img is None or img.size == 0:
                return None
            
            return self.extract_features_from_image(img)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None


def extract_dataset_features(dataset_dir, output_dir, model_name='VGG16'):
    """
    Extract features from all images in the dataset and save to CSV files
    
    Args:
        dataset_dir (str): Directory containing class folders with images
        output_dir (str): Directory to save feature CSV files
        model_name (str): CNN model to use for feature extraction
    """
    print("=" * 60)
    print("FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize feature extractor
    extractor = CNNFeatureExtractor(model_name=model_name)
    
    # Class names and their corresponding labels
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
    class_to_label = {name: idx for idx, name in enumerate(class_names)}
    
    # Storage for all features and labels
    all_features = []
    all_labels = []
    
    print("Processing dataset...")
    print("=" * 60)
    
    # Process each class
    for class_name in class_names:
        class_path = Path(dataset_dir) / class_name
        
        if not class_path.exists():
            print(f"Warning: Class folder '{class_name}' not found. Skipping...")
            continue
        
        # Get all images in this class
        image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        
        if len(image_files) == 0:
            print(f"Warning: No images found in '{class_name}'. Skipping...")
            continue
        
        print(f"\nProcessing class: {class_name.upper()}")
        print(f"  Images found: {len(image_files)}")
        
        # Extract features from each image
        class_features = []
        failed_count = 0
        
        for img_path in tqdm(image_files, desc=f"  Extracting features"):
            features = extractor.extract_features_from_path(img_path)
            
            if features is not None:
                class_features.append(features)
                all_labels.append(class_to_label[class_name])
            else:
                failed_count += 1
        
        all_features.extend(class_features)
        
        print(f"  Successfully processed: {len(class_features)} images")
        if failed_count > 0:
            print(f"  Failed to process: {failed_count} images")
    
    print("\n" + "=" * 60)
    print("SAVING FEATURES TO CSV")
    print("=" * 60)
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    print(f"Total samples: {len(features_array)}")
    print(f"Feature dimension: {features_array.shape[1]}")
    print(f"Classes: {class_names}")
    print(f"Label distribution: {np.bincount(labels_array)}")
    
    # Save to CSV
    features_file = Path(output_dir) / 'features.csv'
    labels_file = Path(output_dir) / 'labels.csv'
    
    print(f"\nSaving features to: {features_file}")
    pd.DataFrame(features_array).to_csv(features_file, index=False, header=False)
    
    print(f"Saving labels to: {labels_file}")
    pd.DataFrame(labels_array).to_csv(labels_file, index=False, header=False)
    
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)
    
    return features_array, labels_array



def main():

    DATASET_DIR = "data/train"  
    OUTPUT_DIR = "data/features"
    
    # Options: 'VGG16' (4096 features, slower but powerful)
    #          'ResNet50' (2048 features, balanced)
    #          'MobileNetV2' (1280 features, fastest)
    MODEL_NAME = 'ResNet50'
    

    
    extract_dataset_features(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME
    )
    
    print("\n Feature extraction complete!")
    print(f" Features saved to: {OUTPUT_DIR}/features.csv")
    print(f" Labels saved to: {OUTPUT_DIR}/labels.csv")
    print("\nYou can now use these files to train the SVM and k-NN classifiers")


if __name__ == "__main__":
    main()