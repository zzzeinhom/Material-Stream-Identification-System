import numpy as np
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
import os
from skimage.feature import hog
from skimage import color
from pathlib import Path
from tqdm import tqdm


class ImageFeatureExtractor:
    def __init__(self, img_size=(128, 128)):

        self.img_size = img_size
        
        # HOG parameters
        self.gradient_config = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'visualize': False,
        }
                
    def preprocess_image(self, img_path):
        # Read image
        img = imread(str(img_path))
        
        if img is None:
            raise IOError(f"Unable to read image: {img_path}")
        
        # Resize to standard size
        img = resize(img, self.img_size)
        
        # Convert BGR to RGB
        img = cvtColor(img, COLOR_BGR2RGB)
        
        return img
    
    def extract_gradient_features(self, img):
        # Convert to grayscale
        gray_img = color.rgb2gray(img)
        
        # Extract HOG features
        gradient_features = hog(gray_img, **self.gradient_config)
        
        return gradient_features
    
    #TODO: Implement extract_color_distribution
    def extract_color_distribution(self, img):
        #TODO
        pass
    
    #TODO: Implement extract_texture_features
    def extract_texture_features(self, img):
        #TODO
        pass
    
    def extract_channel_statistics(self, img):
        features = []
        
        # Analyze each color channel separately
        for channel in range(3):
            channel_values  = img[:, :, channel].flatten()
            features.extend([
                np.mean(channel_values),      
                np.std(channel_values),    
                np.median(channel_values),  
                np.min(channel_values),  
                np.max(channel_values)        
            ])
        
        return np.array(features)
    
    def extract_multiple_features(self, img_path, method='all'):

        # Load and preprocess image
        img = self.preprocess_image(img_path)
        
        features = []
        
        if method in ['gradient', 'all']:
            hog_feat = self.extract_gradient_features(img)
            features.append(hog_feat)
        
        if method in ['color', 'all']:
            color_feat = self.extract_color_distribution(img)
            features.append(color_feat)
        
        if method in ['texture', 'all']:
            lbp_feat = self.extract_texture_features(img)
            features.append(lbp_feat)
        
        if method in ['stat', 'all']:
            stat_feat = self.extract_channel_statistics(img)
            features.append(stat_feat)
        
        # Concatenate all features
        combined_features = np.concatenate(features)
        
        return combined_features
    

def process_dataset(data_dir, output_dir, feature_method='all', img_size=(128, 128)):
    # Initialize feature extractor
    extractor = ImageFeatureExtractor(img_size=img_size)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define class names
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']

    
    all_features = []
    all_labels = []
    
    print("="*60)
    print("="*60)
    print("Starting Feature Extraction")
    print(f"Method: {feature_method.upper()}")
    print("="*60)
    
    # Process each class
    for class_id, class_name in enumerate(class_names):
        class_path = Path(data_dir) / class_name
        
        if not class_path.exists():
            print(f"    Class folder not found: {class_path}")
            continue
        
        # Get all images
        images = list(class_path.glob('*.jpg'))
        
        print("="*60)
        print(f"Processing class: {class_name}")
        
        # Process each image
        for img_path in tqdm(images, desc=f"    Extracting {class_name}"):
            try:
                # Extract features
                features = extractor.extract_multiple_features(img_path, method=feature_method)
                
                all_features.append(features)
                all_labels.append(class_id)
                
            except Exception as e:
                print(f"    Error processing {img_path.name}: {str(e)}")
                continue
        
        print(f"    Completed {class_name}: {len([label for label in all_labels if label == class_id])} features extracted\n")
    
    # Convert to numpy arrays
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
  
    # Save features and labels
    np.savetxt(Path(output_dir) / 'features.csv', features_array, delimiter=',')
    np.savetxt(Path(output_dir) / 'labels.csv', labels_array, fmt='%d', delimiter=',')
    
    return features_array, labels_array


def main():    
    # Set your paths
    TRAIN_DATA_DIR = "data/train" 
    OUTPUT_DIR = "data/features"
    
    # Choose feature extraction method
    FEATURE_METHOD = 'gradient'
    
    # Process the dataset
    features, labels = process_dataset(
        data_dir=TRAIN_DATA_DIR,
        output_dir=OUTPUT_DIR,
        feature_method=FEATURE_METHOD,
        img_size=(128, 128)
    )

if __name__ == "__main__":
    main()