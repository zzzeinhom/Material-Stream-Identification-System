import numpy as np
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB, COLOR_BGR2HSV, Canny, Sobel, CV_64F
import os
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage import color
from pathlib import Path
from tqdm import tqdm


class FeatureExtractor:
   
    
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        
        # HOG parameters
        self.gradient_config = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'visualize': False,
            'feature_vector': True
        }
        
        # LBP parameters (multi-scale)
        self.lbp_radii = [1, 2, 3]
        self.lbp_n_points = [8, 16, 24]
        
    def preprocess_image(self, img_path):
        img = imread(str(img_path))
        if img is None:
            raise IOError(f"Unable to read image: {img_path}")
        img = resize(img, self.img_size)
        img = cvtColor(img, COLOR_BGR2RGB)
        return img
    
    def extract_hog(self, img):
        """
        HOG - Histogram of Oriented Gradients
        Captures edges and shapes
        Good for: Detecting object boundaries and overall shape
        """
        gray_img = color.rgb2gray(img)
        gray_img = (gray_img * 255).astype(np.uint8) / 255.0
        features = hog(gray_img, **self.gradient_config)
        return features
    
    def extract_color_rgb(self, img):
        """
        RGB Color Histograms (96 features)
        Good for: Basic color information
        """
        features = []
        n_bins = 32
        
        for channel in range(3):
            hist, _ = np.histogram(img[:, :, channel], bins=n_bins, range=(0, 1))
            hist = hist.astype(float) / (hist.sum() + 1e-7)
            features.extend(hist)
        
        return np.array(features)
    
    def extract_color_hsv(self, img):
        """
        HSV Color Histograms (96 features)
        Good for: Better color discrimination (separates hue from brightness)
        CRITICAL for materials with distinct colors
        """
        features = []
        n_bins = 32
        
        # Convert to HSV
        hsv_img = cvtColor((img * 255).astype(np.uint8), COLOR_BGR2HSV)
        hsv_img = hsv_img.astype(float) / 255.0
        
        for channel in range(3):
            hist, _ = np.histogram(hsv_img[:, :, channel], bins=n_bins, range=(0, 1))
            hist = hist.astype(float) / (hist.sum() + 1e-7)
            features.extend(hist)
        
        return np.array(features)
    
    def extract_lbp(self, img):
        """
        LBP - Local Binary Patterns (multi-scale, 60 features)
        Good for: Texture analysis (smooth vs rough surfaces)
        CRITICAL for distinguishing material textures
        """
        gray_img = color.rgb2gray(img)
        gray_img = (gray_img * 255).astype(np.uint8)
        
        all_features = []
        
        # Multi-scale LBP
        for radius, n_points in zip(self.lbp_radii, self.lbp_n_points):
            lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(float) / (hist.sum() + 1e-7)
            all_features.extend(hist)
        
        return np.array(all_features)
    
    def extract_glcm(self, img):
        """
        GLCM - Gray Level Co-occurrence Matrix (40 features)
        Good for: Surface texture properties
        CRITICAL for materials like metal vs glass (different reflective properties)
        """
        gray_img = color.rgb2gray(img)
        gray_img = (gray_img * 255).astype(np.uint8)
        
        # Calculate GLCM
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(
            gray_img, 
            distances=distances, 
            angles=angles, 
            levels=256,
            symmetric=True, 
            normed=True
        )
        
        # Extract properties
        features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        for prop in properties:
            prop_values = graycoprops(glcm, prop).ravel()
            features.extend(prop_values)
        
        return np.array(features)
    
    def extract_edge(self, img):
        """
        Edge Density Features (5 features)
        Good for: Edge characteristics (materials have different edge patterns)
        """
        gray_img = color.rgb2gray(img)
        gray_img = (gray_img * 255).astype(np.uint8)
        
        features = []
        
        # Canny edges
        edges = Canny(gray_img, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Sobel gradients
        sobelx = Sobel(gray_img, CV_64F, 1, 0, ksize=3)
        sobely = Sobel(gray_img, CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        
        features.extend([
            np.mean(gradient_mag),
            np.std(gradient_mag),
            np.median(gradient_mag),
            np.max(gradient_mag)
        ])
        
        return np.array(features)
    
    def extract_stats(self, img):
        """
        Statistical Features 
        Good for: Overall brightness and color variance
        """
        features = []
        
        # RGB statistics
        for channel in range(3):
            vals = img[:, :, channel].flatten()
            features.extend([
                np.mean(vals),
                np.std(vals),
                np.median(vals),
                np.min(vals),
                np.max(vals),
                np.percentile(vals, 25),
                np.percentile(vals, 75)
            ])
        
        # HSV statistics
        hsv_img = cvtColor((img * 255).astype(np.uint8), COLOR_BGR2HSV)
        hsv_img = hsv_img.astype(float) / 255.0
        
        for channel in range(3):
            vals = hsv_img[:, :, channel].flatten()
            features.extend([
                np.mean(vals),
                np.std(vals),
                np.median(vals)
            ])
        
        return np.array(features)
    
    def extract_features(self, img_path, method='all'):
        """
        Extract features based on method
        
        Parameters:
            img_path: Path to image
            method: Which features to extract
                - 'hog': Only HOG features (~8100)
                - 'color': RGB + HSV histograms (192)
                - 'lbp': Multi-scale texture (60)
                - 'glcm': Surface texture (40)
                - 'edge': Edge features (5)
                - 'stats': Statistics (30)
                - 'all': All features combined (~8427)
                - Custom: 'hog+color', 'lbp+glcm', etc.
        
        Returns:
            Feature vector as numpy array
        """
        # Load image
        img = self.preprocess_image(img_path)
        
        features = []
        
        # Parse method string
        methods = method.lower().split('+') if '+' in method else [method.lower()]
        
        for m in methods:
            m = m.strip()
            
            if m == 'hog' or m == 'all':
                features.append(self.extract_hog(img))
            
            if m == 'color' or m == 'all':
                features.append(self.extract_color_rgb(img))
                features.append(self.extract_color_hsv(img))
            
            if m == 'lbp' or m == 'all':
                features.append(self.extract_lbp(img))
            
            if m == 'glcm' or m == 'all':
                features.append(self.extract_glcm(img))
            
            if m == 'edge' or m == 'all':
                features.append(self.extract_edge(img))
            
            if m == 'stats' or m == 'all':
                features.append(self.extract_stats(img))
        
        # Combine all features
        if len(features) == 0:
            raise ValueError(f"Unknown method: {method}")
        
        combined = np.concatenate(features)
        return combined


def process_dataset(data_dir, output_dir, method='all', img_size=(128, 128)):
    """
    Process entire dataset and extract features
    
    Parameters:
        data_dir: Directory with class folders
        output_dir: Where to save features
        method: Which features to extract (see extract_features for options)
        img_size: Image resize dimensions
    """
    extractor = FeatureExtractor(img_size=img_size)
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown']
    
    all_features = []
    all_labels = []
    
    print("="*60)
    print("UNIFIED FEATURE EXTRACTION")
    print("="*60)
    print(f"Method: {method.upper()}")
    print(f"Image size: {img_size}")
    print("="*60)
    
    # Process each class
    for class_id, class_name in enumerate(class_names):
        class_path = Path(data_dir) / class_name
        
        if not class_path.exists():
            print(f"\n   Skipping {class_name} - folder not found")
            continue
        
        images = list(class_path.glob('*.jpg'))
        
        if len(images) == 0:
            print(f"\n  No images in {class_name}")
            continue
        
        print(f"\n {class_name} (ID: {class_id})")
        print(f"   Found: {len(images)} images")
        
        for img_path in tqdm(images, desc=f"   Processing"):
            try:
                features = extractor.extract_features(img_path, method=method)
                all_features.append(features)
                all_labels.append(class_id)
            except Exception as e:
                print(f"    Error: {img_path.name}: {str(e)}")
                continue
        
        count = sum(1 for label in all_labels if label == class_id)
        print(f"    Extracted: {count} samples")
    
    # Save
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total samples: {len(features_array)}")
    print(f"Feature size: {features_array.shape[1]}")
    print(f"Classes: {len(np.unique(labels_array))}")
    
    features_path = Path(output_dir) / 'features.csv'
    labels_path = Path(output_dir) / 'labels.csv'
    
    np.savetxt(features_path, features_array, delimiter=',')
    np.savetxt(labels_path, labels_array, fmt='%d', delimiter=',')
    
    print(f"\n Saved: {features_path}")
    print(f" Saved: {labels_path}")
    print("="*60 + "\n")
    
    return features_array, labels_array


def main():
    """Main execution"""
    TRAIN_DIR = "data/train"
    OUTPUT_DIR = "data/features"
    
    # Choose feature method:
    # - 'all': you can use all feature extraction methods together
    # - 'color+lbp': you can combine specific feature extraction methods
    # - 'hog' : you can use a single feature extraction method
    METHOD = 'lbp+color+glcm'

    
    print(" Starting feature extraction...")
    print(f"   Method: {METHOD}\n")
    
    features, labels = process_dataset(
        data_dir=TRAIN_DIR,
        output_dir=OUTPUT_DIR,
        method=METHOD,
        img_size=(128, 128)
    )
    
    print(" Feature extraction complete!")
    print(f"   Shape: {features.shape}")
    print(f"\n Please proceed to train classifiers with these features")


if __name__ == "__main__":
    main()