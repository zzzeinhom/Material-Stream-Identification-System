"""
Feature extraction for GUI application
Matches training configuration: method='lbp+color+glcm', img_size=(128, 128)
Produces exactly 286 features
"""

import numpy as np
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB, COLOR_BGR2HSV, Canny, Sobel, CV_64F
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage import color
from pathlib import Path

class FeatureExtractor:
    """Extracts features for material classification - matches training config"""
    
    # Expected feature dimension for validation
    EXPECTED_FEATURES = 286
    
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        
        # LBP configuration (multi-scale)
        self.lbp_radii = [1, 2, 3]
        self.lbp_n_points = [8, 16, 24]
        
    def preprocess_image(self, image_input):
        """
        Preprocess image from either file path or numpy array
        Args:
            image_input: Either str/pathlib.Path or numpy array (BGR format)
        Returns:
            Preprocessed image in RGB format, normalized to [0, 1]
        """
        if isinstance(image_input, (str, Path)):
            # Load from file
            img = imread(str(image_input))
            if img is None:
                raise IOError(f"Unable to read image: {image_input}")
        else:
            # Use provided array (assume BGR from OpenCV)
            img = image_input.copy()
        
        # Resize
        img = resize(img, self.img_size)
        
        # Convert BGR to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cvtColor(img, COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        return img
    
    def extract_lbp(self, gray_img):
        """Extract multi-scale LBP features (60 features)"""
        all_features = []
        
        for radius, n_points in zip(self.lbp_radii, self.lbp_n_points):
            lbp = local_binary_pattern(gray_img, n_points, radius, method='uniform')
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(float) / (hist.sum() + 1e-7)
            all_features.extend(hist)
        
        return np.array(all_features)
    
    def extract_color_rgb(self, img):
        """Extract RGB color histograms (96 features)"""
        features = []
        n_bins = 32
        
        for channel in range(3):
            hist, _ = np.histogram(img[:, :, channel], bins=n_bins, range=(0, 1))
            hist = hist.astype(float) / (hist.sum() + 1e-7)
            features.extend(hist)
        
        return np.array(features)
    
    def extract_color_hsv(self, img):
        """Extract HSV color histograms (96 features)"""
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
    
    def extract_glcm(self, gray_img):
        """Extract GLCM texture features (40 features)"""
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
        
        features = []
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        for prop in properties:
            prop_values = graycoprops(glcm, prop).ravel()
            features.extend(prop_values)
        
        return np.array(features)
    
    def extract_features(self, image_input):
        """
        Extract all features using the training configuration
        Args:
            image_input: Image file path or numpy array (BGR)
        Returns:
            Feature vector (286 features)
        """
        # Preprocess image
        img = self.preprocess_image(image_input)
        
        # Convert to grayscale once for LBP and GLCM
        gray_img = color.rgb2gray(img)
        gray_img = (gray_img * 255).astype(np.uint8)
        
        # Extract features in same order as training
        features = []
        features.extend(self.extract_lbp(gray_img))        # 60 features
        features.extend(self.extract_color_rgb(img))       # 96 features
        features.extend(self.extract_color_hsv(img))       # 96 features
        features.extend(self.extract_glcm(gray_img))       # 40 features
        
        # Validate feature count
        features_array = np.array(features)
        if features_array.shape[0] != self.EXPECTED_FEATURES:
            raise ValueError(
                f"Feature extraction error: got {features_array.shape[0]} features, "
                f"expected {self.EXPECTED_FEATURES}. Check configuration."
            )
        
        return features_array


# GUI-ready utility functions
def extract_features_for_prediction(image_input, img_size=(128, 128)):
    """
    Main function for GUI - extract features ready for classification
    Args:
        image_input: File path or numpy array
        img_size: Image size (must match training)
    Returns:
        Feature vector (286, )
    """
    extractor = FeatureExtractor(img_size=img_size)
    features = extractor.extract_features(image_input)
    return features.flatten()  # Ensure 1D array


# Example usage for GUI
if __name__ == "__main__":
    # Test with file path
    test_image = "data/train/glass/sample.jpg"
    features = extract_features_for_prediction(test_image)
    print(f"File input: {features.shape}")  # Should be (286,)
    
    # Test with numpy array (simulating GUI capture)
    import cv2
    test_array = cv2.imread(test_image)  # BGR array
    features = extract_features_for_prediction(test_array)
    print(f"Array input: {features.shape}")  # Should be (286,)
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")