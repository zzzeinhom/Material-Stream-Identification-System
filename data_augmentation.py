import numpy as np
import cv2
import os
from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image, ImageEnhance


class DataAugmentor:
    def __init__(self):

        # Augmentation techniques
        self.augmentation_methods = [
            self.rotate_image,
            self.flip_horizontal,
            self.flip_vertical,
            self.adjust_brightness,
            self.adjust_contrast,
            self.add_gaussian_noise,
            self.blur_image,
            self.zoom_image,
            self.translate_image
        ]
    
    def rotate_image(self, img):

        # Choose random angle between -30 and 30 degrees
        angle = random.uniform(-30, 30)
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Calculate rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform rotation
        rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        return rotated
    
    def flip_horizontal(self, img):

        # 1 = horizontal flip
        return cv2.flip(img, 1)
    
    def flip_vertical(self, img):

        # 0 = vertical flip
        return cv2.flip(img, 0)
    
    def adjust_brightness(self, img):

        # Convert to PIL for easier brightness adjustment
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Random brightness factor (0.7 to 1.3)
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(pil_img)
        brightened = enhancer.enhance(factor)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR)
    
    def adjust_contrast(self, img):

        # Convert to PIL for easier contrast adjustment
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Random contrast factor (0.8 to 1.2)
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(pil_img)
        contrasted = enhancer.enhance(factor)
        
        # Convert back to OpenCV format
        return cv2.cvtColor(np.array(contrasted), cv2.COLOR_RGB2BGR)
    
    def add_gaussian_noise(self, img):
 
        # Generate random noise
        noise = np.random.normal(0, 1, img.shape).astype(np.uint8)
        
        # Add noise to image
        noisy_img = cv2.add(img, noise)
        
        return noisy_img
    
    def blur_image(self, img):

        # Random kernel size (must be odd)
        kernel_size = random.choice([3, 5, 7])
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        return blurred
    
    def zoom_image(self, img):

        height, width = img.shape[:2]
        
        # Random zoom factor (0.8 to 1.2)
        zoom_factor = random.uniform(0.8, 1.2)
        
        # Calculate new dimensions
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)
        
        # Resize image
        zoomed = cv2.resize(img, (new_width, new_height))
        
        # zoom in
        if zoom_factor > 1:
            # Crop center
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            zoomed = zoomed[start_y:start_y+height, start_x:start_x+width]

        # zoom out
        else:
            # Fill with black borders
            fill_x = (width - new_width) // 2
            fill_y = (height - new_height) // 2
            zoomed = cv2.copyMakeBorder(zoomed, fill_y, height-new_height-fill_y, 
                                       fill_x, width-new_width-fill_x, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return zoomed
    
    def translate_image(self, img):

        height, width = img.shape[:2]
        
        # Random translation -20 to 20 pixels
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # Apply translation
        translated = cv2.warpAffine(img, translation_matrix, (width, height))
        
        return translated
    
    def augment_image(self, img, num_augmentations=1):

        augmented = img.copy()
        
        # Randomly select augmentation methods
        methods = random.sample(self.augmentation_methods, 
                              min(num_augmentations, len(self.augmentation_methods)))
        
        # Apply each method
        for method in methods:
            augmented = method(augmented)
        
        return augmented


def augment_dataset(input_dir, output_dir, target_count=-1):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Class names
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    
    print("="*60)
    print("STARTING DATA AUGMENTATION")
    print("="*60)
    
    for class_name in class_names:
        print("="*60)
        print("AUGMENTING", class_name.upper())
        print("="*60)
        # Setup paths
        input_class_path = Path(input_dir) / class_name
        output_class_path = Path(output_dir) / class_name
        os.makedirs(output_class_path, exist_ok=True)
        
        if not input_class_path.exists():
            print(f"    Warning: {class_name} folder not found, skipping...")
            continue
        
        # Get all original images
        original_images = list(input_class_path.glob('*.jpg'))
        
        num_original = len(original_images)
        if num_original == 0:
            print(f"    Warning: No images found in {class_name}, skipping...")
            continue

        # Calculate target count
        if target_count == -1:
            target_count = num_original + int(num_original * 35 / 100) #increasing sample size by 35%

        print(f"    Target count for {class_name}: {target_count}")

        # Initialize augmentor
        augmentor = DataAugmentor()
        
        print(f"    Original images: {num_original}")
        
        # Calculate how many augmented images we need
        num_augmented_needed = target_count - num_original
        
        if num_augmented_needed <= 0:
            print(f"    Copying original images only...\n")
            num_augmented_needed = 0            
        
        # Copy originals, filter valid images
        print(f"    Processing original images...")
        valid_originals = []
        failed_copies = []
        
        for idx, img_path in enumerate(tqdm(original_images, desc="    Processing")):
            try:
                # Load image
                img = cv2.imread(str(img_path))
                
                # Check if image loaded successfully
                if img is None or img.size == 0:
                    failed_copies.append(str(img_path))
                    continue
                
                # Save original
                output_path = output_class_path / f"{class_name}_orig_{idx:04d}.jpg"
                success = cv2.imwrite(str(output_path), img)
                
                if success:
                    valid_originals.append(img_path)
                else:
                    failed_copies.append(str(img_path))
                    
            except Exception as e:
                failed_copies.append(f"{img_path} (Error: {str(e)})")
                continue
        
        # Report results
        print(f"\n    Successfully copied: {len(valid_originals)} images")
        print(f"    Failed to copy: {len(failed_copies)} images\n")
        
        # Generate augmented images if needed
        num_augmented_needed = target_count - len(valid_originals)

        print(f"    Need to generate: {num_augmented_needed} augmented images")
        
        if num_augmented_needed > 0:
            if len(valid_originals) == 0:
                print(f"    No valid images found to augment!")
                continue

            print(f"    Generating augmented images...")
            
            successful_aug = 0
            failed_aug = 0
            pbar = tqdm(total=num_augmented_needed, desc="    Augmenting")
            
            while successful_aug < num_augmented_needed:
                try:
                    # Pick a random valid original image
                    source_img_path = random.choice(valid_originals)
                    img = cv2.imread(str(source_img_path))
                    
                    # Double check image loaded
                    if img is None or img.size == 0:
                        failed_aug += 1
                        continue
                    
                    # Apply random augmentations (1-2 techniques)
                    num_techniques = random.randint(1, 2)
                    augmented = augmentor.augment_image(img, num_techniques)
                    
                    # Save augmented image
                    output_path = output_class_path / f"{class_name}_aug_{len(valid_originals) + successful_aug:04d}.jpg"
                                
                    success = cv2.imwrite(str(output_path), augmented)
                    
                    if success:
                        successful_aug += 1
                        pbar.update(1)
                    else:
                        failed_aug += 1
                        
                except Exception as e:
                    failed_aug += 1
                    continue
            
            pbar.close()
            
            print(f"\n    Successfully augmented: {successful_aug}")
            print(f"    Failed to augment: {failed_aug}")
    
    print("="*60)
    print("DATA AUGMENTATION COMPLETE!")
    print("="*60 + "\n")


def create_unknown_class(output_dir, target_count=-1):

    print("="*60)
    print("CREATING 'UNKNOWN' CLASS")
    print("="*60)
    
    # Create unknown folder
    unknown_path = Path(output_dir) / 'unknown'
    os.makedirs(unknown_path, exist_ok=True)
    
    # Get all images from other classes
    class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
    all_images = []
    
    for class_name in class_names:
        class_path = Path(output_dir) / class_name
        if class_path.exists():
            images = list(class_path.glob('*.jpg'))
            all_images.extend(images)
    
    if len(all_images) == 0:
        print("    Error: No images found in any class!")
        return
    
    if target_count == -1:
        target_count = len(all_images) // len(class_names)

    print(f"    Generating {target_count} 'unknown' images...")
    print("    Methods: Heavy blur, extreme distortion, noise")
    
    for idx in tqdm(range(target_count), desc="    Generating unknown"):
        # Pick random image
        source_img_path = random.choice(all_images)
        img = cv2.imread(str(source_img_path))
        
        # Apply extreme transformations to make it "unknown"
        method = random.choice(['heavy_blur', 'extreme_noise', 'dark', 'overexposed'])
        
        if method == 'heavy_blur':
            img = cv2.GaussianBlur(img, (21, 21), 0)
        
        elif method == 'extreme_noise':
            noise = np.random.normal(0, 50, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        elif method == 'dark':
            img = cv2.convertScaleAbs(img, alpha=0.3, beta=0)
        
        elif method == 'overexposed':
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
        
        output_path = unknown_path / f"unknown_{idx:04d}.jpg"
        cv2.imwrite(str(output_path), img)
    
    print(f"\nGenerated {target_count} 'unknown' images")
    print("="*60)
    print("UNKNOWN CLASS COMPLETE!")
    print("="*60 + "\n")


if __name__ == "__main__":

    # Path to original dataset
    ORIGINAL_DATASET_DIR = "dataset"
    
    # Path to save augmented dataset
    AUGMENTED_DATASET_DIR = "data/train"
    
    # Target number of images per class
    TARGET_COUNT_PER_CLASS = 500
    
    augment_dataset(
        input_dir=ORIGINAL_DATASET_DIR,
        output_dir=AUGMENTED_DATASET_DIR,
        target_count=TARGET_COUNT_PER_CLASS
    )
    
    # Create unknown class
    create_unknown_class(
        output_dir=AUGMENTED_DATASET_DIR,
        target_count=TARGET_COUNT_PER_CLASS
    )
