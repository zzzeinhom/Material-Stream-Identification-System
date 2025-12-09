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
            self.add_pixel_noise,
            self.blur_image,
            self.scale_image,
            self.shift_image
        ]
    
    def rotate_image(self, img):

        # Choose random angle between -30 and 30 degrees
        angle = random.uniform(-30, 30)
        
        # Get image dimensions
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # Calculate rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply rotation
        result = cv2.warpAffine(img, M, (w, h))
        return result
    
    def flip_horizontal(self, img):
        return img[:, ::-1]
    
    def flip_vertical(self, img):
        return img[::-1, :]
        
    
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
    
    def add_pixel_noise(self, img):
 
        # Generate random noise
        noise = np.random.normal(0, 1, img.shape).astype(np.uint8)
        
        # Add noise to image
        noisy_img = cv2.add(img, noise)
        
        return noisy_img
    
    def blur_image(self, img):

        # Random kernel dimension
        kernel_dim = random.choice([3, 5, 7])
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (kernel_dim, kernel_dim), 0)
        
        return blurred
    
    def scale_image(self, img):
        h, w = img.shape[:2]
        scaling_factor = random.uniform(0.75, 1.25)
        
        new_h = int(h * scaling_factor)
        new_w = int(w * scaling_factor)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        if scaling_factor > 1.0:
            # Center crop for zoom in
            crop_y = (new_h - h) // 2
            crop_x = (new_w - w) // 2
            return resized[crop_y:crop_y+h, crop_x:crop_x+w]
        else:
            # Pad for zoom out
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            pad_bottom = h - new_h - pad_y
            pad_right = w - new_w - pad_x
            
            return cv2.copyMakeBorder(
                resized, 
                pad_y, pad_bottom, 
                pad_x, pad_right, 
                cv2.BORDER_CONSTANT, 
                value=[0, 0, 0]
            )
    
    def shift_image(self, img):

        h, w = img.shape[:2]
        
        dx = random.randint(-20, 20)
        dy = random.randint(-20, 20)
        
        # Translation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        
        shifted = cv2.warpAffine(img, M, (w, h))
        
        return shifted

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
            print(f"    Skipping {class_name}, folder not found")
            continue
        
        # Get all original images
        original_images = list(input_class_path.glob('*.jpg'))
        
        num_original = len(original_images)
        if num_original == 0:
            print(f"    Skipping {class_name}, no images found")
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
            print(f"    Copying original images only\n")
            num_augmented_needed = 0            
        
        # Copy originals, filter valid images
        print(f"    Processing original images")
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

            print(f"    Generating augmented images")
            
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
        print("    No images found in any class!")
        return
    
    if target_count == -1:
        target_count = len(all_images) // len(class_names)

    print(f"    Generating {target_count} 'unknown' images")
    print("    Methods: Heavy blur, extreme distortion, noise")
    
    for idx in tqdm(range(target_count), desc="    Generating unknown"):
        # Pick random image
        source_img_path = random.choice(all_images)
        img = cv2.imread(str(source_img_path))
        
        # Apply extreme transformations to make it "unknown"
        method = random.choice(['heavy_blur', 'extreme_noise', 'very_dark', 'overexposed'])
        
        if method == 'heavy_blur':
            img = cv2.GaussianBlur(img, (25, 25), 0)
        
        elif method == 'extreme_noise':
            noise = np.random.normal(0, 50, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        elif method == 'very_dark':
            img = cv2.convertScaleAbs(img, alpha=0.3, beta=0)
        
        elif method == 'overexposed':
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=50)
        
        output_path = unknown_path / f"unknown_{idx:04d}.jpg"
        cv2.imwrite(str(output_path), img)
    
    print(f"\nGenerated {target_count} 'unknown' images")
    print("="*60)
    print("UNKNOWN CLASS COMPLETE!")
    print("="*60 + "\n")


def main():
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

if __name__ == "__main__":
    main()