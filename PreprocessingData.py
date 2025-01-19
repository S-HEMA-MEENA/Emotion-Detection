import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import cv2
from concurrent.futures import ThreadPoolExecutor

# Define directories
input_dir = 'fer2013/train'
output_dir = 'fer2013-augmented/train'
lbp_dir = 'lbp'

# Set target counts for each emotion
target_count = {
    "angry": 8358,
    "disgust": 5196,
    "fear": 8560,
    "happy": 7215,
    "neutral": 9888,
    "sad": 9668,
    "surprise": 7071
}

# Image Data Generator for augmentation
data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(lbp_dir, exist_ok=True)

# Function to compute LBP features
def compute_lbp(image):
    image = img_as_ubyte(image)  # Converts to unsigned byte
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp

# Process images in parallel to copy and compute LBP features
def process_image(emotion, img_name, output_emotion_dir, lbp_dir):
    original_image_path = os.path.join(input_dir, emotion, img_name)
    img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Save original image to augmented directory
    destination_image_path = os.path.join(output_emotion_dir, img_name)
    shutil.copy2(original_image_path, destination_image_path)

    # Compute and save LBP features
    lbp_features = compute_lbp(img)
    lbp_features_path = os.path.join(lbp_dir, emotion, f"{emotion}_{img_name.split('.')[0]}.npy")
    np.save(lbp_features_path, lbp_features)

# Augment images up to the target count for each class
for emotion, target in target_count.items():
    output_emotion_dir = os.path.join(output_dir, emotion)
    os.makedirs(output_emotion_dir, exist_ok=True)
    os.makedirs(os.path.join(lbp_dir, emotion), exist_ok=True)

    images = os.listdir(os.path.join(input_dir, emotion))
    total_count = len(images)

    # Copy original images and compute LBP features
    with ThreadPoolExecutor(max_workers=4) as executor:
        for img_name in images:
            executor.submit(process_image, emotion, img_name, output_emotion_dir, lbp_dir)

    # Perform augmentation up to the target count
    current_img = 0
    while total_count < target:
        img_path = os.path.join(input_dir, emotion, images[current_img % len(images)])
        img = load_img(img_path, color_mode='grayscale')
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        augment_count = 0
        for batch in data_gen.flow(img_array, batch_size=1):
            # Apply weighted average between original and augmented image
            aug_img = 0.7 * img_array[0] + 0.3 * batch[0]
            aug_img = aug_img.squeeze()  # Remove extra dimension

            # Normalize for img_as_ubyte compatibility
            aug_img_normalized = aug_img / 255.0

            aug_img_path = os.path.join(output_emotion_dir, f"{emotion}_{total_count}.jpg")
            save_img(aug_img_path, np.expand_dims(aug_img, axis=-1))

            # Compute and save LBP features of the normalized weighted image
            lbp_features = compute_lbp(aug_img_normalized)
            lbp_features_path = os.path.join(lbp_dir, emotion, f"{emotion}_{total_count}.npy")
            np.save(lbp_features_path, lbp_features)

            total_count += 1
            augment_count += 1

            if augment_count >= 2:  # Generate only 2 weighted-augmented images per original image
                break

        current_img += 1
        if current_img >= len(images):  # Loop through images
            current_img = 0

print("Weighted data augmentation and LBP extraction completed for all classes.")
