import os
import shutil
import random

# Set the directory paths
fer2013_dir = 'fer2013-augmented/train'
lbp_dir = 'lbp'
output_dirs = {
    'train': ('fer2013_split/train', 'lbp_split/train'),
    'validation': ('fer2013_split/validation', 'lbp_split/validation'),
    'test': ('fer2013_split/test', 'lbp_split/test')
}

# Create output directories for each split
for split in output_dirs.values():
    os.makedirs(split[0], exist_ok=True)  # FER2013 split
    os.makedirs(split[1], exist_ok=True)  # LBP split

# Set the split ratios
train_ratio, val_ratio, test_ratio = 0.2, 0.2, 0.6

# Process each class
for class_name in os.listdir(fer2013_dir):
    class_dir = os.path.join(fer2013_dir, class_name)
    lbp_class_dir = os.path.join(lbp_dir, class_name)

    # Skip if not a directory
    if not os.path.isdir(class_dir):
        continue

    # List all images in the class directory
    images = os.listdir(class_dir)
    random.shuffle(images)  # Shuffle for random splitting

    # Calculate split indices
    total_images = len(images)
    train_idx = int(total_images * train_ratio)
    val_idx = train_idx + int(total_images * val_ratio)

    # Split images for each set
    train_images = images[:train_idx]
    val_images = images[train_idx:val_idx]
    test_images = images[val_idx:]

    # Define a helper function to move images
    def move_images(image_list, split_name):
        fer_output_dir, lbp_output_dir = output_dirs[split_name]
        fer_class_output = os.path.join(fer_output_dir, class_name)
        lbp_class_output = os.path.join(lbp_output_dir, class_name)
        os.makedirs(fer_class_output, exist_ok=True)
        os.makedirs(lbp_class_output, exist_ok=True)

        for img_name in image_list:
            # Move image in fer2013
            src_img_path = os.path.join(class_dir, img_name)
            dst_img_path = os.path.join(fer_class_output, img_name)
            shutil.copy2(src_img_path, dst_img_path)

            # Move corresponding LBP feature in lbp
            lbp_name = img_name.replace('.jpg', '.npy')  # Assuming .npy for LBP files
            src_lbp_path = os.path.join(lbp_class_dir, lbp_name)
            dst_lbp_path = os.path.join(lbp_class_output, lbp_name)
            if os.path.exists(src_lbp_path):
                shutil.copy2(src_lbp_path, dst_lbp_path)

    # Move the images and their LBP features
    move_images(train_images, 'train')
    move_images(val_images, 'validation')
    move_images(test_images, 'test')

print("Splitting completed.")