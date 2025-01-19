from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
import os
from keras.models import model_from_json
from skimage.feature import local_binary_pattern
from skimage import img_as_ubyte
import numpy as np

def compute_lbp(image):
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)
        image = img_as_ubyte(image)
    else:
        image = img_as_ubyte(image)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp

# Create lbp_generator to yield both images and LBP features
def lbp_generator(image_generator, lbp_dir):
    while True:
        batch_x, batch_y = next(image_generator)
        batch_lbp = []

        for i in range(batch_x.shape[0]):
            img_path = image_generator.filenames[i]
            emotion = img_path.split(os.sep)[-2]  # Emotion category
            img_index = img_path.split(os.sep)[-1].split('_')[-1].split('.')[0]  # Image ID
            lbp_path = os.path.join(lbp_dir, emotion, f"{emotion}_{img_index}.npy")

            if os.path.exists(lbp_path):
                lbp_features = np.load(lbp_path)
            else:
                lbp_features = compute_lbp(batch_x[i].squeeze())
                np.save(lbp_path, lbp_features)

            #reshape to (26,48)
            lbp_f = lbp_features[:26, :]
            batch_lbp.append(lbp_f)

        batch_lbp = np.array(batch_lbp)
        yield ((batch_x, batch_lbp), batch_y)

# Prepare training and validation generators
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

lbp_dir='lbp'
train_generator = lbp_generator(train_data_gen.flow_from_directory(
    'fer2013_split/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
), lbp_dir)

validation_generator = lbp_generator(validation_data_gen.flow_from_directory(
    'fer2013_split/validation',
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical',
    classes=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
),lbp_dir)
