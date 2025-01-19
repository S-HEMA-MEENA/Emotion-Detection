import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disable oneDNN

import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, Concatenate, Reshape
from keras.optimizers import Adam
from skimage import feature

# Set TensorFlow environment variables to reduce logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: disable oneDNN

def ulbp(image):
    # Convert image to grayscale if it is not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute LBP (this function should be implemented or replaced with a valid one)
    # Example implementation of LBP
    radius = 1
    points = 8
    lbp = feature.local_binary_pattern(image, points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=56, range=(0, 56))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6  # Prevent division by zero

    return hist

# Define the CNN model
image_input = Input(shape=(48, 48, 1))  # Input for CNN

x = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x= Dense(1024, activation='relu')(x)  # CNN output

# LBP Input Layer
lbp_input = Input(shape=(26,48))  # Adjust based on LBP feature size
lbp_reshaped = Reshape((26 * 48,))(lbp_input)
y=Dense(1024, activation='relu')(lbp_reshaped)

# Concatenate CNN features with LBP features
combined = Concatenate()([x, y])

# Additional Dense Layers after concatenation
combined = Dense(512, activation='relu')(combined)
combined = Dropout(0.3)(combined)

combined = Dense(256, activation='relu')(combined)
combined = Dropout(0.3)(combined)

# Final Classification Layer
final_output = Dense(7, activation='softmax')(combined)

# Create the final model
model = Model(inputs=[image_input, lbp_input], outputs=final_output)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Print the model summary
model.summary()
