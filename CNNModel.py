from BulidingCNN import model
from PreprocessingData import train_generator,validation_generator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2


final_output = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)  # Added regularization
final_output = Dense(7, activation='softmax')(final_output) # Added back final classification layer


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)


total_train_samples = sum(len(files) for _, _, files in os.walk("fer2013_split/test"))
batch_size = 64 
total_validation_samples = sum(len(files) for _, _, files in os.walk("fer2013_split/validation"))


emotion_model = model.fit(
    train_generator,
    steps_per_epoch=total_train_samples // batch_size,
    epochs=50,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=total_validation_samples // batch_size,
    callbacks=[early_stopping, lr_scheduler]
)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")
