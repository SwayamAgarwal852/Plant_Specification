# Plant Species Identification using CNN
# Author: Ctrl+Alt+Win (Team)

# certificate
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import warnings

# Silence TensorFlow logs (INFO & WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ignore general warnings
warnings.filterwarnings("ignore")


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ==========================
# 1. DATA LOADING
# ==========================
# Change this path to your dataset folder
DATA_DIR = "data"

# Image preprocessing & augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    validation_split=0.2, #split in testing and training
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Training data
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

# Validation data
val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)




# ==========================
# 2. CNN MODEL
# ==========================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax")  # Output layer
])

# ==========================
# 3. COMPILE MODEL
# ==========================
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

# ==========================
# 4. TRAIN MODEL
# ==========================
history = model.fit(
    train_generator,
    epochs=3,   # Increase if you have time
    validation_data=val_generator
)

# ==========================
# 4b. SAVE THE MODEL
# ==========================
model.save("plant_model.h5")
print("Model saved as plant_model.h5")

# ==========================
# 5. EVALUATE MODEL
# ==========================
loss, acc = model.evaluate(train_generator)
print(f"Test Accuracy: {acc*100:.2f}%")

# ==========================
# 6. VISUALIZE TRAINING
# ==========================
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.legend()
plt.show()

# ==========================
# 7. PREDICTION EXAMPLE
# ==========================
import numpy as np
from tensorflow.keras.preprocessing import image

sample_img = "dataset/test/some_class/sample.jpg"  # change path
img = image.load_img(sample_img, target_size=(128,128))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
pred_class = np.argmax(prediction)
class_labels = list(train_generator.class_indices.keys())

print("Predicted Plant Species:", class_labels[pred_class])


