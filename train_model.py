import pandas as pd
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Paths
CSV_PATH = "./Training_Set/RFMiD_Training_Labels.csv"
IMAGE_DIR = "./Training_Set/Training"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

# Load CSV
df = pd.read_csv(CSV_PATH)

# Keep only rows where image exists
df["image_path"] = df["ID"].astype(str) + ".png"
df["image_path"] = df["image_path"].apply(lambda x: os.path.join(IMAGE_DIR, x))
df = df[df["image_path"].apply(os.path.exists)]

# Extract data
image_paths = df["image_path"].values
labels = df["Disease_Risk"].values

# Image loader
def load_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label

# Dataset
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(dataset, epochs=EPOCHS)

# Save
model.save("rfmid_model.h5")
print("✅ Model saved as rfmid_model.h5")
