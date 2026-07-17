import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
import os
import cv2
import pandas as pd
import numpy as np

# ============================
# Read CSV File
# ============================

csv_path = "dataset/train/_classes.csv"
data = pd.read_csv(csv_path)

# ============================
# Create Empty Lists
# ============================

images = []
labels = []

# ============================
# Read Every Image
# ============================

for index, row in data.iterrows():

    image_name = row["filename"]

    image_path = os.path.join("dataset", "train", image_name)

    image = cv2.imread(image_path)

    if image is None:
        continue

    # Resize image
    image = cv2.resize(image, (224, 224))

    # Normalize image
    image = image / 255.0

    images.append(image)

    # Read label columns (ignore filename)
    label = row.iloc[1:].values.astype(np.float32)

    # Convert one-hot label to class index
    class_index = np.argmax(label)

    labels.append(class_index)

# Convert to NumPy arrays
images = np.array(images)
labels = np.array(labels)

print("Total Images :", len(images))
print("Images Shape :", images.shape)
print("Labels Shape :", labels.shape)

print("\nFirst 10 Labels:")
print(labels[:10])

# ============================
# Build CNN Model
# ============================

model = Sequential()

# First Convolution Block
model.add(Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Second Convolution Block
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Third Convolution Block
model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# Convert Feature Maps to a Vector
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation="relu"))

# Reduce Overfitting
model.add(Dropout(0.5))

# Output Layer
num_classes = len(data.columns) - 1
model.add(Dense(num_classes, activation="softmax"))

# Show Model Summary
model.summary()
# ============================
# Compile the CNN Model
# ============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n✅ Model compiled successfully!")
# ============================
# Train the CNN Model
# ============================

history = model.fit(
    images,
    labels,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    shuffle=True
)
# ============================
# Save the Trained Model
# ============================

model.save("models/fetal_brain_cnn.keras")

print("\n✅ Model saved successfully!")