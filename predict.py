import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================
# Load the trained model
# ============================

model = tf.keras.models.load_model("models/fetal_brain_cnn.keras")

print("✅ Model loaded successfully!")

# ============================
# Image Path
# ============================

image_path = "dataset/test/Copy-of-holoprosencephaly-17a_aug_0_png_jpg.rf.ce2427430cacad8c07581cffb4e49deb.jpg"

# ============================
# Read Image
# ============================

image = cv2.imread(image_path)

if image is None:
    print("❌ Error: Image not found!")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to the same size used during training
image_resized = cv2.resize(image_rgb, (224, 224))

# Normalize pixel values
image_normalized = image_resized / 255.0

# Add batch dimension
image_input = np.expand_dims(image_normalized, axis=0)

print("✅ Image loaded and preprocessed successfully!")

# Display the image
plt.imshow(image_rgb)
plt.title("Test Image")
plt.axis("off")
plt.show()
# ============================
# Predict the Disease
# ============================

prediction = model.predict(image_input)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction) * 100

class_names = [
    "Arnold-Chiari Malformation",
    "Arachnoid Cyst",
    "Cerebellar Hypoplasia",
    "Colpocephaly",
    "Encephalocele",
    "Holoprosencephaly",
    "Hydranencephaly",
    "Intracranial Hemorrhage",
    "Intracranial Tumor",
    "Mega Cisterna Magna",
    "Mild Ventriculomegaly",
    "Moderate Ventriculomegaly",
    "Normal",
    "Porencephaly",
    "Severe Ventriculomegaly",
    "Vein of Galen Malformation"
]

print("\n" + "=" * 50)
print("      FETAL BRAIN ABNORMALITY PREDICTION")
print("=" * 50)
print(f"Predicted Disease : {class_names[predicted_class]}")
print(f"Confidence        : {confidence:.2f}%")
print("=" * 50)