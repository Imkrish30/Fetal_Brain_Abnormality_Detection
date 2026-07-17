from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model only once when the app starts
model = tf.keras.models.load_model("models/fetal_brain_cnn.keras")

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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    image = request.files["image"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
    image.save(filepath)

    # Read uploaded image
    img = cv2.imread(filepath)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

        # Predict
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    disease = class_names[predicted_class]

    filename = image.filename

    return render_template(
        "result.html",
        disease=disease,
        confidence=confidence,
        filename=filename
    )
if __name__ == "__main__":
    app.run(debug=True)
