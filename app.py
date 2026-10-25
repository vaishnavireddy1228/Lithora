
from flask import Flask, render_template, request, url_for
import cv2
import numpy as np
import os
import uuid
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
MODEL_PATH = os.path.join("models", "lithora_model.keras")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the original trained model once
model = tf.keras.models.load_model(MODEL_PATH)

IMAGE_SIZE = (128, 128)

CLASS_NAMES = {
    0: "No stone detected",
    1: "Stone detected"
}


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyzer")
def upload_page():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "scan" not in request.files:
        return "No file was uploaded.", 400

    file = request.files["scan"]

    if file.filename == "":
        return "Please select a file.", 400

    file_bytes = file.read()

    image_array = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return "Invalid image. Please upload a PNG or JPEG image.", 400

    height, width = image.shape[:2]

    # Save uploaded image
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    if not cv2.imwrite(image_path, image):
        return "Could not save the uploaded image.", 500

    # Preprocess image exactly as during training
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, IMAGE_SIZE)

    model_input = resized_image.astype(np.float32) / 255.0
    model_input = np.expand_dims(model_input, axis=(0, -1))

    # Predict using the original model
    probability = float(
        model.predict(model_input, verbose=0)[0][0]
    )

    predicted_class = 1 if probability >= 0.4 else 0
    prediction = CLASS_NAMES[predicted_class]

    return render_template(
        "results.html",
        image_url=url_for(
            "static",
            filename=f"uploads/{filename}"
        ),
        width=width,
        height=height,
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
