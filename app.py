from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras.utils import load_img, img_to_array
import urllib.request
import numpy as np
import os
from io import BytesIO
import yaml

# Initialize Flask app
app = Flask(__name__)

# Load model info from YAML
with open("model_info.yaml", "r") as f:
    model_info = yaml.safe_load(f)
MODEL_URL = model_info["MODEL_URL"]
MODEL_PATH = model_info["MODEL_PATH"]

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model file...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
else:
    print("Found existing model file.")

# Load model once at startup
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    interpreter = None

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ["glioma", "meningioma", "notumor", "pituitary"]

# Default images for the gallery
DEFAULT_IMAGES = [
    "static/images/default_glioma.jpg",
    "static/images/default_meningioma.jpg",
    "static/images/default_notumor.jpg",
    "static/images/default_pituitary.jpg",
]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    # Pass the list of default images to the template
    return render_template("index.html", image_filenames=DEFAULT_IMAGES)

@app.route("/predict", methods=["POST"])
def predict():
    # Handle both new file uploads and gallery image predictions
    filename = request.form.get('filename')
    
    if filename:
        # Prediction from a gallery image
        filepath = os.path.join(app.root_path, filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Image not found."}), 404
        
        # Preprocess image from file path
        img = load_img(filepath, target_size=(224, 224))
        
    else:
        # New file upload
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "" or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Please upload a jpg, jpeg, or png image."}), 400
        
        # Read the image file into memory
        img_stream = BytesIO(file.read())
        
        # Preprocess image directly from the in-memory stream
        img = load_img(img_stream, target_size=(224, 224))
        
    if img and interpreter:
        try:
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            interpreter.set_tensor(input_details[0]["index"], img_array)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred_label = class_labels[np.argmax(output_data)]
            return jsonify({
                "prediction": pred_label,
            }), 200

        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({"error": "An error occurred during prediction."}), 500
    else:
        return jsonify({"error": "Model not loaded or file not provided."}), 500

if __name__ == "__main__":
    app.run(debug=False)
