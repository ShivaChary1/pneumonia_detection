from flask import Flask, request, render_template, redirect, url_for
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

# Initialize the Flask application
app = Flask(__name__)

# Allow CORS (Cross-Origin Resource Sharing)
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('./pneumonia_detection.h5')

def preprocess_image(image: Image.Image, target_size: tuple):
    """Preprocess the image to the target size required by the model."""
    image = image.convert('RGB')  # Ensure the image is in RGB format
    image = image.resize(target_size)  # Resize the image to the target size
    image = np.array(image)  # Convert the image to a numpy array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Rescale the image
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return render_template('upload.html', error="No file provided")

        # Load and preprocess the image
        image = Image.open(BytesIO(file.read()))
        processed_image = preprocess_image(image, target_size=(256, 256))

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Prepare the response
        class_labels = {0: 'NORMAL', 1: 'PNEUMONIA'}
        result = class_labels.get(predicted_class, 'Unknown')

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('upload.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
