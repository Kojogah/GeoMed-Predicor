from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time
import cv2

app = Flask(__name__)

# Loading trained model
model_path = 'C:/Users/KOJOGAH/Desktop/school/FINAL YEAR PROJECT/sa-pso-vgg19_model.h5'
model = load_model(model_path)

@app.route('/', methods=['GET'])
def index():
    # Rendering the upload HTML page
    return render_template('GeoMed.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image-upload' not in request.files:
        return 'No file part', 400
    file = request.files['image-upload']
    if file.filename == '':
        return 'No selected file', 400
    
    # Read the image file
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (224, 224))  # Resize to match the model's expected input
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Model expects images in batches, so we add an extra dimension
    
    # Make prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction, axis=1)
    class_labels = {0: 'Normal', 1: 'Glioma Tumor', 2: 'Meningioma Tumor', 3: 'Pituitary Tumor'}
    result = class_labels[class_index[0]]
    
    return result

if __name__ == '__main__':
    app.run(debug=True)
