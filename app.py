from flask import Flask, request, jsonify
import traceback
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import logging
from werkzeug.utils import secure_filename
from batch_predictor import batch_predict as batch_predict_handler

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the model
try:
    model = load_model('palm_tree_counting_model.h5')
    logging.info('Model loaded successfully.')
except Exception as e:
    logging.error(f'Failed to load model: {e}\n{traceback.format_exc()}')

def preprocess_image(input_image):
    try:
        if isinstance(input_image, str):
            img = Image.open(input_image).convert('RGB')
        else:
            img = Image.open(input_image.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        logging.error(f'Image preprocessing failed: {e}\n{traceback.format_exc()}')
        return None

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            logging.error('No image file found in the request')
            return jsonify({'error': 'No image file found in the request'}), 400
        image_file = request.files['image']
        preprocessed_image = preprocess_image(image_file)
        if preprocessed_image is not None:
            prediction = model.predict(preprocessed_image)
            count = int(round(prediction[0][0]))
            logging.info(f'Prediction made successfully: {count} palm trees')
            return jsonify({'count': count})
        else:
            return jsonify({'error': 'Error processing the image'}), 500
    except Exception as e:
        logging.error(f'Unexpected error occurred: {e}\n{traceback.format_exc()}')
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    try:
        logging.info("Received batch prediction request.")
        response = batch_predict_handler(request, model)
        logging.info("Batch prediction request processed successfully.")
        return response
    except Exception as e:
        logging.error(f"Error processing batch prediction request: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Failed to process batch prediction request'}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)