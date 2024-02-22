import io
import csv
import zipfile  # Make sure this import is included
from flask import Response
from werkzeug.utils import secure_filename
from image_preprocessor import preprocess_image  # Assuming this is the correct import for preprocess_image
import logging
import traceback
from flask import jsonify  # Import jsonify to format JSON responses

def batch_predict(request, model):
    if 'zipfile' not in request.files:
        logging.error('No zip file found in the request')
        return jsonify({'error': 'No zip file found in the request'}), 400
    zip_file = request.files['zipfile']
    filename = secure_filename(zip_file.filename)
    
    try:
        predictions = []
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Assuming the ZIP file directly contains the images
            for image_name in zip_ref.namelist():
                with zip_ref.open(image_name) as image_file:
                    preprocessed_image = preprocess_image(io.BytesIO(image_file.read()))
                    if preprocessed_image is not None:
                        prediction = model.predict(preprocessed_image)
                        count = int(round(prediction[0][0]))
                        logging.info(f'Processed {image_name} with count {count}')
                        predictions.append([image_name, count])
                    else:
                        logging.warning(f'Preprocessing of image {image_name} failed.')
        
        # Generate CSV in memory
        si = io.StringIO()
        cw = csv.writer(si)
        cw.writerow(['Image Name', 'Palm Tree Count'])
        cw.writerows(predictions)
        logging.info('Predictions CSV file created successfully.')
        
        # After generating the CSV content in 'si' (StringIO object)
        si.seek(0)  # Reset cursor to the beginning

        # Create a response object with the CSV data, setting the MIME type and content disposition
        response = Response(si.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=predictions.csv'})
        return response
    
    except Exception as e:
        logging.error(f'Batch prediction failed: {e}\n{traceback.format_exc()}')
        return jsonify({'error': 'Failed to process batch prediction'}), 500
