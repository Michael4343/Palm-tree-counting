import io
from PIL import Image
import numpy as np

def preprocess_image(image_file, target_size=(224, 224)):
    image = Image.open(io.BytesIO(image_file.read()))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.asarray(image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)