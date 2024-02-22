import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3
INPUT_DIR = 'C:/TreeImages' # INPUT_REQUIRED: specify the path to your input images
OUTPUT_DIR = 'C:/TreeImagesOut' # INPUT_REQUIRED: specify the path to save output images
AUGMENTATION_MULTIPLIER = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'augmented'), exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def preprocess_image(file_path, output_path, augment=False):
    try:
        img = Image.open(file_path)
    except Exception as e:
        print(f'Error opening image: {file_path}. Error: {e}')
        return
    img_resized = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    if augment:
        img_array = img_to_array(img_resized)
        img_array = img_array.reshape((1,) + img_array.shape)
        i = 0
        for batch in datagen.flow(img_array, batch_size=1, save_to_dir=output_path, save_prefix='aug_', save_format='jpeg'):
            i += 1
            if i >= AUGMENTATION_MULTIPLIER:
                break
    else:
        img_resized.save(output_path)
        print(f'Saved resized image to: {output_path}')

def main():
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(INPUT_DIR, filename)
            output_path = os.path.join(OUTPUT_DIR, filename)
            preprocess_image(file_path, output_path)
            augment_output_path = os.path.join(OUTPUT_DIR, 'augmented')
            preprocess_image(file_path, augment_output_path, augment=True)
            print(f'Processed and augmented: {filename}')

if __name__ == "__main__":
    main()
