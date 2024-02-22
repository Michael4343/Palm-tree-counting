from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
import os

INPUT_DIR = 'C:/TreeImagesOut'
LABELS_FILE = 'C:/TreeImagesOut/labels.csv'
BATCH_SIZE = 32

logging.basicConfig(level=logging.INFO)

def validate_labels_csv_structure():
    required_columns = ['filename', 'count']
    if not os.path.exists(LABELS_FILE):
        logging.error(f"The labels file {LABELS_FILE} does not exist.")
        return False
    try:
        df = pd.read_csv(LABELS_FILE)
        if not all(column in df.columns for column in required_columns):
            logging.error("The labels CSV does not have the required columns: 'filename' and 'count'.")
            return False
        return True
    except Exception as e:
        logging.error(f"Failed to validate labels CSV structure due to an error: {e}")
        return False

def get_data_generators():
    if not validate_labels_csv_structure():
        logging.error("Data generators initialization failed due to invalid labels CSV structure.")
        return None, None
    try:
        labels_df = pd.read_csv(LABELS_FILE)
        train_df, validate_df = train_test_split(labels_df, test_size=0.2, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop=True)

        train_datagen = ImageDataGenerator(rescale=1./255)
        validate_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_df,
            directory=INPUT_DIR,
            x_col='filename',
            y_col='count',
            target_size=(224, 224),
            class_mode='raw',
            batch_size=BATCH_SIZE
        )

        validation_generator = validate_datagen.flow_from_dataframe(
            dataframe=validate_df,
            directory=INPUT_DIR,
            x_col='filename',
            y_col='count',
            target_size=(224, 224),
            class_mode='raw',
            batch_size=BATCH_SIZE
        )

        logging.info('Data generators initialized successfully.')
        return train_generator, validation_generator
    except Exception as e:
        logging.error(f'Failed to initialize data generators:\n{e}', exc_info=True)
        return None, None