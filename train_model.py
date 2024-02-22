import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from modify_cnn_model import modify_pretrained_model
import logging
import traceback

logging.basicConfig(level=logging.INFO)

INPUT_DIR = 'C:/TreeImagesOut'
LABELS_FILE = 'C:/TreeImagesOut/labels.csv'
BATCH_SIZE = 32
EPOCHS = 50

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

    model = modify_pretrained_model(model_name='ResNet50')
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=len(validate_df) // BATCH_SIZE
    )
    model.save('palm_tree_counting_model.h5')
    logging.info('Model training completed and saved.')
except Exception as e:
    logging.error('Failed to complete model training:\n' + traceback.format_exc())