import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from modify_cnn_model import modify_pretrained_model
from data_generators import get_data_generators
import logging

logging.basicConfig(level=logging.INFO)

EPOCHS = 50

def train_model():
    try:
        train_generator, validation_generator = get_data_generators()
        if not train_generator or not validation_generator:
            logging.error('Data generators were not initialized successfully.')
            return
        
        model = modify_pretrained_model(model_name='ResNet50')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001, mode='min', verbose=1)

        logging.info("Starting model training...")

        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=[early_stopping, reduce_lr]
        )

        model.save('palm_tree_counting_model.h5')
        logging.info('Model training completed and saved successfully.')
    except Exception as e:
        logging.error('Model training failed:\n' + str(e), exc_info=True)

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logging.error('Failed to complete model training: ' + str(e), exc_info=True)