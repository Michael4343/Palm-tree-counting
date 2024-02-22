import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

def modify_pretrained_model(model_name='ResNet50', num_classes=1):
    input_shape = (224, 224, 3)
    if model_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    elif model_name == 'EfficientNetB0':
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    else:
        raise ValueError("Unsupported model name. Please choose either 'ResNet50' or 'EfficientNetB0'.")
    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(), loss='mean_squared_error', metrics=['mean_squared_error'])
    return model

if __name__ == "__main__":
    model_name = 'ResNet50'
    model = modify_pretrained_model(model_name=model_name)
    model.summary()