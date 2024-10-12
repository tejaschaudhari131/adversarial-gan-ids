import tensorflow as tf
from tensorflow.keras import layers

def build_ids_model(input_dim=128, num_classes=2):
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Fully connected layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    
    # Output layer for binary classification (normal or attack)
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

if __name__ == "__main__":
    ids_model = build_ids_model()
    ids_model.summary()
