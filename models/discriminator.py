import tensorflow as tf
from tensorflow.keras import layers

def build_discriminator(input_dim=128):
    model = tf.keras.Sequential()
    
    # Input: synthetic or real network traffic
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Fully connected layers
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # Output: probability that the input is real or adversarial
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

if __name__ == "__main__":
    discriminator = build_discriminator()
    discriminator.summary()
