import tensorflow as tf
from tensorflow.keras import layers

def build_generator(input_dim=100, output_dim=128):
    model = tf.keras.Sequential()
    
    # Input: random noise (latent space)
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Fully connected layers
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    # Output: synthetic network traffic data
    model.add(layers.Dense(output_dim, activation='tanh'))

    return model

if __name__ == "__main__":
    generator = build_generator()
    generator.summary()
