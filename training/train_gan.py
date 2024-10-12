import tensorflow as tf
import numpy as np
from models.generator import build_generator
from models.discriminator import build_discriminator

# Hyperparameters
latent_dim = 100
batch_size = 64
epochs = 10000

# Load real network traffic data
real_data = np.loadtxt('data/train_data.csv', delimiter=',')

# Instantiate generator and discriminator
generator = build_generator(input_dim=latent_dim)
discriminator = build_discriminator(input_dim=real_data.shape[1])

# Compile the discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])

# Build and compile the GAN
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002), loss='binary_crossentropy')

# Training the GAN
for epoch in range(epochs):
    # Train the discriminator with real data
    real_labels = np.ones((batch_size, 1))
    idx = np.random.randint(0, real_data.shape[0], batch_size)
    real_samples = real_data[idx]
    
    d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    
    # Train the discriminator with fake (generated) data
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(noise)
    fake_labels = np.zeros((batch_size, 1))
    
    d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
    
    # Train the generator via the GAN
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | Discriminator Loss: {d_loss_real[0] + d_loss_fake[0]} | Generator Loss: {g_loss}")

# Save the models
generator.save('models/generator.h5')
discriminator.save('models/discriminator.h5')
