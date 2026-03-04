import tensorflow as tf
import numpy as np
import os
import logging
from models.generator import build_generator
from models.discriminator import build_discriminator

logger = logging.getLogger(__name__)


def train_gan(data_path, latent_dim=100, batch_size=64, epochs=10000,
              learning_rate=0.0002, log_interval=1000, output_dir="models"):
    """
    Train a GAN to generate adversarial network traffic.

    Args:
        data_path: Path to real network traffic CSV.
        latent_dim: Dimension of the generator's input noise vector.
        batch_size: Training batch size.
        epochs: Number of training epochs.
        learning_rate: Adam optimizer learning rate.
        log_interval: Print progress every N epochs.
        output_dir: Directory to save generator and discriminator models.

    Returns:
        Tuple of (generator, discriminator) trained models.
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Loading real network traffic from %s", data_path)
    real_data = np.loadtxt(data_path, delimiter=',')

    # Instantiate generator and discriminator
    generator = build_generator(input_dim=latent_dim, output_dim=real_data.shape[1])
    discriminator = build_discriminator(input_dim=real_data.shape[1])

    # Compile the discriminator
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    # Build and compile the GAN (generator + frozen discriminator)
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(latent_dim,))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
    )

    logger.info("Starting GAN training for %d epochs", epochs)

    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train discriminator with real data
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        real_samples = real_data[idx]
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)

        # Train discriminator with generated data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator(noise, training=False).numpy()
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

        # Train generator via GAN
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        if epoch % log_interval == 0:
            d_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])
            logger.info("Epoch %d/%d | D loss: %.4f | G loss: %.4f",
                        epoch, epochs, d_loss, g_loss)

    # Save trained models
    os.makedirs(output_dir, exist_ok=True)
    gen_path = os.path.join(output_dir, "generator.h5")
    disc_path = os.path.join(output_dir, "discriminator.h5")
    generator.save(gen_path)
    discriminator.save(disc_path)
    logger.info("Models saved to %s", output_dir)

    return generator, discriminator


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train GAN for adversarial traffic generation")
    parser.add_argument("--data", required=True, help="Path to real traffic CSV")
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    train_gan(
        args.data,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        log_interval=args.log_interval,
        output_dir=args.output_dir,
    )
