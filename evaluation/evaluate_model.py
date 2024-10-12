import numpy as np
import tensorflow as tf
from models.generator import build_generator
from models.ids_model import build_ids_model

# Load the trained IDS model and generator
ids_model = tf.keras.models.load_model('models/ids_model.h5')
generator = tf.keras.models.load_model('models/generator.h5')

# Load test data
test_data = np.loadtxt('data/test_data.csv', delimiter=',')
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

# Evaluate the IDS on normal test data
_, baseline_acc = ids_model.evaluate(X_test, y_test)
print(f"Baseline accuracy on normal test data: {baseline_acc}")

# Generate adversarial samples using the GAN
latent_dim = 100
noise = np.random.normal(0, 1, (X_test.shape[0], latent_dim))
adversarial_samples = generator.predict(noise)

# Evaluate the IDS on adversarial samples
adversarial_labels = np.zeros((X_test.shape[0],))  # All adversarial samples are attacks
_, adversarial_acc = ids_model.evaluate(adversarial_samples, adversarial_labels)
print(f"Accuracy on adversarial samples: {adversarial_acc}")

