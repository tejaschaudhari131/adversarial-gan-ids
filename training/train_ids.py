import numpy as np
import os
import logging
from tensorflow.keras.utils import to_categorical
from models.ids_model import build_ids_model

logger = logging.getLogger(__name__)


def train_basic_ids(train_path, test_path, output_path="models/ids_model.h5",
                    epochs=10, batch_size=32):
    """
    Train the basic IDS classifier.

    Args:
        train_path: Path to training data CSV (features + label in last column).
        test_path: Path to test data CSV.
        output_path: Where to save the trained model.
        epochs: Number of training epochs.
        batch_size: Training batch size.

    Returns:
        Training history object.
    """
    for path in (train_path, test_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading training data from %s", train_path)
    train_data = np.loadtxt(train_path, delimiter=',')
    test_data = np.loadtxt(test_path, delimiter=',')

    # Extract features and labels
    X_train = train_data[:, :-1]
    y_train = to_categorical(train_data[:, -1])

    X_test = test_data[:, :-1]
    y_test = to_categorical(test_data[:, -1])

    # Build and compile the IDS model
    ids_model = build_ids_model(input_dim=X_train.shape[1], num_classes=y_train.shape[1])
    ids_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    logger.info("Training basic IDS model (%d samples, %d features)",
                X_train.shape[0], X_train.shape[1])

    history = ids_model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ids_model.save(output_path)
    logger.info("Model saved to %s", output_path)

    return history


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train basic IDS model")
    parser.add_argument("--train-data", required=True, help="Path to training CSV")
    parser.add_argument("--test-data", required=True, help="Path to test CSV")
    parser.add_argument("--output", default="models/ids_model.h5", help="Model output path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_basic_ids(args.train_data, args.test_data, args.output, args.epochs, args.batch_size)
