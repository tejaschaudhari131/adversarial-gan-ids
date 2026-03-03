import tensorflow as tf
import logging
from preprocessing.data_preprocessing import prepare_dataset
from models.advanced_ids_model import build_advanced_ids_model, compile_ids_model

logger = logging.getLogger(__name__)


def train_ids_model(dataset_path, epochs=50, batch_size=64, patience=10,
                    model_save_path="best_ids_model.h5"):
    """
    Train the advanced IDS model with early stopping and checkpointing.

    Args:
        dataset_path: Path to the CIC-IDS2017 CSV dataset.
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        patience: Early stopping patience (epochs without improvement).
        model_save_path: Path to save the best model checkpoint.

    Returns:
        Dictionary with trained model, history, and test results.
    """
    prepared_data = prepare_dataset(dataset_path)

    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    label_encoder = prepared_data['label_encoder']

    num_classes = len(label_encoder.classes_)
    input_dim = X_train.shape[1]

    logger.info("Building model: input_dim=%d, num_classes=%d", input_dim, num_classes)

    # Build and compile model
    ids_model = build_advanced_ids_model(input_dim, num_classes)
    ids_model = compile_ids_model(ids_model)

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_save_path,
        save_best_only=True,
    )

    # Train
    logger.info("Training for up to %d epochs (patience=%d)", epochs, patience)
    history = ids_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint],
    )

    # Evaluate
    test_results = ids_model.evaluate(X_test, y_test)
    logger.info("Test results: %s", test_results)

    return {
        'model': ids_model,
        'history': history,
        'test_results': test_results,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train advanced IDS model")
    parser.add_argument("--dataset", required=True, help="Path to CIC-IDS2017 CSV dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", default="best_ids_model.h5", help="Model save path")
    args = parser.parse_args()

    results = train_ids_model(args.dataset, args.epochs, args.batch_size,
                              args.patience, args.output)
    print("Model training completed!")
    print(f"Test results: {results['test_results']}")
