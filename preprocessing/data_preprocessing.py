import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os

logger = logging.getLogger(__name__)


def load_cicids2017_dataset(file_path):
    """
    Load and preprocess the CIC-IDS-2017 dataset.

    Args:
        file_path: Path to the CSV dataset file.

    Returns:
        Tuple of (scaled_features, encoded_labels, label_encoder).

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or missing the 'Label' column.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {file_path}")

    if 'Label' not in df.columns:
        # Try stripping whitespace from column names (common in CIC-IDS2017)
        df.columns = df.columns.str.strip()
        if 'Label' not in df.columns:
            raise ValueError(
                f"'Label' column not found in dataset. Available columns: {list(df.columns)}"
            )

    # Handle missing and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        logger.info("Dropped %d rows with missing/infinite values (%.1f%%)",
                     rows_dropped, 100 * rows_dropped / rows_before)

    # Separate features and labels
    X = df.drop(['Label'], axis=1)
    y = df['Label']

    # Drop non-numeric columns from features
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.info("Dropping non-numeric feature columns: %s", non_numeric)
        X = X.select_dtypes(include=[np.number])

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    logger.info("Classes found: %s", list(label_encoder.classes_))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, label_encoder


def prepare_dataset(file_path, test_size=0.2, random_state=42):
    """
    Prepare dataset for model training with stratified train/test split.

    Args:
        file_path: Path to the CSV dataset file.
        test_size: Fraction of data to reserve for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with X_train, X_test, y_train, y_test, and label_encoder.
    """
    X, y, label_encoder = load_cicids2017_dataset(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info("Training set: %d samples, Test set: %d samples", len(X_train), len(X_test))

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Preprocess CIC-IDS2017 dataset")
    parser.add_argument("--dataset", required=True, help="Path to CIC-IDS2017 CSV dataset")
    args = parser.parse_args()

    prepared_data = prepare_dataset(args.dataset)
    print(f"Dataset prepared: {prepared_data['X_train'].shape[0]} train, "
          f"{prepared_data['X_test'].shape[0]} test samples")
    print(f"Features: {prepared_data['X_train'].shape[1]}")
    print(f"Classes: {list(prepared_data['label_encoder'].classes_)}")
