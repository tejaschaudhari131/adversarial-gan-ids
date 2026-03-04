import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import os
import logging

logger = logging.getLogger(__name__)


def evaluate_model_performance(model, X_test, y_test, label_encoder, output_dir="results"):
    """
    Comprehensive model evaluation with classification report, confusion matrix, and ROC curve.

    Args:
        model: Trained Keras model.
        X_test: Test features.
        y_test: True labels (integer-encoded).
        label_encoder: Fitted LabelEncoder for class names.
        output_dir: Directory to save evaluation plots.

    Returns:
        Dictionary with predictions, classification report text, and AUC score.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification Report
    report = classification_report(
        y_test,
        y_pred_classes,
        target_names=label_encoder.classes_
    )
    print("Classification Report:")
    print(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45, ha='right')
    plt.yticks(tick_marks, label_encoder.classes_)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", cm_path)

    results = {
        'y_pred_classes': y_pred_classes,
        'classification_report': report,
        'confusion_matrix': cm,
    }

    # ROC Curve (only for binary classification)
    if y_pred.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        logger.info("ROC curve saved to %s", roc_path)

        results['roc_auc'] = roc_auc
    else:
        logger.info("Skipping ROC curve — only supported for binary classification.")

    return results


if __name__ == "__main__":
    from preprocessing.data_preprocessing import prepare_dataset
    from tensorflow.keras.models import load_model
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained IDS model")
    parser.add_argument("--dataset", required=True, help="Path to CIC-IDS2017 CSV dataset")
    parser.add_argument("--model", default="best_ids_model.h5", help="Path to trained model file")
    parser.add_argument("--output-dir", default="results", help="Directory to save plots")
    args = parser.parse_args()

    prepared_data = prepare_dataset(args.dataset)
    trained_model = load_model(args.model)

    evaluate_model_performance(
        trained_model,
        prepared_data['X_test'],
        prepared_data['y_test'],
        prepared_data['label_encoder'],
        output_dir=args.output_dir,
    )
