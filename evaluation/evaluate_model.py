import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)

def evaluate_model_performance(model, X_test, y_test, label_encoder):
    """
    Comprehensive model evaluation
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(
        y_test, 
        y_pred_classes, 
        target_names=label_encoder.classes_
    ))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10,7))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()
    
    # ROC Curve
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
    plt.show()

# Usage in main script
if __name__ == ```python
"__main__":
    from training.train_advanced_ids import train_ids_model
    from models.advanced_ids_model import build_advanced_ids_model
    from preprocessing.data_preprocessing import prepare_dataset

    dataset_path = "path/to/cicids2017_dataset.csv"
    prepared_data = prepare_dataset(dataset_path)

    X_test = prepared_data['X_test']
    y_test = prepared_data['y_test']
    label_encoder = prepared_data['label_encoder']

    # Load the trained model
    from tensorflow.keras.models import load_model
    trained_model = load_model('best_ids_model.h5')

    # Evaluate the model
    evaluate_model_performance(trained_model, X_test, y_test, label_encoder)
