import tensorflow as tf
from preprocessing.data_preprocessing import prepare_dataset
from models.advanced_ids_model import build_advanced_ids_model, compile_ids_model

def train_ids_model(dataset_path):
    # Prepare dataset
    prepared_data = prepare_dataset(dataset_path)
    
    X_train = prepared_data['X_train']
    X_test = prepared_data['X_test']
    y_train = prepared_data['y_train']
    y_test = prepared_data['y_test']
    label_encoder = prepared_data['label_encoder']
    
    # Get number of classes
    num_classes = len(label_encoder.classes_)
    input_dim = X_train.shape[1]
    
    # Build and compile model
    ids_model = build_advanced_ids_model(input_dim, num_classes)
    ids_model = compile_ids_model(ids_model)
    
    # Early stopping and model checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_ids_model.h5', 
        save_best_only=True
    )
    
    # Train the model
    history = ids_model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Evaluate the model
    test_results = ids_model.evaluate(X_test, y_test)
    
    return {
        'model': ids_model,
        'history': history,
        'test_results': test_results
    }

# Main execution
if __name__ == "__main__":
    dataset_path = "path/to/cicids2017_dataset.csv"
    training_results = train_ids_model(dataset_path)
    print("Model Training Completed!")
