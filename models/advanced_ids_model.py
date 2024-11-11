import tensorflow as tf
from tensorflow.keras import layers, regularizers

def build_advanced_ids_model(input_dim, num_classes):
    """
    Build an advanced neural network for Intrusion Detection
    """
    model = tf.keras.Sequential([
        # Input layer with L2 regularization
        layers.Input(shape=(input_dim,)),
        
        # First dense layer with dropout and regularization
        layers.Dense(
            512, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Second dense layer
        layers.Dense(
            256, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Third dense layer
        layers.Dense(
            128, 
            activation='relu', 
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer with softmax activation
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_ids_model(model, learning_rate=0.001):
    """
    Compile the IDS model with optimal parameters
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(), 
                 tf.keras.metrics.Recall()]
    )
    
    return model
