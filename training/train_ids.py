import numpy as np
from tensorflow.keras.utils import to_categorical
from models.ids_model import build_ids_model

# Load dataset
train_data = np.loadtxt('data/train_data.csv', delimiter=',')
test_data = np.loadtxt('data/test_data.csv', delimiter=',')

# Extract features and labels
X_train = train_data[:, :-1]
y_train = to_categorical(train_data[:, -1])

X_test = test_data[:, :-1]
y_test = to_categorical(test_data[:, -1])

# Build the IDS model
ids_model = build_ids_model(input_dim=X_train.shape[1])

# Compile the model
ids_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
ids_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained IDS model
ids_model.save('models/ids_model.h5')
