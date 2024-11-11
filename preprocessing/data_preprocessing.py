import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_cicids2017_dataset(file_path):
    """
    Load and preprocess CIC-IDS-2017 dataset
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df.dropna(inplace=True)
    
    # Separate features and labels
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, label_encoder

def prepare_dataset(file_path, test_size=0.2, random_state=42):
    """
    Prepare dataset for model training
    """
    # Load and preprocess data
    X, y, label_encoder = load_cicids2017_dataset(file_path)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder
    }

# Example usage
if __name__ == "__main__":
    dataset_path = "path/to/cicids2017_dataset.csv"
    prepared_data = prepare_dataset(dataset_path)
    print("Dataset prepared successfully!")
