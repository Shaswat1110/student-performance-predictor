import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def engineer_features(input_path: str, output_dir: str):
    """
    Applies One-Hot encoding, Feature Scaling, and Train/Test Split.
    Saves the processed arrays and the fitted preprocessors.
    """
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Target variable is G3 (Final Score)
    target_col = 'G3'
    
    X = df.drop(columns=[target_col, 'G1', 'G2'])
    y = df[target_col]
    
    # Identify numerical and categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting preprocessor and transforming data...")
    # Scale numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])
    
    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])
    
    # Combine features
    X_train_processed = np.hstack((X_train_num, X_train_cat))
    X_test_processed = np.hstack((X_test_num, X_test_cat))
    
    # Get feature names for plotting feature importance later
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)
    feature_names = numerical_cols + list(cat_feature_names)
    
    # Create a preprocessor dictionary to save
    preprocessor = {
        'scaler': scaler,
        'encoder': encoder,
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols
    }
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '../models'), exist_ok=True)
    
    # Save the processed data
    print("Saving processed datasets...")
    joblib.dump(X_train_processed, os.path.join(output_dir, 'X_train.pkl'))
    joblib.dump(X_test_processed, os.path.join(output_dir, 'X_test.pkl'))
    joblib.dump(y_train, os.path.join(output_dir, 'y_train.pkl'))
    joblib.dump(y_test, os.path.join(output_dir, 'y_test.pkl'))
    joblib.dump(feature_names, os.path.join(output_dir, 'feature_names.pkl'))
    
    # Save the preprocessor pipeline to apply to new incoming data (e.g. from Streamlit)
    preprocessor_path = os.path.join(output_dir, '../models/preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    print("Feature engineering complete.")

if __name__ == "__main__":
    INPUT_FILE = "../data/student_clean.csv"
    OUTPUT_DIR = "../data/"
    
    if not os.path.exists(INPUT_FILE) and os.path.exists("data/student_clean.csv"):
        INPUT_FILE = "data/student_clean.csv"
        OUTPUT_DIR = "data/"
        
    engineer_features(INPUT_FILE, OUTPUT_DIR)
