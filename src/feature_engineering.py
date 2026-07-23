import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def engineer_features(input_path: str, output_dir: str):
    """
    Applies Label/One-Hot encoding, Feature Scaling, and Train/Test Split.
    Saves the processed arrays and the fitted preprocessor pipeline.
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
    
    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Fitting preprocessor and transforming data...")
    # Fit on training data and transform both
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding for feature importance plotting later
    try:
        # For newer scikit-learn versions
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(cat_feature_names)
    except AttributeError:
        # Fallback
        feature_names = numerical_cols + [f"cat_{i}" for i in range(X_train_processed.shape[1] - len(numerical_cols))]
    
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
