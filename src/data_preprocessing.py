import pandas as pd
import numpy as np
import os

def load_and_clean_data(input_path: str, output_path: str):
    """
    Loads raw data, performs basic cleaning, handles missing values/duplicates,
    and saves the cleaned data.
    """
    print(f"Loading data from {input_path}...")
    # The UCI dataset uses ';' as a separator
    df = pd.read_csv(input_path, sep=';')
    
    print(f"Initial shape: {df.shape}")
    
    # 1. Missing value detection and handling
    missing_sum = df.isnull().sum().sum()
    print(f"Total missing values detected: {missing_sum}")
    if missing_sum > 0:
        # Since this is a structured dataset, we forward fill or drop based on threshold
        df = df.dropna()
        print(f"Shape after dropping missing values: {df.shape}")
        
    # 2. Duplicate removal
    duplicates = df.duplicated().sum()
    print(f"Duplicates detected: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    INPUT_FILE = "../data/student-mat.csv"
    OUTPUT_FILE = "../data/student_clean.csv"
    
    # Adjust paths if running from root directory
    if not os.path.exists(INPUT_FILE) and os.path.exists("data/student-mat.csv"):
        INPUT_FILE = "data/student-mat.csv"
        OUTPUT_FILE = "data/student_clean.csv"
        
    load_and_clean_data(INPUT_FILE, OUTPUT_FILE)
