import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_and_evaluate_models(data_dir: str, model_dir: str):
    """
    Trains multiple regression models, compares them, and saves the best one.
    """
    print("Loading preprocessed data...")
    X_train = joblib.load(os.path.join(data_dir, 'X_train.pkl'))
    X_test = joblib.load(os.path.join(data_dir, 'X_test.pkl'))
    y_train = joblib.load(os.path.join(data_dir, 'y_train.pkl'))
    y_test = joblib.load(os.path.join(data_dir, 'y_test.pkl'))
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
        
    best_model_name = None
    best_model = None
    best_rmse = float('inf')
    
    os.makedirs(model_dir, exist_ok=True)
    
    print("\nTraining and evaluating models...")
    print("-" * 60)
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
    print("-" * 60)
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name:<20} | {mae:<10.4f} | {rmse:<10.4f} | {r2:<10.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model
            
        model_path = os.path.join(model_dir, f'{name.replace(" ", "_")}.pkl')
        joblib.dump(model, model_path)
            
    print("-" * 60)
    print(f"\nBest Model: {best_model_name}")
    
    best_model_path = os.path.join(model_dir, 'best_model.pkl')
    
    print(f"Saving {best_model_name} to {best_model_path}...")
    joblib.dump(best_model, best_model_path)
    
    # Save the name of the best model so evaluate script knows what it is
    with open(os.path.join(model_dir, 'best_model_name.txt'), 'w') as f:
        f.write(best_model_name)
        
    print("Training complete.")

if __name__ == "__main__":
    DATA_DIR = "../data/"
    MODEL_DIR = "../models/"
    
    if not os.path.exists(DATA_DIR) and os.path.exists("data/X_train.pkl"):
        DATA_DIR = "data/"
        MODEL_DIR = "models/"
        
    train_and_evaluate_models(DATA_DIR, MODEL_DIR)
