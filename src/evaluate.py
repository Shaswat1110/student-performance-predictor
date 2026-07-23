import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(data_dir: str, model_dir: str, outputs_dir: str):
    """
    Evaluates the best trained model on the test set and generates performance plots.
    """
    print("Loading test data and best model...")
    X_test = joblib.load(os.path.join(data_dir, 'X_test.pkl'))
    y_test = joblib.load(os.path.join(data_dir, 'y_test.pkl'))
    feature_names = joblib.load(os.path.join(data_dir, 'feature_names.pkl'))
    
    best_model_path = os.path.join(model_dir, 'best_model.pkl')
    best_model = joblib.load(best_model_path)
    
    try:
        with open(os.path.join(model_dir, 'best_model_name.txt'), 'r') as f:
            model_name = f.read().strip()
    except FileNotFoundError:
        model_name = "Best Model"
        
    print(f"Evaluating {model_name}...")
    
    y_pred = best_model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("\nEvaluation Metrics:")
    print("-" * 25)
    print(f"MAE:  {mae:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")
    
    # Visualizations
    plots_dir = os.path.join(outputs_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Prediction vs Actual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='b')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Final Score')
    plt.ylabel('Predicted Final Score')
    plt.title(f'Actual vs Predicted Final Score ({model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # 2. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color='purple')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Final Score')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot ({model_name})')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residual_plot.png'))
    plt.close()
    
    # 3. Residual Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='green')
    plt.xlabel('Residual Error')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residual_distribution.png'))
    plt.close()
    
    # 4. Feature Importance (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]
        
        # Take top 15 features for clean plotting
        top_n = 15
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_importances, y=top_features, palette="viridis")
        plt.title('Top 15 Feature Importances')
        plt.xlabel('Relative Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
        plt.close()
        
    print(f"\nAll plots have been saved to {plots_dir}")
    print("Evaluation complete.")

if __name__ == "__main__":
    DATA_DIR = "../data/"
    MODEL_DIR = "../models/"
    OUTPUTS_DIR = "../outputs/"
    
    if not os.path.exists(DATA_DIR) and os.path.exists("data/X_train.pkl"):
        DATA_DIR = "data/"
        MODEL_DIR = "models/"
        OUTPUTS_DIR = "outputs/"
        
    evaluate_model(DATA_DIR, MODEL_DIR, OUTPUTS_DIR)
