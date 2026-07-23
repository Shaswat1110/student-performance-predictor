# Student Performance Prediction System 🎓

An end-to-end machine learning project predicting student academic performance based on demographic, social, and academic factors. This project demonstrates data preprocessing, feature engineering, model training and evaluation using various regression algorithms, and a web deployment using Streamlit.

## 📂 Folder Structure

```text
Student-Performance-Prediction/
├── data/                       # Contains raw and cleaned CSV datasets
├── models/                     # Saved preprocessor and best trained model (.pkl)
├── notebooks/                  # Jupyter notebook for Exploratory Data Analysis (EDA)
├── outputs/plots/              # Generated visualizations (Actual vs Predicted, Feature Importance, etc.)
├── src/                        # Source code for the ML pipeline
│   ├── data_preprocessing.py   # Handles missing values and duplicates
│   ├── feature_engineering.py  # Applies One-Hot Encoding, Scaling, and Train/Test Split
│   ├── train.py                # Trains Regression models and selects the best one
│   └── evaluate.py             # Evaluates best model on test data and plots metrics
├── app.py                      # Streamlit application for interactive predictions
├── main.py                     # Master script to run the entire ML pipeline
├── download_dataset.py         # Script to download UCI dataset
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Tech Stack

- **Language:** Python 3.12+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Web App:** Streamlit

## 🚀 Installation & Setup

1. **Navigate to Project Directory:**
   ```bash
   cd Student-Performance-Prediction
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Dataset:**
   ```bash
   python download_dataset.py
   ```
   *This automatically pulls the Student Performance dataset from the UCI Machine Learning Repository.*

## 🧠 Running the ML Pipeline

To automatically preprocess the data, engineer features, train multiple models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost), evaluate the best one, and save the plots, simply run:

```bash
python main.py
```

Check the `outputs/plots/` folder to view:
- Feature Importance
- Actual vs Predicted Score
- Residual Distribution

## 🌐 Running the Streamlit App

Once the pipeline has run and generated the models, you can launch the interactive web dashboard:

```bash
streamlit run app.py
```

The app allows you to tweak student parameters (like study time, internet access, absences) to see how they impact the predicted final score in real-time.

## 📈 Future Improvements
- Implement hyperparameter tuning (GridSearchCV/RandomizedSearchCV) for the chosen model.
- Include deep learning (TensorFlow/Keras) as an additional comparative model.
- Deploy the Streamlit app to Streamlit Community Cloud or Heroku.
