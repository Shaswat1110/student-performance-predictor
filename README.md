# 🎓 Student Performance Prediction System

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)

An end-to-end Machine Learning project that predicts a student's final academic performance based on their demographic, social, and lifestyle background. 

This repository demonstrates the complete data science lifecycle: data preprocessing, feature engineering, training multiple regression algorithms, evaluating metrics, and deploying an interactive web application.

---

## ✨ Features

- **Automated ML Pipeline:** A single `main.py` script executes the entire pipeline (cleaning, engineering, training, evaluating, and plotting).
- **Multiple Algorithms:** Automatically trains and evaluates **Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost**.
- **Interactive Web Dashboard:** Built with Streamlit, the app features a completely live-updating UI where tweaking sliders (like Study Time or Absences) recalculates predictions instantly.
- **Engine Selection:** Users can dynamically switch between different ML models directly in the web UI to observe how different algorithms (like trees vs. linear models) interpret the exact same student data.

## 🧠 Interesting ML Insights (Outlier Distortion)
While testing this dataset, an interesting real-world ML pitfall emerged: **Outlier Distortion in Linear Regression**. 
Because the UCI dataset contains real students, a few "genius" outliers skipped dozens of classes but still passed. Because Linear Regression tries to draw a single straight line through the data, these outliers artificially skewed the slope upwards, causing the model to incorrectly assume *more absences = higher scores*. 
Switching the web app engine to a **Gradient Boosting** or **Decision Tree** algorithm fixes this, as tree-based models handle extreme outliers much more logically!

## 📂 Repository Structure

```text
Student-Performance-Prediction/
├── data/                       # Contains raw and cleaned CSV datasets
├── models/                     # Saved preprocessor and trained models (.pkl)
├── notebooks/                  # Jupyter notebook for Exploratory Data Analysis (EDA)
├── outputs/plots/              # Generated visualizations (Feature Importance, etc.)
├── src/                        # Source code for the ML pipeline
│   ├── data_preprocessing.py   # Handles missing values and duplicates
│   ├── feature_engineering.py  # Applies One-Hot Encoding, Scaling, and Train/Test Split
│   ├── train.py                # Trains multiple models and saves them to disk
│   └── evaluate.py             # Evaluates best model on test data and plots metrics
├── app.py                      # Live-updating Streamlit web application
├── main.py                     # Master script to run the entire pipeline
├── download_dataset.py         # Script to automatically download the UCI dataset
└── requirements.txt            # Python dependencies
```

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Dataset:**
   ```bash
   python download_dataset.py
   ```
   *(This automatically pulls the Student Performance dataset from the UCI Machine Learning Repository.)*

3. **Run the ML Pipeline:**
   ```bash
   python main.py
   ```
   *(This will clean the data, train all 5 machine learning models, and generate evaluation plots in the `outputs/plots` folder).*

4. **Launch the Web Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *(Once running, open your browser and tweak the lifestyle sliders to see live predictions!)*
