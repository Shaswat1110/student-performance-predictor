# 🎓 Student Performance Prediction System

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Web%20App-Streamlit-red.svg)

An end-to-end Machine Learning project that predicts a student's final academic performance based on their demographic, social, and lifestyle background. 

This repository demonstrates a fundamental data science lifecycle: data cleaning, exploratory data analysis (EDA), feature encoding, feature scaling, training regression algorithms, and deploying an interactive web application. It is designed to be highly beginner-friendly, focusing on transparent, foundational data science concepts without relying on complex, black-box abstractions.

## 🌐 Live Application
You can try the live, interactive web dashboard here:
**[Student Performance Predictor](https://student-performance-predictor-ml1.streamlit.app/)**

---

## ✨ Core Concepts Implemented

- **Real-World Dataset:** Uses the UCI Student Performance dataset.
- **Foundational Preprocessing:** Manual, transparent implementation of handling missing values, standard feature scaling, and one-hot encoding without black-box `sklearn` Pipelines.
- **Classic Algorithms:** Trains and evaluates exactly three foundational models: **Linear Regression, Decision Tree, and Random Forest**.
- **Model Evaluation:** Compares models using standard regression metrics: MAE, MSE, RMSE, and R².
- **Interactive Web Dashboard:** Built with Streamlit, the app features a completely live-updating UI where tweaking sliders recalculates predictions instantly.
- **Engine Selection:** Users can dynamically switch between the three ML models directly in the web UI to observe how different algorithms (like trees vs. linear models) interpret the exact same student data.
- **Custom Profile Generation:** Dynamic state management allows users to build and save custom student profiles using all 30 parameters for immediate model evaluation.

## 🧠 Key Data Insights
During the development of this model, removing prior academic grades (`G1`, `G2`) forced the algorithm to predict final scores based strictly on demographics and lifestyle. This led to some fascinating real-world data insights:

1. **The Anchoring Effect**: We found that extreme behaviors (like 90+ absences) do not automatically crash a student's predicted score to 0. Because the model evaluates all 30 parameters holistically, if the other 29 parameters reflect a highly supported, average student, the model anchors the score near the mean.
2. **Counter-Intuitive Truths**: The absolute mathematically highest-scoring student profile in the dataset does *not* perfectly align with human intuition. The algorithm learned that students who receive **no** extra educational support or tutoring actually score the highest (likely because naturally gifted students don't seek out extra support). Furthermore, having exactly 0 absences was statistically less optimal than having a few (e.g., 4) absences!

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
   *(This will clean the data, train all 3 machine learning models, and generate evaluation plots in the `outputs/plots` folder).*

4. **Launch the Web Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *(Once running, open your browser and tweak the lifestyle sliders to see live predictions!)*
