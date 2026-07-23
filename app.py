import streamlit as st
import pandas as pd
import joblib
import os

# Set page config
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Predictor")
st.markdown("""
This application predicts a student's final score (G3) based on their demographic, social, and academic background. 
It uses a machine learning model trained on the UCI Student Performance dataset.
""")

# Load model and preprocessor
@st.cache_resource
def load_assets(selected_model_option):
    model_dir = "models"
    
    preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.pkl'))
    
    if selected_model_option == "Best Automatically Selected Model":
        best_model = joblib.load(os.path.join(model_dir, 'best_model.pkl'))
        with open(os.path.join(model_dir, 'best_model_name.txt'), 'r') as f:
            actual_name = f.read().strip()
        return preprocessor, best_model, actual_name
    else:
        model_filename = selected_model_option.replace(" ", "_") + ".pkl"
        model = joblib.load(os.path.join(model_dir, model_filename))
        return preprocessor, model, selected_model_option

st.sidebar.markdown("### ⚙️ Engine Settings")
selected_model_option = st.sidebar.selectbox(
    "Choose ML Algorithm",
    ["Best Automatically Selected Model", "Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]
)

try:
    preprocessor, model, model_name = load_assets(selected_model_option)
    st.sidebar.success(f"Active Model: {model_name}")
except Exception as e:
    st.error(f"Error loading models. Have you trained them yet? \n({e})")
    st.stop()

st.subheader("Student Lifestyle Tweaks")
    
col1, col2 = st.columns(2)

with col1:
    studytime = st.selectbox("Study Time", options=[1, 2, 3, 4], format_func=lambda x: ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"][x-1], index=1)
    absences = st.slider("Number of Absences", min_value=0, max_value=93, value=4)
    failures = st.number_input("Past Class Failures", min_value=0, max_value=4, value=0)
    activities = st.selectbox("Extra-curricular Activities", options=["yes", "no"])
    
with col2:
    internet = st.selectbox("Internet Access at Home", options=["yes", "no"])
    famrel = st.slider("Quality of Family Relationships", min_value=1, max_value=5, value=4)
    goout = st.slider("Going Out with Friends", min_value=1, max_value=5, value=3)
    health = st.slider("Current Health Status", min_value=1, max_value=5, value=4)
        
# Hardcode the less interesting demographic parameters behind the scenes
school = 'GP'
sex = 'F'
age = 16
address = 'U'
famsize = 'GT3'
Pstatus = 'T'
Medu = 2
Fedu = 2
Mjob = 'other'
Fjob = 'other'
reason = 'course'
guardian = 'mother'
traveltime = 1
schoolsup = 'no'
famsup = 'yes'
paid = 'no'
nursery = 'yes'
higher = 'yes'
romantic = 'no'
freetime = 3
Dalc = 1
Walc = 1

# Create input dataframe matching the original training data format
input_data = pd.DataFrame([{
        'school': school, 'sex': sex, 'age': age, 'address': address,
        'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
        'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
        'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
        'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid, 'activities': activities,
        'nursery': nursery, 'higher': higher, 'internet': internet, 'romantic': romantic,
        'famrel': famrel, 'freetime': freetime, 'goout': goout, 'Dalc': Dalc,
        'Walc': Walc, 'health': health, 'absences': absences
    }])
    
try:
    # Preprocess
    processed_data = preprocessor.transform(input_data)
    # Predict
    prediction = model.predict(processed_data)[0]
    
    st.success(f"### Predicted Final Score (G3): {prediction:.2f} / 20.00")
    
    # Display some context
    st.info(f"Model used for prediction: {model_name}")
    
except Exception as e:
    st.error(f"Prediction error: {e}")
