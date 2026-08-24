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
    model_filename = selected_model_option.replace(" ", "_") + ".pkl"
    model = joblib.load(os.path.join(model_dir, model_filename))
    return preprocessor, model, selected_model_option

st.sidebar.markdown("### ⚙️ Engine Settings")
selected_model_option = st.sidebar.selectbox(
    "Choose ML Algorithm",
    ["Random Forest", "Linear Regression", "Decision Tree"]
)

try:
    preprocessor, model, model_name = load_assets(selected_model_option)
    st.sidebar.success(f"Active Model: {model_name}")
except Exception as e:
    st.error(f"Error loading models. Have you trained them yet? \n({e})")
    st.stop()

# --- PROFILE MANAGEMENT ---
if 'profiles' not in st.session_state:
    st.session_state.profiles = {
        "The Average Student": {"study": 2, "abs": 4, "fail": 0, "higher": "yes", "walc": 2, "dalc": 1, "freetime": 3, "goout": 3, "activities": "no", "internet": "yes", "famrel": 4, "health": 4},
        "The High Achiever": {"study": 4, "abs": 0, "fail": 0, "higher": "yes", "walc": 1, "dalc": 1, "freetime": 2, "goout": 2, "activities": "yes", "internet": "yes", "famrel": 5, "health": 5},
        "The Struggling Student": {"study": 1, "abs": 45, "fail": 3, "higher": "no", "walc": 5, "dalc": 3, "freetime": 5, "goout": 5, "activities": "no", "internet": "no", "famrel": 2, "health": 2},
        "The Social Butterfly": {"study": 1, "abs": 12, "fail": 1, "higher": "yes", "walc": 4, "dalc": 2, "freetime": 5, "goout": 5, "activities": "yes", "internet": "yes", "famrel": 5, "health": 4},
        "The Homebody": {"study": 3, "abs": 2, "fail": 0, "higher": "yes", "walc": 1, "dalc": 1, "freetime": 4, "goout": 1, "activities": "no", "internet": "yes", "famrel": 4, "health": 3},
        "The Rebel": {"study": 1, "abs": 65, "fail": 4, "higher": "no", "walc": 5, "dalc": 5, "freetime": 5, "goout": 5, "activities": "no", "internet": "no", "famrel": 1, "health": 2}
    }

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Student Profiles")
st.sidebar.markdown("Instantly change hidden background features to see how the model reacts to different extremes.")
profile_name = st.sidebar.selectbox("Load a preset background", list(st.session_state.profiles.keys()))

p = st.session_state.profiles[profile_name]

st.subheader("Student Lifestyle Tweaks")
    
col1, col2 = st.columns(2)

with col1:
    studytime = st.selectbox("Study Time", options=[1, 2, 3, 4], format_func=lambda x: ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"][x-1], index=p['study']-1)
    absences = st.slider("Number of Absences", min_value=0, max_value=93, value=p['abs'])
    failures = st.number_input("Past Class Failures", min_value=0, max_value=4, value=p['fail'])
    activities = st.selectbox("Extra-curricular Activities", options=["yes", "no"], index=0 if p['activities']=="yes" else 1)
    
with col2:
    internet = st.selectbox("Internet Access at Home", options=["yes", "no"], index=0 if p['internet']=="yes" else 1)
    famrel = st.slider("Quality of Family Relationships", min_value=1, max_value=5, value=p['famrel'])
    goout = st.slider("Going Out with Friends", min_value=1, max_value=5, value=p['goout'])
    health = st.slider("Current Health Status", min_value=1, max_value=5, value=p['health'])

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
romantic = 'no'

# Variables from profile (hidden from main UI but can be saved in custom profile)
higher = p['higher']
freetime = p['freetime']
Dalc = p['dalc']
Walc = p['walc']

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
    # Preprocess using the saved dictionary
    input_num = input_data[preprocessor['numerical_cols']]
    input_cat = input_data[preprocessor['categorical_cols']]
    
    processed_num = preprocessor['scaler'].transform(input_num)
    processed_cat = preprocessor['encoder'].transform(input_cat)
    
    # Combine numerical and categorical features
    import numpy as np
    processed_data = np.hstack((processed_num, processed_cat))
    
    # Predict
    prediction = model.predict(processed_data)[0]
    
    st.success(f"### Predicted Final Score (G3): {prediction:.2f} / 20.00")
    st.info(f"Model used for prediction: {model_name}")
    
except Exception as e:
    st.error(f"Prediction error: {e}")

st.markdown("---")
with st.expander("➕ Create Custom Profile"):
    st.write("Tweak the sliders above, adjust the hidden parameters below, and save as a new profile!")
    new_name = st.text_input("Profile Name (e.g. 'My Custom Student')")
    
    colA, colB = st.columns(2)
    with colA:
        new_higher = st.selectbox("Wants Higher Education", ["yes", "no"], index=0 if higher=="yes" else 1)
        new_walc = st.slider("Weekend Alcohol Consumption", 1, 5, value=Walc)
    with colB:
        new_dalc = st.slider("Workday Alcohol Consumption", 1, 5, value=Dalc)
        new_freetime = st.slider("Amount of Free Time", 1, 5, value=freetime)
        
    if st.button("Save Profile"):
        if new_name:
            st.session_state.profiles[new_name] = {
                "study": studytime,
                "abs": absences,
                "fail": failures,
                "activities": activities,
                "internet": internet,
                "famrel": famrel,
                "goout": goout,
                "health": health,
                "higher": new_higher,
                "walc": new_walc,
                "dalc": new_dalc,
                "freetime": new_freetime
            }
            st.rerun()
