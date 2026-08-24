import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Set page config
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 Student Performance Predictor")
st.markdown("This application predicts a student's final score (G3) based on their demographic, social, and academic background.")

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
        "The Average Student": {
            "studytime": 2, "absences": 4, "failures": 0, "higher": "yes", "Walc": 2, "Dalc": 1, 
            "freetime": 3, "goout": 3, "activities": "no", "internet": "yes", "famrel": 4, "health": 4,
            "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
            "traveltime": 1, "schoolsup": "no", "famsup": "yes", "paid": "no", "nursery": "yes", "romantic": "no"
        },
        "The High Achiever": {
            "studytime": 4, "absences": 0, "failures": 0, "higher": "yes", "Walc": 1, "Dalc": 1, 
            "freetime": 2, "goout": 2, "activities": "yes", "internet": "yes", "famrel": 5, "health": 5,
            "school": "GP", "sex": "F", "age": 15, "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 4, "Fedu": 4, "Mjob": "teacher", "Fjob": "teacher", "reason": "reputation", "guardian": "mother",
            "traveltime": 1, "schoolsup": "yes", "famsup": "yes", "paid": "yes", "nursery": "yes", "romantic": "no"
        },
        "The Struggling Student": {
            "studytime": 1, "absences": 45, "failures": 3, "higher": "no", "Walc": 5, "Dalc": 3, 
            "freetime": 5, "goout": 5, "activities": "no", "internet": "no", "famrel": 2, "health": 2,
            "school": "MS", "sex": "M", "age": 19, "address": "R", "famsize": "LE3", "Pstatus": "A",
            "Medu": 1, "Fedu": 1, "Mjob": "at_home", "Fjob": "at_home", "reason": "course", "guardian": "other",
            "traveltime": 4, "schoolsup": "no", "famsup": "no", "paid": "no", "nursery": "no", "romantic": "yes"
        },
        "The Social Butterfly": {
            "studytime": 1, "absences": 12, "failures": 1, "higher": "yes", "Walc": 4, "Dalc": 2, 
            "freetime": 5, "goout": 5, "activities": "yes", "internet": "yes", "famrel": 5, "health": 4,
            "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 3, "Fedu": 3, "Mjob": "services", "Fjob": "services", "reason": "home", "guardian": "mother",
            "traveltime": 1, "schoolsup": "no", "famsup": "yes", "paid": "no", "nursery": "yes", "romantic": "yes"
        },
        "The Homebody": {
            "studytime": 3, "absences": 2, "failures": 0, "higher": "yes", "Walc": 1, "Dalc": 1, 
            "freetime": 4, "goout": 1, "activities": "no", "internet": "yes", "famrel": 4, "health": 3,
            "school": "GP", "sex": "M", "age": 15, "address": "U", "famsize": "GT3", "Pstatus": "T",
            "Medu": 2, "Fedu": 2, "Mjob": "other", "Fjob": "other", "reason": "course", "guardian": "mother",
            "traveltime": 1, "schoolsup": "no", "famsup": "yes", "paid": "yes", "nursery": "yes", "romantic": "no"
        },
        "The Rebel": {
            "studytime": 1, "absences": 65, "failures": 4, "higher": "no", "Walc": 5, "Dalc": 5, 
            "freetime": 5, "goout": 5, "activities": "no", "internet": "no", "famrel": 1, "health": 2,
            "school": "MS", "sex": "M", "age": 18, "address": "U", "famsize": "GT3", "Pstatus": "A",
            "Medu": 1, "Fedu": 1, "Mjob": "services", "Fjob": "other", "reason": "other", "guardian": "other",
            "traveltime": 2, "schoolsup": "no", "famsup": "no", "paid": "no", "nursery": "no", "romantic": "yes"
        }
    }

st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Student Profiles")
profile_name = st.sidebar.selectbox("Load a preset background", list(st.session_state.profiles.keys()))

st.sidebar.markdown("---")
demo_mode = st.sidebar.checkbox("🌟 Presentation Demo Mode (Stretch Scores)")
if demo_mode:
    st.sidebar.info("Demo Mode scales predictions from 0 to 20 for dramatic effect during presentations.")

p = st.session_state.profiles[profile_name]

# Placeholder for the score banner at the top
score_placeholder = st.empty()
info_placeholder = st.empty()
st.markdown("---")

# The inputs are now housed in this expander at the bottom
with st.expander("➕ Tweak All Student Parameters (Live Update)", expanded=False):
    st.markdown("#### Core Lifestyle")
    col1, col2 = st.columns(2)
    with col1:
        studytime = st.selectbox("Study Time", options=[1, 2, 3, 4], format_func=lambda x: ["<2 hrs", "2-5 hrs", "5-10 hrs", ">10 hrs"][x-1], index=p['studytime']-1)
        absences = st.slider("Number of Absences", min_value=0, max_value=93, value=p['absences'])
        failures = st.number_input("Past Class Failures", min_value=0, max_value=4, value=p['failures'])
        activities = st.selectbox("Extra-curricular Activities", options=["yes", "no"], index=0 if p['activities']=="yes" else 1)
        
    with col2:
        internet = st.selectbox("Internet Access at Home", options=["yes", "no"], index=0 if p['internet']=="yes" else 1)
        famrel = st.slider("Quality of Family Relationships", min_value=1, max_value=5, value=p['famrel'])
        goout = st.slider("Going Out with Friends", min_value=1, max_value=5, value=p['goout'])
        health = st.slider("Current Health Status", min_value=1, max_value=5, value=p['health'])
    
    st.markdown("#### Demographics & Schooling")
    colA, colB, colC = st.columns(3)
    with colA:
        new_school = st.selectbox("School", ["GP", "MS"], index=["GP", "MS"].index(p['school']))
        new_sex = st.selectbox("Sex", ["F", "M"], index=["F", "M"].index(p['sex']))
    with colB:
        new_age = st.slider("Age", 15, 22, value=p['age'])
        new_address = st.selectbox("Address Type", ["U", "R"], index=["U", "R"].index(p['address']))
    with colC:
        new_reason = st.selectbox("Reason for choosing school", ["home", "reputation", "course", "other"], index=["home", "reputation", "course", "other"].index(p['reason']))
        new_traveltime = st.slider("Travel Time to School", 1, 4, value=p['traveltime'])

    st.markdown("#### Family Background")
    colD, colE, colF = st.columns(3)
    with colD:
        new_famsize = st.selectbox("Family Size", ["LE3", "GT3"], index=["LE3", "GT3"].index(p['famsize']))
        new_Pstatus = st.selectbox("Parents Cohabitation Status", ["T", "A"], index=["T", "A"].index(p['Pstatus']))
        new_guardian = st.selectbox("Guardian", ["mother", "father", "other"], index=["mother", "father", "other"].index(p['guardian']))
    with colE:
        new_Medu = st.slider("Mother's Education (0-4)", 0, 4, value=p['Medu'])
        new_Fedu = st.slider("Father's Education (0-4)", 0, 4, value=p['Fedu'])
    with colF:
        new_Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(p['Mjob']))
        new_Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(p['Fjob']))

    st.markdown("#### Support & Extra Lifestyle")
    colG, colH, colI = st.columns(3)
    with colG:
        new_schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"], index=["yes", "no"].index(p['schoolsup']))
        new_famsup = st.selectbox("Family Educational Support", ["yes", "no"], index=["yes", "no"].index(p['famsup']))
        new_paid = st.selectbox("Extra Paid Classes", ["yes", "no"], index=["yes", "no"].index(p['paid']))
    with colH:
        new_nursery = st.selectbox("Attended Nursery School", ["yes", "no"], index=["yes", "no"].index(p['nursery']))
        new_higher = st.selectbox("Wants Higher Education", ["yes", "no"], index=["yes", "no"].index(p['higher']))
        new_romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"], index=["yes", "no"].index(p['romantic']))
    with colI:
        new_Walc = st.slider("Weekend Alcohol Consumption", 1, 5, value=p['Walc'])
        new_Dalc = st.slider("Workday Alcohol Consumption", 1, 5, value=p['Dalc'])
        new_freetime = st.slider("Amount of Free Time", 1, 5, value=p['freetime'])
        
    st.markdown("---")
    new_name = st.text_input("Save as New Profile (Name)")
    if st.button("Save Custom Profile"):
        if new_name:
            st.session_state.profiles[new_name] = {
                "studytime": studytime, "absences": absences, "failures": failures, "activities": activities,
                "internet": internet, "famrel": famrel, "goout": goout, "health": health, "school": new_school,
                "sex": new_sex, "age": new_age, "address": new_address, "famsize": new_famsize, "Pstatus": new_Pstatus,
                "reason": new_reason, "traveltime": new_traveltime, "guardian": new_guardian, "Medu": new_Medu,
                "Fedu": new_Fedu, "Mjob": new_Mjob, "Fjob": new_Fjob, "schoolsup": new_schoolsup, "famsup": new_famsup,
                "paid": new_paid, "nursery": new_nursery, "higher": new_higher, "romantic": new_romantic,
                "Walc": new_Walc, "Dalc": new_Dalc, "freetime": new_freetime
            }
            st.rerun()

# --- PREDICTION LOGIC ---
input_data = pd.DataFrame([{
        'school': new_school, 'sex': new_sex, 'age': new_age, 'address': new_address,
        'famsize': new_famsize, 'Pstatus': new_Pstatus, 'Medu': new_Medu, 'Fedu': new_Fedu,
        'Mjob': new_Mjob, 'Fjob': new_Fjob, 'reason': new_reason, 'guardian': new_guardian,
        'traveltime': new_traveltime, 'studytime': studytime, 'failures': failures,
        'schoolsup': new_schoolsup, 'famsup': new_famsup, 'paid': new_paid, 'activities': activities,
        'nursery': new_nursery, 'higher': new_higher, 'internet': internet, 'romantic': new_romantic,
        'famrel': famrel, 'freetime': new_freetime, 'goout': goout, 'Dalc': new_Dalc,
        'Walc': new_Walc, 'health': health, 'absences': absences
    }])
    
try:
    input_num = input_data[preprocessor['numerical_cols']]
    input_cat = input_data[preprocessor['categorical_cols']]
    
    processed_num = preprocessor['scaler'].transform(input_num)
    processed_cat = preprocessor['encoder'].transform(input_cat)
    
    processed_data = np.hstack((processed_num, processed_cat))
    prediction = model.predict(processed_data)[0]
    
    if demo_mode:
        # Stretch prediction from historical min/max (0.7 to 16.2) out to 0 to 20 range
        min_pred = 0.71
        max_pred = 16.20
        # Normalize and scale
        stretched = ((prediction - min_pred) / (max_pred - min_pred)) * 20.0
        # Cap between 0 and 20
        prediction = max(0.0, min(20.0, stretched))
    
    # Update the UI at the top
    score_placeholder.success(f"### Predicted Final Score (G3): {prediction:.2f} / 20.00")
    info_placeholder.info(f"Model used for prediction: {model_name}")
    
except Exception as e:
    score_placeholder.error(f"Prediction error: {e}")
