import streamlit as st
import joblib
import numpy as np

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #4C72B0;
        color: white;
        font-size: 18px;
        padding: 10px 40px;
        border-radius: 10px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #3a5a8f; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

#load model
model=joblib.load("model/student_model.pkl")

st.title("Student Pass Predictor")

#input
studytime=st.selectbox("Weekly study time(1-low, 10-high)",options=[1,2,3,4,5,6,7,8,9,10])
failures=st.selectbox("Nummber of Past Failures", options=[0, 1, 2, 3])

absences = st.slider("Number of Absences", min_value=0, max_value=93, value=5)

health = st.selectbox("Health Status (1=very bad, 5=very good)", options=[1, 2, 3, 4, 5])

famrel = st.selectbox("Family Relationship Quality (1=very bad, 5=excellent)", options=[1, 2, 3, 4, 5])

goout = st.selectbox("Goes Out With Friends (1=rarely, 5=very often)", options=[1, 2, 3, 4, 5])

#main predict

if st.button("predict"):
    data=np.array([[studytime,failures,absences,health,famrel,goout]])
    prediction=model.predict(data)[0]
    probability=model.predict_proba(data)[0]

    if prediction==1:
        st.success(f"Student will pass!:{round(probability[1]*100,1)}%")

    else:
        st.error(f"Student will Fail!:{round(probability[0]*100,1)}%")