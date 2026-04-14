import streamlit as st
import pickle
import pandas as pd

# ---------------- LOAD MODEL + ENCODERS ---------------- #
model = pickle.load(open("xgb_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

st.set_page_config(page_title="Accident Severity Predictor", layout="centered")

st.title("🚧 Road Accident Severity Predictor")
st.markdown("Predict accident severity based on road and traffic conditions")

# ---------------- FEATURE ORDER (CRITICAL FIX) ---------------- #
feature_order = [
    "Weather",
    "Vehicle_Type",
    "Collision_Type",
    "Time_of_Day",
    "Lighting",
    "Speed_Category",
    "Number_of_Lanes",
    "Traffic_Volume",
    "Road_Geometry",
    "Gradient"
]

# ---------------- INPUTS ---------------- #
inputs = {}

inputs["Collision_Type"] = st.selectbox("Collision Type", encoders['Collision_Type'].classes_)
inputs["Road_Geometry"] = st.selectbox("Road Geometry", encoders['Road_Geometry'].classes_)
inputs["Speed_Category"] = st.selectbox("Speed Category", encoders['Speed_Category'].classes_)
inputs["Number_of_Lanes"] = st.selectbox("Number of Lanes", encoders['Number_of_Lanes'].classes_)
inputs["Gradient"] = st.selectbox("Gradient", encoders['Gradient'].classes_)
inputs["Weather"] = st.selectbox("Weather", encoders['Weather'].classes_)
inputs["Vehicle_Type"] = st.selectbox("Vehicle Type", encoders['Vehicle_Type'].classes_)
inputs["Time_of_Day"] = st.selectbox("Time of Day", encoders['Time_of_Day'].classes_)
inputs["Traffic_Volume"] = st.selectbox("Traffic Volume", encoders['Traffic_Volume'].classes_)
inputs["Lighting"] = st.selectbox("Lighting", encoders['Lighting'].classes_)

# ---------------- PREDICTION ---------------- #
if st.button("Predict Severity"):

    # ✅ ENCODE WITH GUARANTEED SAFE MAPPING
    encoded = {
        col: encoders[col].transform([inputs[col]])[0]
        for col in inputs
    }

    # ✅ FORCE COLUMN ORDER (THIS FIXES YOUR ERROR)
    input_df = pd.DataFrame([[encoded[col] for col in feature_order]],
                            columns=feature_order)

    # prediction
    prediction = model.predict(input_df)[0]

    severity = encoders['Severity'].inverse_transform([prediction])[0]

    st.subheader(f"🚨 Predicted Severity: {severity}")

    # ---------------- RECOMMENDATIONS ---------------- #
    if severity == "Fatal":
        st.error("High Risk! Immediate intervention required.")
        st.write("- Improve road design")
        st.write("- Install speed control measures")
        st.write("- Increase enforcement")

    elif severity == "Grievous Injury":
        st.warning("Moderate-High Risk")
        st.write("- Add warning signs")
        st.write("- Improve lighting")
        st.write("- Traffic calming")

    elif severity == "Minor Injury":
        st.info("Moderate Risk")
        st.write("- Monitor traffic")
        st.write("- Improve road markings")

    else:
        st.success("Low Risk")
        st.write("- Maintain current conditions")
