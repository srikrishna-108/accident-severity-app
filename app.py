import streamlit as st
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Accident Severity AI", layout="wide")

st.title("🚧 Accident Severity Predictor")
st.markdown("Smart Road Risk & Severity Analysis")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return xgb.Booster(model_file="xgb_model.json")

model = load_model()

# ---------------- MAPS ---------------- #
weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2}
vehicle_map = {"Two-wheeler": 0, "Car": 1, "Truck": 2, "Bus": 3}
collision_map = {"Rear-end": 0, "Side": 1, "Head-on": 2, "Pedestrian": 3}
time_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
lighting_map = {"Good": 0, "Poor": 1}
speed_map = {"Low": 0, "Medium": 1, "High": 2}
lanes_map = {"2": 0, "4": 1, "6+": 2}
traffic_map = {"Low": 0, "Medium": 1, "High": 2}
geometry_map = {"Straight": 0, "Moderate Curve": 1, "Sharp Curve": 2}
gradient_map = {"Flat": 0, "Moderate": 1, "Steep": 2}

feature_names = [
    "Weather","Vehicle_Type","Collision_Type","Time_of_Day",
    "Lighting","Speed_Category","Number_of_Lanes",
    "Traffic_Volume","Road_Geometry","Gradient"
]

# ---------------- INPUT ---------------- #
st.sidebar.header("📋 Inputs")

weather = st.sidebar.selectbox("Weather", weather_map.keys())
vehicle = st.sidebar.selectbox("Vehicle", vehicle_map.keys())
collision = st.sidebar.selectbox("Collision", collision_map.keys())
time = st.sidebar.selectbox("Time", time_map.keys())
lighting = st.sidebar.selectbox("Lighting", lighting_map.keys())
speed = st.sidebar.selectbox("Speed", speed_map.keys())
lanes = st.sidebar.selectbox("Lanes", lanes_map.keys())
traffic = st.sidebar.selectbox("Traffic", traffic_map.keys())
geometry = st.sidebar.selectbox("Geometry", geometry_map.keys())
gradient = st.sidebar.selectbox("Gradient", gradient_map.keys())

# ---------------- PREDICT ---------------- #
if st.sidebar.button("🚀 Predict"):

    input_data = np.array([[
        weather_map[weather],
        vehicle_map[vehicle],
        collision_map[collision],
        time_map[time],
        lighting_map[lighting],
        speed_map[speed],
        lanes_map[lanes],
        traffic_map[traffic],
        geometry_map[geometry],
        gradient_map[gradient]
    ]])

    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
    probs = model.predict(dmatrix)
    pred = int(np.argmax(probs))

    severity_map = {
        0: "Property Damage",
        1: "Minor Injury",
        2: "Grievous Injury",
        3: "Fatal"
    }

    result = severity_map[pred]

    # ---------------- RISK SCORE (FIXED) ---------------- #
    risk_score = 0

    if speed == "High": risk_score += 3
    if collision in ["Head-on", "Pedestrian"]: risk_score += 3
    if weather != "Clear": risk_score += 2
    if lighting == "Poor": risk_score += 2
    if traffic == "High": risk_score += 2
    if geometry == "Sharp Curve": risk_score += 2
    if gradient == "Steep": risk_score += 2
    if vehicle in ["Truck", "Bus"]: risk_score += 1

    # ---------------- OUTPUT ---------------- #
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚨 Prediction")

        if result == "Fatal":
            st.error(result)
        elif result == "Grievous Injury":
            st.warning(result)
        elif result == "Minor Injury":
            st.info(result)
        else:
            st.success(result)

        st.subheader("📊 Risk Score")
        st.progress(min(risk_score/15,1.0))
        st.write(f"{risk_score}/15")

    # ---------------- GRAPH (FIXED PROPERLY) ---------------- #
    with col2:
        st.subheader("📈 Input Impact (Proxy)")

        values = input_data.flatten()

        fig, ax = plt.subplots()
        ax.barh(feature_names, values)
        ax.set_xlabel("Encoded Value (Relative Impact Proxy)")
        st.pyplot(fig)

    # ---------------- EXPLANATION (INPUT BASED) ---------------- #
    st.markdown("---")
    st.subheader("🧠 Why this prediction?")

    reasons = []

    if speed == "High": reasons.append("High speed increases severity")
    if collision in ["Head-on", "Pedestrian"]: reasons.append("Dangerous collision type")
    if weather != "Clear": reasons.append("Adverse weather conditions")
    if lighting == "Poor": reasons.append("Low visibility")
    if geometry == "Sharp Curve": reasons.append("Sharp curve risk")
    if gradient == "Steep": reasons.append("Steep gradient impact")

    for r in reasons:
        st.write(f"- {r}")

    # ---------------- SMART RECOMMENDATIONS ---------------- #
    st.markdown("---")
    st.subheader("🚧 Recommendations")

    if speed == "High":
        st.write("- Enforce speed limits")

    if lighting == "Poor":
        st.write("- Improve road lighting")

    if geometry == "Sharp Curve":
        st.write("- Install warning signs & barriers")

    if weather != "Clear":
        st.write("- Add skid-resistant surface")

    if collision in ["Head-on", "Pedestrian"]:
        st.write("- Introduce divider / pedestrian crossings")

    if traffic == "High":
        st.write("- Optimize traffic control systems")
