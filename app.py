import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Accident Severity Predictor", layout="centered")

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align: center;'>🚧 Accident Severity Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered road safety analysis tool</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return pickle.load(open("xgb_model.pkl", "rb"))

model = load_model()

# ---------------- MANUAL ENCODING ---------------- #
collision_map = {"Head-on": 0, "Rear-end": 1, "Side": 2}
speed_map = {"Low": 0, "Medium": 1, "High": 2}
geometry_map = {"Straight": 0, "Curve": 1}
lanes_map = {"2": 0, "4": 1, "6": 2}
gradient_map = {"Flat": 0, "Moderate": 1, "Steep": 2}
weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2}
vehicle_map = {"Light": 0, "Heavy": 1}
time_map = {"Day": 0, "Night": 1}
traffic_map = {"Low": 0, "Medium": 1, "High": 2}
lighting_map = {"Good": 0, "Poor": 1}

# ---------------- INPUT UI ---------------- #
st.subheader("📋 Input Road Conditions")

col1, col2 = st.columns(2)

with col1:
    collision = st.selectbox("Collision Type", collision_map.keys())
    speed = st.selectbox("Speed Category", speed_map.keys())
    geometry = st.selectbox("Road Geometry", geometry_map.keys())
    lanes = st.selectbox("Number of Lanes", lanes_map.keys())
    gradient = st.selectbox("Gradient", gradient_map.keys())

with col2:
    weather = st.selectbox("Weather", weather_map.keys())
    vehicle = st.selectbox("Vehicle Type", vehicle_map.keys())
    time = st.selectbox("Time of Day", time_map.keys())
    traffic = st.selectbox("Traffic Volume", traffic_map.keys())
    lighting = st.selectbox("Lighting Condition", lighting_map.keys())

st.markdown("---")

# ---------------- RISK SCORE FUNCTION ---------------- #
def calculate_risk():
    score = 0
    
    if speed == "High": score += 3
    if collision == "Head-on": score += 3
    if weather != "Clear": score += 2
    if lighting == "Poor": score += 2
    if traffic == "High": score += 2
    if gradient == "Steep": score += 2
    if geometry == "Curve": score += 2
    if vehicle == "Heavy": score += 1
    
    return score

# ---------------- PREDICTION ---------------- #
if st.button("🚀 Predict Severity"):

    input_data = np.array([[
        collision_map[collision],
        speed_map[speed],
        geometry_map[geometry],
        lanes_map[lanes],
        gradient_map[gradient],
        weather_map[weather],
        vehicle_map[vehicle],
        time_map[time],
        traffic_map[traffic],
        lighting_map[lighting]
    ]])

    prediction = model.predict(input_data)[0]

    severity_map = {
        0: "Property Damage",
        1: "Minor Injury",
        2: "Grievous Injury",
        3: "Fatal"
    }

    result = severity_map[prediction]

    # ---------------- RISK SCORE ---------------- #
    risk_score = calculate_risk()

    st.subheader("📊 Results")

    # Risk Meter
    st.progress(min(risk_score / 15, 1.0))
    st.write(f"**Risk Score:** {risk_score}/15")

    # ---------------- SEVERITY OUTPUT ---------------- #
    if result == "Fatal":
        st.error(f"🚨 Severity: {result}")
        st.markdown("### 🚧 Recommended Actions")
        st.markdown("""
        - Immediate road redesign  
        - Speed enforcement (cameras, penalties)  
        - Install crash barriers  
        - Improve lighting & signage  
        """)

    elif result == "Grievous Injury":
        st.warning(f"⚠️ Severity: {result}")
        st.markdown("### 🚧 Recommended Actions")
        st.markdown("""
        - Add warning signs  
        - Improve road markings  
        - Monitor traffic flow  
        - Reduce speed limits  
        """)

    elif result == "Minor Injury":
        st.info(f"ℹ️ Severity: {result}")
        st.markdown("### 🚧 Recommended Actions")
        st.markdown("""
        - Routine inspection  
        - Minor geometric improvements  
        - Awareness measures  
        """)

    else:
        st.success(f"✅ Severity: {result}")
        st.markdown("### 🚧 Recommended Actions")
        st.markdown("""
        - Maintain road conditions  
        - Regular monitoring  
        """)

    # ---------------- INSIGHT BOX ---------------- #
    st.markdown("---")
    st.subheader("🧠 Insight")

    if risk_score >= 10:
        st.write("High-risk road segment detected. Immediate intervention required.")
    elif risk_score >= 6:
        st.write("Moderate risk. Preventive measures recommended.")
    else:
        st.write("Low-risk conditions. Maintain standards.")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown("<p style='text-align: center;'>Built for Road Safety Analysis 🚀</p>", unsafe_allow_html=True)
