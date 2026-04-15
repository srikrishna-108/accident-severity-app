import streamlit as st
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Accident Severity AI", layout="wide")

# ---------------- HEADER ---------------- #
st.markdown("<h1 style='text-align:center;'>🚧 Accident Severity AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Smart Road Risk Analysis & Decision System</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    return xgb.Booster(model_file="xgb_model.json")

model = load_model()

# ---------------- ENCODING ---------------- #
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

# ✅ EXACT TRAINING FEATURE ORDER
feature_names = [
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

# ---------------- SIDEBAR INPUT ---------------- #
st.sidebar.header("📋 Input Road Conditions")

collision = st.sidebar.selectbox("Collision Type", collision_map.keys())
speed = st.sidebar.selectbox("Speed Category", speed_map.keys())
geometry = st.sidebar.selectbox("Road Geometry", geometry_map.keys())
lanes = st.sidebar.selectbox("Number of Lanes", lanes_map.keys())
gradient = st.sidebar.selectbox("Gradient", gradient_map.keys())

weather = st.sidebar.selectbox("Weather", weather_map.keys())
vehicle = st.sidebar.selectbox("Vehicle Type", vehicle_map.keys())
time = st.sidebar.selectbox("Time of Day", time_map.keys())
traffic = st.sidebar.selectbox("Traffic Volume", traffic_map.keys())
lighting = st.sidebar.selectbox("Lighting Condition", lighting_map.keys())

# ---------------- MAIN ---------------- #
if st.sidebar.button("🚀 Predict Severity"):

    # ✅ CORRECT ORDER (MATCHES TRAINING)
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

    # ---------------- RISK SCORE ---------------- #
    risk_score = sum([
        3 if speed == "High" else 0,
        3 if collision == "Head-on" else 0,
        2 if weather != "Clear" else 0,
        2 if lighting == "Poor" else 0,
        2 if traffic == "High" else 0,
        2 if gradient == "Steep" else 0,
        2 if geometry == "Curve" else 0,
        1 if vehicle == "Heavy" else 0
    ])

    col1, col2 = st.columns(2)

    # ---------------- RESULT ---------------- #
    with col1:
        st.subheader("🚨 Prediction Result")

        if result == "Fatal":
            st.error(f"🚨 {result}")
        elif result == "Grievous Injury":
            st.warning(f"⚠️ {result}")
        elif result == "Minor Injury":
            st.info(f"ℹ️ {result}")
        else:
            st.success(f"✅ {result}")

        st.markdown("### 📊 Risk Score")
        st.progress(min(risk_score / 15, 1.0))
        st.write(f"{risk_score} / 15")

    # ---------------- FEATURE IMPORTANCE ---------------- #
    with col2:
        st.subheader("📈 Feature Importance")

        importance = model.get_score(importance_type='weight')
        values = [importance.get(f"f{i}", 0) for i in range(len(feature_names))]

        fig, ax = plt.subplots()
        ax.barh(feature_names, values)
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    # ---------------- EXPLANATION ---------------- #
    st.markdown("---")
    st.subheader("🧠 Model Insight")

    top_idx = np.argmax(values)
    st.write(f"Most influential factor: **{feature_names[top_idx]}**")

    # ---------------- RECOMMENDATIONS ---------------- #
    st.markdown("---")
    st.subheader("🚧 Safety Recommendations")

    if result == "Fatal":
        st.write("- Immediate road redesign")
        st.write("- Install crash barriers")
        st.write("- Strict speed enforcement")

    elif result == "Grievous Injury":
        st.write("- Improve signage")
        st.write("- Reduce speed limits")
        st.write("- Increase monitoring")

    elif result == "Minor Injury":
        st.write("- Routine inspection")
        st.write("- Minor safety improvements")

    else:
        st.write("- Maintain current safe conditions")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown("<p style='text-align:center;'>🚀 Built for Smart Road Safety Analysis</p>", unsafe_allow_html=True)
