import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("xgb_model.pkl", "rb"))
le_dict = pickle.load(open("encoders.pkl", "rb"))

st.title("Accident Severity Predictor")

weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy"])
vehicle = st.selectbox("Vehicle", ["Two-wheeler", "Car", "Bus", "Truck"])
collision = st.selectbox("Collision", ["Rear-end", "Head-on", "Side", "Pedestrian"])
road = st.selectbox("Road", ["NH", "Urban"])
junction = st.selectbox("Junction", ["Signalized", "Uncontrolled", "T-intersection"])
time = st.selectbox("Time", ["Morning", "Afternoon", "Evening", "Night"])
lighting = st.selectbox("Lighting", ["Good", "Poor"])
traffic = st.selectbox("Traffic", ["Working", "Not Working"])
speed = st.selectbox("Speed", ["Low", "Medium", "High"])
heavy = st.selectbox("Heavy Vehicle", [0, 1])

if st.button("Predict"):

    data = pd.DataFrame([{
        'Weather': weather,
        'Vehicle_Type': vehicle,
        'Collision_Type': collision,
        'Road_Type': road,
        'Junction_Type': junction,
        'Time_of_Day': time,
        'Lighting_Condition': lighting,
        'Traffic_Control': traffic,
        'Speed_Category': speed,
        'Heavy_Vehicle': heavy
    }])

    for col in data.columns:
        if col in le_dict:
            data[col] = le_dict[col].transform(data[col])

    pred = model.predict(data)
    result = le_dict['Severity'].inverse_transform(pred)

    st.success(f"Predicted Severity: {result[0]}")