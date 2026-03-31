# Road Accident Severity Prediction System

This project builds an end-to-end machine learning system to predict the severity of road accidents based on multiple real-world factors such as vehicle type, collision type, weather conditions, and road characteristics.

---

##  Problem Statement
Road accidents result in varying levels of severity ranging from property damage to fatal injuries. Understanding the factors influencing severity helps authorities improve road safety and decision-making.

---

##  Dataset
- Synthetic dataset generated using realistic traffic distributions
- 50,000+ records
- Features include:
  - Weather conditions
  - Vehicle type
  - Collision type
  - Road type
  - Time of day
  - Lighting condition
  - Speed category
  - Traffic control status

---

##  Approach

### Data Processing
- Generated realistic accident data using probabilistic distributions  
- Cleaned and encoded categorical features  
- Performed Exploratory Data Analysis (EDA)  

### Modeling
- Applied multiple classification models:
  - Logistic Regression
  - Random Forest
  - XGBoost  
- Best performance achieved using **XGBoost (~77% accuracy)**  

### Model Explainability
- Used **SHAP (SHapley Additive Explanations)**  
- Identified impact of features like speed, collision type, and lighting on severity  

---

##  Deployment
- Built an interactive web application using **Streamlit**
- Users can input accident conditions and get real-time severity prediction

Live App:  
https://accident-severity-app-major-project.streamlit.app/

---

##  Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- SHAP  
- Streamlit  

---

##  Key Insights
- High speed and head-on collisions significantly increase severity  
- Poor lighting and night conditions increase fatal risk  
- Heavy vehicles contribute to higher severity outcomes  

---

##  Run Locally


pip install -r requirements.txt
streamlit run app.py
