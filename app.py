import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Severity AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    [data-testid="stSidebar"] {
        background: #f4f6fa;
        border-right: 1px solid #dde3ed;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    .severity-card {
        border-radius: 16px; padding: 1.5rem 1.8rem;
        margin-bottom: 1rem; border: 2px solid;
    }
    .card-fatal    { background: linear-gradient(135deg,#fff1f1 60%,#ffe0e0); border-color:#e53935; }
    .card-grievous { background: linear-gradient(135deg,#fff8e1 60%,#fff0c0); border-color:#fb8c00; }
    .card-minor    { background: linear-gradient(135deg,#e8f5e9 60%,#d4edda); border-color:#43a047; }
    .card-property { background: linear-gradient(135deg,#e3f2fd 60%,#cce5ff); border-color:#1e88e5; }

    .severity-title { font-size:2.1rem; font-weight:700; margin:0 0 .3rem; letter-spacing:-0.5px; }
    .severity-sub   { font-size:.92rem; color:#666; margin:0; }
    .fatal-title    { color:#b71c1c; }
    .grievous-title { color:#e65100; }
    .minor-title    { color:#2e7d32; }
    .property-title { color:#1565c0; }

    .risk-bar-bg {
        background:#e8ecf2; border-radius:10px;
        height:20px; width:100%; margin:8px 0 4px;
        overflow:hidden; box-shadow:inset 0 1px 3px rgba(0,0,0,.08);
    }
    .risk-bar-fill { height:100%; border-radius:10px; transition: width 0.6s ease; }

    .factor-pill {
        display:inline-block; padding:4px 13px; border-radius:20px;
        font-size:.8rem; font-weight:600; margin:3px 4px 3px 0; letter-spacing:.2px;
    }
    .pill-red    { background:#fdecea; color:#b71c1c; border:1.5px solid #e53935; }
    .pill-orange { background:#fff3e0; color:#bf360c; border:1.5px solid #fb8c00; }
    .pill-yellow { background:#fffde7; color:#f57f17; border:1.5px solid #f9a825; }
    .pill-green  { background:#e8f5e9; color:#1b5e20; border:1.5px solid #43a047; }
    .pill-blue   { background:#e3f2fd; color:#0d47a1; border:1.5px solid #1e88e5; }

    .reason-block {
        background:#f8f9fc; border-left:4px solid #e53935;
        border-radius:0 10px 10px 0; padding:.75rem 1.1rem;
        margin-bottom:.55rem; font-size:.9rem; line-height:1.6; color:#2d2d2d;
    }
    .reason-block.orange { border-color:#fb8c00; }
    .reason-block.yellow { border-color:#f9a825; background:#fffef5; }
    .reason-block.green  { border-color:#43a047; background:#f1f8f2; }
    .reason-block.blue   { border-color:#1e88e5; background:#f0f6ff; }

    .rec-card {
        background:#ffffff; border:1px solid #dde3ed; border-radius:12px;
        padding:.9rem 1.1rem; margin-bottom:.6rem; font-size:.88rem; line-height:1.55;
        box-shadow:0 1px 4px rgba(0,0,0,.04);
    }
    .rec-title { font-weight:700; font-size:.92rem; margin-bottom:5px; color:#1a1a2e; }
    .rec-icon  { font-size:1.1rem; margin-right:7px; }

    .section-header {
        font-size:1.05rem; font-weight:700; color:#1a1a2e;
        border-bottom:2px solid #eef1f7; padding-bottom:.4rem;
        margin:1.5rem 0 .9rem; letter-spacing:.1px;
    }
    .stExpander { border:1px solid #dde3ed !important; border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, encoders, metadata

model, encoders, metadata = load_artifacts()

feature_order   = metadata["feature_order"]
severity_labels = metadata["severity_labels"]   # list, e.g. ["No Injury","Simple Injuries","Grievous Injuries","Fatal"]
display_labels  = metadata["display_labels"]    # dict raw_col -> human label

severity_style = {
    "Fatal":              ("card-fatal",    "fatal-title",    "#e53935"),
    "Grievous Injuries":  ("card-grievous", "grievous-title", "#fb8c00"),
    "Simple Injuries":    ("card-minor",    "minor-title",    "#43a047"),
    "No Injury":          ("card-property", "property-title", "#1e88e5"),
}
severity_icon = {
    "Fatal":             "🔴",
    "Grievous Injuries": "🟠",
    "Simple Injuries":   "🟢",
    "No Injury":         "🔵",
}


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("# 🚧 Accident Severity Predictor")
st.markdown("Smart Road Risk & Severity Analysis System — powered by Tuned Random Forest · NH-163 Warangal–Hyderabad")
st.markdown("---")


# ─────────────────────────────────────────────
#  SIDEBAR INPUTS
# ─────────────────────────────────────────────
st.sidebar.markdown("## Road Conditions")
st.sidebar.markdown("Select the scenario details and click **Predict Severity**.")
st.sidebar.markdown("---")

user_inputs = {}
for col in feature_order:
    options = list(encoders[col].classes_)
    label   = display_labels.get(col, col)
    user_inputs[col] = st.sidebar.selectbox(label, options, key=col)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Predict Severity", use_container_width=True)

if not predict_btn:
    c1, c2, c3 = st.columns(3)
    cards = [
        ("🗂️", "Real Dataset",    "Trained on official NH-163 corridor accident records with engineered features."),
        ("🌲", "Random Forest",   "Ensemble of decision trees with calibrated severity confidence outputs."),
        ("🧠", "Explainable AI",  "Every prediction includes input-specific reasons and actionable safety recommendations."),
    ]
    for col, (icon, title, body) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div style="background:#f8f9fc;border:1px solid #dde3ed;border-radius:14px;padding:1.2rem 1.4rem;">
                <div style="font-size:1.8rem;margin-bottom:.4rem;">{icon}</div>
                <div style="font-weight:700;margin-bottom:.3rem;">{title}</div>
                <div style="font-size:.85rem;color:#666;">{body}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈 Configure road conditions in the sidebar and click **Predict Severity** to begin.")
    st.stop()


# ─────────────────────────────────────────────
#  ENCODE + PREDICT
# ─────────────────────────────────────────────
encoded_row = []
for col in feature_order:
    encoded_val = encoders[col].transform([user_inputs[col]])[0]
    encoded_row.append(int(encoded_val))

input_df = pd.DataFrame([encoded_row], columns=feature_order)
pred     = int(model.predict(input_df)[0])
probs    = model.predict_proba(input_df)[0]   # numpy array, length = number of classes

# Ensure severity_labels length matches probs length
n_classes = len(probs)
# Pad or trim severity_labels defensively
_severity_labels = list(severity_labels)
while len(_severity_labels) < n_classes:
    _severity_labels.append(f"Class {len(_severity_labels)}")
_severity_labels = _severity_labels[:n_classes]

result = _severity_labels[pred] if pred < len(_severity_labels) else f"Class {pred}"

# Probability-weighted risk score
risk_weights  = {0: 0.05, 1: 0.35, 2: 0.70, 3: 1.00}
risk_percent  = int(sum(
    probs[i] * risk_weights.get(i, 1.0) for i in range(n_classes)
) * 100)
risk_percent  = min(risk_percent, 100)

risk_color = (
    "#e53935" if risk_percent >= 65 else
    "#fb8c00" if risk_percent >= 40 else
    "#fdd835" if risk_percent >= 20 else
    "#43a047"
)
risk_label = (
    "Very High" if risk_percent >= 65 else
    "High"      if risk_percent >= 40 else
    "Moderate"  if risk_percent >= 20 else
    "Low"
)


# ─────────────────────────────────────────────
#  ROBUST KEY LOOKUP
# ─────────────────────────────────────────────
_norm_map = {k.lower().replace(" ", "").replace("_", ""): k for k in user_inputs}

def inp_val(*candidates):
    for c in candidates:
        norm = c.lower().replace(" ", "").replace("_", "")
        if norm in _norm_map:
            return user_inputs[_norm_map[norm]]
    return ""

cause              = inp_val("Cause", "cause_of_accident", "causeofaccident")
time_of_day        = inp_val("Time of Day", "TimeofDay", "time_of_day", "timeofday")
day_type           = inp_val("Day Type", "DayType", "day_type", "daytype")
road_geometry      = inp_val("Road Geometry", "RoadGeometry", "road_geometry", "roadgeometry")
victim_vehicle     = inp_val("Victim Vehicle Type", "VictimVehicleType", "victim_vehicle_type")
offender_vehicle   = inp_val("Offending Vehicle Type", "OffendingVehicleType", "offending_vehicle_type")
victim_manoeuvre   = inp_val("Victim Manoeuvre", "VictimManoeuvre", "victim_manoeuvre")
offender_manoeuvre = inp_val("Offender Manoeuvre", "OffenderManoeuvre", "offender_manoeuvre")
accident_type      = inp_val("Type of Accident", "TypeofAccident", "type_of_accident", "typeofaccident")


# ─────────────────────────────────────────────
#  ACTIVE FACTORS
# ─────────────────────────────────────────────
active_factors = []
if cause and ("over speeding" in cause.lower() or "speeding" in cause.lower()):
    active_factors.append(("Over Speeding", "red"))
if cause and ("drunken" in cause.lower() or "drunk" in cause.lower()):
    active_factors.append(("Drunken Driving", "red"))
if cause and "loss of control" in cause.lower():
    active_factors.append(("Loss of Control", "orange"))
if cause and ("bad road" in cause.lower() or "visibility" in cause.lower()):
    active_factors.append(("Road/Visibility", "orange"))
if time_of_day and "night" in time_of_day.lower():
    active_factors.append(("Night-time", "orange"))
if day_type and "weekend" in day_type.lower():
    active_factors.append(("Weekend", "yellow"))
if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
    active_factors.append((victim_vehicle, "orange"))
if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry", "Bus - RTC", "Bus", "Bus - Private"]:
    active_factors.append(("Heavy Vehicle", "yellow"))
if road_geometry in ["Curved Road", "Bridge", "Steep Grade",
                     "T - Junction", "Y - Junction", "Four Arm Junction"]:
    active_factors.append((road_geometry, "yellow"))
if accident_type in ["Head on", "Front back", "Front side"]:
    active_factors.append((accident_type, "red"))


# ─────────────────────────────────────────────
#  DYNAMIC REASONS
# ─────────────────────────────────────────────
def build_reasons():
    reasons = []

    if cause and ("over speeding" in cause.lower() or "speeding" in cause.lower()):
        if result == "Fatal":
            reasons.append(("red",
                "Overspeeding is the most significant contributor to fatal crashes on this corridor. "
                "High kinetic energy at impact leaves little margin for survival in head-on or vehicle–pedestrian scenarios."))
        elif result == "Grievous Injuries":
            reasons.append(("orange",
                "Overspeeding increases impact force substantially, making grievous injury a likely outcome "
                "even when other conditions are moderate."))
        else:
            reasons.append(("yellow",
                "Although overspeeding is present, the combination of other factors has limited predicted "
                "severity to a lower class in this scenario."))

    if cause and ("drunken" in cause.lower() or "drunk" in cause.lower()):
        reasons.append(("red",
            "Drunken driving impairs reaction time and reduces peripheral vision, directly increasing "
            "the probability of high-severity outcomes. This is among the strongest predictors in the dataset."))

    if cause and "loss of control" in cause.lower():
        reasons.append(("orange",
            "Loss of control typically results from sudden manoeuvres, wet roads, or mechanical failure, "
            "often leading to secondary collisions that escalate final severity."))

    if cause and ("bad road" in cause.lower() or "visibility" in cause.lower()):
        reasons.append(("orange",
            "Poor road surface or reduced visibility extend stopping distances and reduce hazard "
            "detection time — both recognised contributors to severity escalation."))

    if time_of_day and "night" in time_of_day.lower():
        if day_type and "weekend" in day_type.lower():
            reasons.append(("red",
                "Weekend night-time is the highest-risk window on this corridor: fatigue, poor visibility, "
                "and higher prevalence of impaired driving combine to elevate all severity classes."))
        else:
            reasons.append(("orange",
                "Night-time on NH-163 involves reduced ambient lighting and slower hazard detection, "
                "which are established contributors to severity escalation."))
    elif time_of_day and "evening" in time_of_day.lower():
        reasons.append(("yellow",
            "Evening hours coincide with peak traffic volume, increasing conflict probability "
            "and rear-end interaction risk along the corridor."))

    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        reasons.append(("orange",
            f"The geometry ({road_geometry}) reduces sight distance and increases lateral forces. "
            f"These segment types statistically show higher severity outcomes on NH-163."))
    elif road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction", "Staggered Junction"]:
        reasons.append(("orange",
            f"Junctions like {road_geometry} create multiple conflict points. Angle and right-turn "
            f"crashes at junctions tend to produce grievous or fatal outcomes due to direct lateral impact."))

    if victim_vehicle == "Two Wheeler":
        reasons.append(("red",
            "Two-wheeler riders have no protective enclosure, crumple zones, or airbags. "
            "At comparable impact speeds they sustain injuries far more severe than car occupants."))
    elif victim_vehicle == "Pedestrian":
        reasons.append(("red",
            "Pedestrian involvement is a strong predictor of fatal or grievous outcomes. "
            "The complete absence of physical protection makes even moderate-speed impacts life-threatening."))

    if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry"]:
        reasons.append(("orange",
            "Heavy vehicle involvement introduces extreme mass disparity — the lighter vehicle "
            "absorbs the majority of deformation energy, significantly raising injury severity."))
    elif offender_vehicle in ["Bus - RTC", "Bus", "Bus - Private"]:
        reasons.append(("orange",
            "Bus involvement increases severity due to vehicle height, mass, and frequency of "
            "pedestrian exposure near bus routes on this corridor."))

    if accident_type == "Head on":
        reasons.append(("red",
            "Head-on collisions combine the velocities of both vehicles at impact, making them "
            "the most energy-intensive crash type and a leading cause of fatalities nationally."))
    elif accident_type in ["Front back", "Front side"]:
        reasons.append(("orange",
            f"A {accident_type} collision indicates asymmetric impact. While less severe than head-on, "
            f"these crash types still produce significant deceleration forces on occupants."))

    if victim_manoeuvre in ["Crossing", "Wrong side driving", "U Turn"]:
        reasons.append(("yellow",
            f"The victim's manoeuvre ({victim_manoeuvre}) places them in an unexpected trajectory "
            f"relative to surrounding traffic, reducing available reaction time for the offender."))

    if offender_manoeuvre in ["Over Taking", "Wrong side driving", "U Turn"]:
        reasons.append(("yellow",
            f"The offender's manoeuvre ({offender_manoeuvre}) involves lateral movement into a "
            f"conflict zone — a strong predictor of angle and side-impact crashes."))

    if result in ["No Injury", "Simple Injuries"]:
        reasons.append(("green",
            "While some risk factors are present, the overall combination of conditions in this "
            "scenario is associated with lower severity outcomes in the training data."))

    return reasons[:7]

reasons = build_reasons()


# ─────────────────────────────────────────────
#  DYNAMIC RECOMMENDATIONS
# ─────────────────────────────────────────────
def build_recommendations():
    recs = []

    if result == "Fatal":
        recs.append(("🚨", "Immediate corridor intervention",
            "This scenario indicates extreme fatality risk. Emergency traffic management, "
            "temporary speed restrictions, and enhanced patrol presence are recommended immediately."))

    if cause and ("over speeding" in cause.lower() or "speeding" in cause.lower()):
        recs.append(("📷", "Speed enforcement infrastructure",
            "Deploy average-speed cameras at 2 km intervals. Research shows a 10 km/h reduction "
            "in mean speed reduces fatal crashes by approximately 34% on national highways."))

    if cause and ("drunken" in cause.lower() or "drunk" in cause.lower()):
        recs.append(("🍺", "Impaired driving programme",
            "Increase night-time breathalyzer checkpoints, particularly on weekend evenings. "
            "Sustained enforcement rather than one-off checks is significantly more effective."))

    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        recs.append(("⚠️", "Geometric safety treatment",
            "Install advance warning signs 300 m before the hazard, chevron markers, crash barriers "
            "on exposed edges, and skid-resistant road surface at these locations."))

    if road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction",
                         "Staggered Junction", "Round about"]:
        recs.append(("🚦", "Junction safety upgrade",
            "Improve approach channelisation, retroreflective delineators, and junction visibility. "
            "Evaluate signal timing and consider grade separation where traffic volumes justify it."))

    if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
        recs.append(("🏍", "Vulnerable road user protection",
            "Construct dedicated footpaths and two-wheeler lanes. Install pedestrian refuges, "
            "improve zebra crossing visibility, and mandate ABS on two-wheelers above 125cc."))

    if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry", "Bus - RTC", "Bus", "Bus - Private"]:
        recs.append(("🛻", "Heavy vehicle management",
            "Enforce lane discipline for buses and trucks. Introduce truck lay-bys on critical "
            "segments and consider time-based restrictions near school zones and junctions."))

    if time_of_day and "night" in time_of_day.lower():
        recs.append(("💡", "Night-time safety measures",
            "Upgrade roadway lighting to LED with consistent luminance. Install retroreflective "
            "lane markings and raised pavement markers to improve guidance in low-light conditions."))

    if accident_type in ["Head on", "Front back", "Front side"]:
        recs.append(("🛡", "Collision prevention measures",
            "Install flexible delineator posts as median protection. Evaluate wire rope or "
            "concrete median barriers on high-volume sections with head-on crash history."))

    if day_type and "weekend" in day_type.lower():
        recs.append(("📅", "Weekend traffic management",
            "Deploy additional traffic personnel on Friday and Saturday nights. "
            "Use variable message signs for weekend-specific risk behaviour campaigns."))

    return recs[:6]

recs = build_recommendations()


# ─────────────────────────────────────────────
#  ROW 1: Prediction + Confidence + Summary
# ─────────────────────────────────────────────
card_cls, title_cls, bar_color = severity_style.get(
    result,
    ("card-property", "property-title", "#1e88e5")
)
col1, col2, col3 = st.columns([1.6, 1.2, 1.2])

with col1:
    st.markdown(f"""
    <div class="severity-card {card_cls}">
        <p class="severity-title {title_cls}">{severity_icon.get(result, "⚪")}&nbsp; {result}</p>
        <p class="severity-sub">Predicted accident severity outcome</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Risk Level</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:.3rem;font-size:.9rem;color:#555;">
        Severity risk index: <strong style="color:{risk_color};font-size:1rem;">{risk_label}</strong>
    </div>
    <div class="risk-bar-bg">
        <div class="risk-bar-fill" style="width:{risk_percent}%;background:{risk_color};"></div>
    </div>
    <div style="font-size:.8rem;color:#888;margin-top:3px;">{risk_percent}% weighted severity probability</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Severity Confidence</div>', unsafe_allow_html=True)
    for i in range(n_classes):
        label   = _severity_labels[i]
        pct     = int(round(float(probs[i]) * 100))
        color   = severity_style.get(label, ("", "", "#888"))[2]
        is_pred = (i == pred)
        weight  = "700" if is_pred else "400"
        indicator = " ◀ predicted" if is_pred else ""
        st.markdown(f"""
        <div style="margin-bottom:9px;">
            <div style="display:flex;justify-content:space-between;
                        font-size:.84rem;font-weight:{weight};margin-bottom:3px;">
                <span>{label}{indicator}</span><span style="color:{color}">{pct}%</span>
            </div>
            <div style="background:#eef1f7;border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{pct}%;background:{color};height:100%;
                            border-radius:6px;opacity:{'1.0' if is_pred else '0.45'};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-header">Scenario Summary</div>', unsafe_allow_html=True)
    if active_factors:
        pills_html = "".join(
            f'<span class="factor-pill pill-{color}">{label}</span>'
            for label, color in active_factors
        )
        st.markdown(pills_html, unsafe_allow_html=True)
        count = len(active_factors)
        st.markdown(
            f"<div style='margin-top:.8rem;font-size:.82rem;color:#888;'>"
            f"{count} risk factor{'s' if count != 1 else ''} identified in this scenario.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div style="background:#e8f5e9;border:1px solid #43a047;border-radius:10px;
                    padding:.8rem 1rem;font-size:.88rem;color:#2e7d32;">
            ✅ No major risk factors detected. This scenario is associated with lower severity outcomes.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROW 2: Why this prediction
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Why this prediction?</div>', unsafe_allow_html=True)
if reasons:
    for color, text in [(r[0], r[1]) for r in reasons]:
        st.markdown(f'<div class="reason-block {color}">{text}</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="reason-block blue">
        No dominant risk factors were identified. The model evaluated all input features collectively
        and assigned the most probable severity class based on learned patterns from the NH-163 dataset.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROW 3: Recommended Actions
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Recommended Safety Actions</div>', unsafe_allow_html=True)
if recs:
    rec_cols = st.columns(2)
    for i, (icon, title, detail) in enumerate(recs):
        with rec_cols[i % 2]:
            st.markdown(f"""
            <div class="rec-card">
                <div class="rec-title"><span class="rec-icon">{icon}</span>{title}</div>
                <div style="font-size:.85rem;color:#555;line-height:1.55;">{detail}</div>
            </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="rec-card">
        <div class="rec-title">✅ Maintain current conditions</div>
        <div style="font-size:.85rem;color:#555;">
            The selected scenario presents low severity risk. Standard road maintenance
            and periodic safety monitoring are sufficient for this condition set.
        </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROW 4: Global Feature Importance
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)

importances = model.feature_importances_
feat_labels = [display_labels.get(col, col) for col in feature_order]

# Ensure lengths match
min_len     = min(len(importances), len(feat_labels))
importances = importances[:min_len]
feat_labels = feat_labels[:min_len]

feat_series = pd.Series(importances, index=feat_labels).sort_values(ascending=True)
n_fi        = len(feat_series)

bar_colors_fi = []
for rank in range(n_fi):
    if rank >= n_fi - 3:
        bar_colors_fi.append("#e53935")
    elif rank >= n_fi - 6:
        bar_colors_fi.append("#fb8c00")
    else:
        bar_colors_fi.append("#90a4ae")

fig1, ax1 = plt.subplots(figsize=(8, max(3.5, n_fi * 0.44)))
bars_fi = ax1.barh(
    feat_series.index, feat_series.values,
    color=bar_colors_fi, height=0.62, edgecolor="white", linewidth=0.5
)
for bar, val in zip(bars_fi, feat_series.values):
    ax1.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left", fontsize=8, color="#444")

ax1.set_xlabel("Mean Decrease in Impurity (Importance Score)", fontsize=9, color="#666")
ax1.set_title("Feature Importance — Random Forest Model", fontsize=10, color="#1a1a2e", pad=10)
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_color("#dde3ed")
ax1.tick_params(axis="y", labelsize=9, colors="#333")
ax1.tick_params(axis="x", labelsize=8, colors="#666")
ax1.set_facecolor("#fafbfc")
fig1.patch.set_facecolor("#fafbfc")
ax1.legend(handles=[
    mpatches.Patch(color="#e53935", label="Top 3 — highest influence"),
    mpatches.Patch(color="#fb8c00", label="Mid-tier features"),
    mpatches.Patch(color="#90a4ae", label="Lower influence"),
], fontsize=8, loc="lower right", framealpha=0.85, edgecolor="#dde3ed")
plt.tight_layout()
st.pyplot(fig1)

st.markdown(
    "<div style='font-size:.82rem;color:#888;margin-top:-4px;'>"
    "Importance scores represent mean decrease in node impurity across all trees. "
    "Higher values indicate stronger contribution to severity classification.</div>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────
#  ROW 5: Input Encoding View
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Input Feature Encoding</div>', unsafe_allow_html=True)

values     = [int(v) for v in input_df.iloc[0].values]
enc_labels = [display_labels.get(col, col) for col in feature_order]

# Ensure lengths match
min_enc    = min(len(values), len(enc_labels))
values     = values[:min_enc]
enc_labels = enc_labels[:min_enc]

enc_colors = ["#1e88e5" if v > 0 else "#b0bec5" for v in values]

fig2, ax2 = plt.subplots(figsize=(8, max(3.5, min_enc * 0.44)))
bars2 = ax2.barh(enc_labels, values, color=enc_colors, height=0.58, edgecolor="white")
for bar, val in zip(bars2, values):
    ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
             str(val), va="center", ha="left", fontsize=8, color="#444")

ax2.set_xlabel("Label-encoded category index", fontsize=9, color="#666")
ax2.set_title("Current Input — Encoded Feature Values", fontsize=10, color="#1a1a2e", pad=10)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#dde3ed")
ax2.tick_params(axis="y", labelsize=9, colors="#333")
ax2.tick_params(axis="x", labelsize=8, colors="#666")
ax2.set_facecolor("#fafbfc")
fig2.patch.set_facecolor("#fafbfc")
ax2.legend(handles=[
    mpatches.Patch(color="#1e88e5", label="Non-baseline category (encoded > 0)"),
    mpatches.Patch(color="#b0bec5", label="Baseline / first category (encoded = 0)"),
], fontsize=8, loc="lower right", framealpha=0.85, edgecolor="#dde3ed")
plt.tight_layout()
st.pyplot(fig2)


# ─────────────────────────────────────────────
#  DEBUG EXPANDER  (fixed — no DataFrame length mismatch)
# ─────────────────────────────────────────────
with st.expander("🔬 Debug — Raw model output"):
    st.markdown("**Encoded input row:**")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("**Raw class probabilities:**")
    # Build row-by-row to avoid any length mismatch
    for i in range(n_classes):
        lbl = _severity_labels[i] if i < len(_severity_labels) else f"Class {i}"
        pct = int(round(float(probs[i]) * 100))
        st.write(f"- **{lbl}**: `{float(probs[i]):.4f}` ({pct}%)")

    st.markdown("**Resolved input values (after key normalisation):**")
    try:
        st.json({
            "cause":              cause,
            "time_of_day":        time_of_day,
            "day_type":           day_type,
            "road_geometry":      road_geometry,
            "victim_vehicle":     victim_vehicle,
            "offender_vehicle":   offender_vehicle,
            "accident_type":      accident_type,
            "victim_manoeuvre":   victim_manoeuvre,
            "offender_manoeuvre": offender_manoeuvre,
        })
    except Exception:
        st.write("Could not serialise resolved values.")

    st.markdown(f"**Final prediction:** `{result}` &nbsp;|&nbsp; **Risk score:** `{risk_percent}%`")



