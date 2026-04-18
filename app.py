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

    [data-testid="stSidebar"] { background:#f4f6fa; border-right:1px solid #dde3ed; }
    [data-testid="stSidebar"] .block-container { padding-top:1.5rem; }

    .severity-card { border-radius:16px; padding:1.5rem 1.8rem; margin-bottom:1rem; border:2px solid; }
    .card-fatal    { background:linear-gradient(135deg,#fff1f1 60%,#ffe0e0); border-color:#e53935; }
    .card-grievous { background:linear-gradient(135deg,#fff8e1 60%,#fff0c0); border-color:#fb8c00; }
    .card-minor    { background:linear-gradient(135deg,#e8f5e9 60%,#d4edda); border-color:#43a047; }
    .card-property { background:linear-gradient(135deg,#e3f2fd 60%,#cce5ff); border-color:#1e88e5; }

    .severity-title { font-size:2.1rem; font-weight:700; margin:0 0 .3rem; letter-spacing:-0.5px; }
    .severity-sub   { font-size:.92rem; color:#666; margin:0; }
    .fatal-title    { color:#b71c1c; }
    .grievous-title { color:#e65100; }
    .minor-title    { color:#2e7d32; }
    .property-title { color:#1565c0; }

    .risk-bar-bg {
        background:#e8ecf2; border-radius:10px; height:20px; width:100%;
        margin:8px 0 4px; overflow:hidden; box-shadow:inset 0 1px 3px rgba(0,0,0,.08);
    }
    .risk-bar-fill { height:100%; border-radius:10px; }

    .factor-pill {
        display:inline-block; padding:4px 13px; border-radius:20px;
        font-size:.8rem; font-weight:600; margin:3px 4px 3px 0;
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
        background:#fff; border:1px solid #dde3ed; border-radius:12px;
        padding:.9rem 1.1rem; margin-bottom:.6rem; font-size:.88rem;
        line-height:1.55; box-shadow:0 1px 4px rgba(0,0,0,.04);
    }
    .rec-title { font-weight:700; font-size:.92rem; margin-bottom:5px; color:#1a1a2e; }
    .rec-icon  { font-size:1.1rem; margin-right:7px; }

    .section-header {
        font-size:1.05rem; font-weight:700; color:#1a1a2e;
        border-bottom:2px solid #eef1f7; padding-bottom:.4rem;
        margin:1.5rem 0 .9rem;
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

feature_order   = metadata["feature_order"]        # list of raw column names
severity_labels = metadata["severity_labels"]      # list e.g. ["No Injury","Simple Injuries","Grievous Injuries","Fatal"]
display_labels  = metadata["display_labels"]       # dict raw_col -> human label

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
#  SIDEBAR INPUTS  — ALL features from feature_order
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

# ─────────────────────────────────────────────
#  WELCOME STATE
# ─────────────────────────────────────────────
if not predict_btn:
    c1, c2, c3 = st.columns(3)
    for col, (icon, title, body) in zip([c1, c2, c3], [
        ("🗂️", "Real Dataset",   "Trained on official NH-163 corridor accident records."),
        ("🌲", "Random Forest",  "Ensemble of decision trees with calibrated severity confidence."),
        ("🧠", "Explainable AI", "Every prediction includes input-specific reasons and safety recommendations."),
    ]):
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
    val = int(encoders[col].transform([user_inputs[col]])[0])
    encoded_row.append(val)

input_df  = pd.DataFrame([encoded_row], columns=feature_order)
pred      = int(model.predict(input_df)[0])
probs     = model.predict_proba(input_df)[0]          # numpy array
n_classes = len(probs)

# Safely align severity_labels with probs length
_sev_labels = list(severity_labels)
while len(_sev_labels) < n_classes:
    _sev_labels.append(f"Class {len(_sev_labels)}")
_sev_labels = _sev_labels[:n_classes]

result = _sev_labels[pred] if pred < n_classes else f"Class {pred}"

# Probability-weighted risk score
_rw       = {0: 0.05, 1: 0.35, 2: 0.70, 3: 1.00}
risk_pct  = int(min(sum(float(probs[i]) * _rw.get(i, 1.0) for i in range(n_classes)) * 100, 100))
risk_color = "#e53935" if risk_pct >= 65 else "#fb8c00" if risk_pct >= 40 else "#fdd835" if risk_pct >= 20 else "#43a047"
risk_label = "Very High" if risk_pct >= 65 else "High" if risk_pct >= 40 else "Moderate" if risk_pct >= 20 else "Low"

# ─────────────────────────────────────────────
#  ROBUST KEY LOOKUP  (handles spaces/underscores/case)
# ─────────────────────────────────────────────
_norm = {k.lower().replace(" ", "").replace("_", ""): k for k in user_inputs}

def _get(*candidates):
    for c in candidates:
        k = _norm.get(c.lower().replace(" ", "").replace("_", ""))
        if k is not None:
            return user_inputs[k]
    return ""

cause              = _get("Cause", "cause_of_accident", "causeofaccident", "accidentcause")
time_of_day        = _get("Time of Day", "TimeofDay", "time_of_day", "timeofday", "time")
day_type           = _get("Day Type", "DayType", "day_type", "daytype")
road_geometry      = _get("Road Geometry", "RoadGeometry", "road_geometry", "roadgeometry", "geometry")
victim_vehicle     = _get("Victim Vehicle Type", "VictimVehicleType", "victim_vehicle_type", "victimvehicle")
offender_vehicle   = _get("Offending Vehicle Type", "OffendingVehicleType", "offending_vehicle_type", "offendervehicle")
victim_manoeuvre   = _get("Victim Manoeuvre", "VictimManoeuvre", "victim_manoeuvre", "victimmanoeuvre")
offender_manoeuvre = _get("Offender Manoeuvre", "OffenderManoeuvre", "offender_manoeuvre", "offendermanoeuvre")
accident_type      = _get("Type of Accident", "TypeofAccident", "type_of_accident", "typeofaccident", "accidenttype")

# ─────────────────────────────────────────────
#  ACTIVE RISK FACTORS  (scenario summary pills)
# ─────────────────────────────────────────────
active_factors = []
c_lo = cause.lower()
if "over speeding" in c_lo or "speeding" in c_lo:
    active_factors.append(("Over Speeding",  "red"))
if "drunken" in c_lo or "drunk" in c_lo:
    active_factors.append(("Drunken Driving","red"))
if "loss of control" in c_lo:
    active_factors.append(("Loss of Control","orange"))
if "bad road" in c_lo or "visibility" in c_lo:
    active_factors.append(("Road/Visibility","orange"))
if time_of_day and "night" in time_of_day.lower():
    active_factors.append(("Night-time",     "orange"))
if day_type and "weekend" in day_type.lower():
    active_factors.append(("Weekend",        "yellow"))
if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
    active_factors.append((victim_vehicle,   "orange"))
if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry", "Bus - RTC", "Bus", "Bus - Private"]:
    active_factors.append(("Heavy Vehicle",  "yellow"))
if road_geometry in ["Curved Road", "Bridge", "Steep Grade",
                     "T - Junction", "Y - Junction", "Four Arm Junction"]:
    active_factors.append((road_geometry,    "yellow"))
if accident_type in ["Head on", "Front back", "Front side"]:
    active_factors.append((accident_type,    "red"))

# ─────────────────────────────────────────────
#  DYNAMIC REASONS  (context-aware, severity-sensitive)
# ─────────────────────────────────────────────
def build_reasons():
    r = []
    c = cause.lower()

    if "over speeding" in c or "speeding" in c:
        if result == "Fatal":
            r.append(("red",
                "Overspeeding is the dominant contributor to fatal crashes on this corridor. "
                "High kinetic energy at impact leaves little margin for survival in head-on or vehicle–pedestrian scenarios."))
        elif result == "Grievous Injuries":
            r.append(("orange",
                "Overspeeding substantially increases impact force, making grievous injury likely "
                "even when other conditions are only moderately adverse."))
        else:
            r.append(("yellow",
                "Although overspeeding is present, the combination of other factors has limited "
                "predicted severity to a lower class in this scenario."))

    if "drunken" in c or "drunk" in c:
        r.append(("red",
            "Drunken driving impairs reaction time by up to 30% and reduces peripheral vision, "
            "directly increasing the probability of high-severity outcomes — one of the strongest predictors in the dataset."))

    if "loss of control" in c:
        r.append(("orange",
            "Loss of control typically results from sudden manoeuvres, wet roads, or mechanical failure, "
            "often leading to secondary collisions that escalate final severity."))

    if "bad road" in c or "visibility" in c:
        r.append(("orange",
            "Poor road surface or reduced visibility extend stopping distances and delay hazard detection — "
            "both well-established contributors to severity escalation."))

    if time_of_day and "night" in time_of_day.lower():
        if day_type and "weekend" in day_type.lower():
            r.append(("red",
                "Weekend night-time is the highest-risk window on this corridor: fatigue, poor visibility, "
                "and elevated prevalence of impaired driving compound all severity factors."))
        else:
            r.append(("orange",
                "Night-time driving on NH-163 involves reduced ambient lighting and slower hazard detection, "
                "both established contributors to higher severity outcomes."))
    elif time_of_day and "evening" in time_of_day.lower():
        r.append(("yellow",
            "Evening hours coincide with peak traffic volume, increasing conflict probability "
            "and rear-end interaction risk along the corridor."))

    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        r.append(("orange",
            f"The geometry ({road_geometry}) reduces sight distance and increases lateral forces on vehicles. "
            f"These segment types statistically show higher severity outcomes on NH-163."))
    elif road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction", "Staggered Junction"]:
        r.append(("orange",
            f"Junctions like {road_geometry} create multiple conflict points. Angle and right-turn crashes "
            f"at junctions produce grievous or fatal outcomes due to direct lateral impact."))

    if victim_vehicle == "Two Wheeler":
        r.append(("red",
            "Two-wheeler riders have no protective enclosure, crumple zones, or airbags. "
            "At comparable impact speeds they sustain injuries far more severe than car occupants."))
    elif victim_vehicle == "Pedestrian":
        r.append(("red",
            "Pedestrian involvement is a strong predictor of fatal or grievous outcomes. "
            "The complete absence of physical protection makes even moderate-speed impacts life-threatening."))

    if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry"]:
        r.append(("orange",
            "Heavy vehicle involvement introduces extreme mass disparity — the lighter vehicle absorbs "
            "the majority of deformation energy, significantly raising injury severity."))
    elif offender_vehicle in ["Bus - RTC", "Bus", "Bus - Private"]:
        r.append(("orange",
            "Bus involvement increases severity due to vehicle height, mass, and frequency of pedestrian "
            "exposure near bus routes on this corridor."))

    if accident_type == "Head on":
        r.append(("red",
            "Head-on collisions combine the velocities of both vehicles at impact, making them the most "
            "energy-intensive crash type and a leading cause of fatalities nationally."))
    elif accident_type in ["Front back", "Front side"]:
        r.append(("orange",
            f"A {accident_type} collision indicates asymmetric impact. While less severe than head-on, "
            f"these crash types still produce significant deceleration forces on occupants."))

    if victim_manoeuvre in ["Crossing", "Wrong side driving", "U Turn"]:
        r.append(("yellow",
            f"The victim's manoeuvre ({victim_manoeuvre}) places them in an unexpected trajectory, "
            f"reducing available reaction time for the offender."))

    if offender_manoeuvre in ["Over Taking", "Wrong side driving", "U Turn"]:
        r.append(("yellow",
            f"The offender's manoeuvre ({offender_manoeuvre}) involves lateral movement into a conflict zone — "
            f"a strong predictor of angle and side-impact crashes."))

    if result in ["No Injury", "Simple Injuries"]:
        r.append(("green",
            "While some risk factors are present, the overall combination of conditions in this scenario "
            "is associated with lower severity outcomes in the training data."))

    return r[:7]

reasons = build_reasons()

# ─────────────────────────────────────────────
#  DYNAMIC RECOMMENDATIONS
# ─────────────────────────────────────────────
def build_recommendations():
    r = []
    c = cause.lower()

    if result == "Fatal":
        r.append(("🚨", "Immediate corridor intervention",
            "This scenario indicates extreme fatality risk. Emergency traffic management, "
            "temporary speed restrictions, and enhanced patrol presence are recommended immediately."))

    if "over speeding" in c or "speeding" in c:
        r.append(("📷", "Speed enforcement infrastructure",
            "Deploy average-speed cameras at 2 km intervals. A 10 km/h reduction in mean speed "
            "reduces fatal crashes by approximately 34% on national highways."))

    if "drunken" in c or "drunk" in c:
        r.append(("🍺", "Impaired driving programme",
            "Increase night-time breathalyzer checkpoints, particularly on weekend evenings. "
            "Sustained enforcement rather than one-off checks is significantly more effective."))

    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        r.append(("⚠️", "Geometric safety treatment",
            "Install advance warning signs 300 m before the hazard, chevron markers, crash barriers "
            "on exposed edges, and skid-resistant road surface at these locations."))

    if road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction",
                         "Staggered Junction", "Round about"]:
        r.append(("🚦", "Junction safety upgrade",
            "Improve approach channelisation, retroreflective delineators, and junction visibility. "
            "Evaluate signal timing and grade separation where traffic volumes justify it."))

    if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
        r.append(("🏍", "Vulnerable road user protection",
            "Construct dedicated footpaths and two-wheeler lanes. Install pedestrian refuges, "
            "improve zebra crossing visibility, and mandate ABS on two-wheelers above 125cc."))

    if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry", "Bus - RTC", "Bus", "Bus - Private"]:
        r.append(("🛻", "Heavy vehicle management",
            "Enforce lane discipline for buses and trucks. Introduce truck lay-bys on critical segments "
            "and consider time-based restrictions near school zones and junctions."))

    if time_of_day and "night" in time_of_day.lower():
        r.append(("💡", "Night-time safety measures",
            "Upgrade roadway lighting to LED with consistent luminance. Install retroreflective lane "
            "markings and raised pavement markers to improve guidance in low-light conditions."))

    if accident_type in ["Head on", "Front back", "Front side"]:
        r.append(("🛡", "Collision prevention measures",
            "Install flexible delineator posts as median protection. Evaluate wire rope or concrete "
            "median barriers on high-volume sections with head-on crash history."))

    if day_type and "weekend" in day_type.lower():
        r.append(("📅", "Weekend traffic management",
            "Deploy additional traffic personnel on Friday and Saturday nights. "
            "Use variable message signs for weekend-specific risk behaviour campaigns."))

    if not r:
        r.append(("✅", "Maintain current conditions",
            "The selected scenario presents low severity risk. Standard road maintenance "
            "and periodic safety monitoring are sufficient for this condition set."))

    return r[:6]

recs = build_recommendations()

# ─────────────────────────────────────────────
#  ROW 1 — Prediction card + Confidence + Summary
# ─────────────────────────────────────────────
_style   = severity_style.get(result, ("card-property", "property-title", "#1e88e5"))
card_cls, title_cls, _ = _style

col1, col2, col3 = st.columns([1.6, 1.2, 1.2])

with col1:
    st.markdown(f"""
    <div class="severity-card {card_cls}">
        <p class="severity-title {title_cls}">{severity_icon.get(result,"⚪")}&nbsp;{result}</p>
        <p class="severity-sub">Predicted accident severity outcome</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Risk Level</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:.3rem;font-size:.9rem;color:#555;">
        Severity risk index: <strong style="color:{risk_color};font-size:1rem;">{risk_label}</strong>
    </div>
    <div class="risk-bar-bg">
        <div class="risk-bar-fill" style="width:{risk_pct}%;background:{risk_color};"></div>
    </div>
    <div style="font-size:.8rem;color:#888;margin-top:3px;">{risk_pct}% weighted severity probability</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Severity Confidence</div>', unsafe_allow_html=True)
    for i in range(n_classes):
        lbl     = _sev_labels[i]
        pct     = int(round(float(probs[i]) * 100))
        clr     = severity_style.get(lbl, ("","","#888"))[2]
        is_pred = (i == pred)
        wt      = "700" if is_pred else "400"
        tag     = " ◀ predicted" if is_pred else ""
        st.markdown(f"""
        <div style="margin-bottom:9px;">
            <div style="display:flex;justify-content:space-between;
                        font-size:.84rem;font-weight:{wt};margin-bottom:3px;">
                <span>{lbl}{tag}</span><span style="color:{clr}">{pct}%</span>
            </div>
            <div style="background:#eef1f7;border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{pct}%;background:{clr};height:100%;border-radius:6px;
                            opacity:{'1.0' if is_pred else '0.45'};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-header">Scenario Summary</div>', unsafe_allow_html=True)
    if active_factors:
        st.markdown(
            "".join(f'<span class="factor-pill pill-{clr}">{lbl}</span>'
                    for lbl, clr in active_factors),
            unsafe_allow_html=True
        )
        n_af = len(active_factors)
        st.markdown(
            f"<div style='margin-top:.8rem;font-size:.82rem;color:#888;'>"
            f"{n_af} risk factor{'s' if n_af != 1 else ''} identified.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div style="background:#e8f5e9;border:1px solid #43a047;border-radius:10px;
                    padding:.8rem 1rem;font-size:.88rem;color:#2e7d32;">
            ✅ No major risk factors detected. This scenario is associated with lower severity outcomes.
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ROW 2 — Why this prediction
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Why this prediction?</div>', unsafe_allow_html=True)
if reasons:
    for clr, txt in [(r[0], r[1]) for r in reasons]:
        st.markdown(f'<div class="reason-block {clr}">{txt}</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="reason-block blue">
        No dominant risk factors were identified. The model evaluated all features collectively
        and assigned the most probable severity class based on learned patterns from the NH-163 dataset.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ROW 3 — Recommended Actions
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Recommended Safety Actions</div>', unsafe_allow_html=True)
rc1, rc2 = st.columns(2)
for i, (icon, title, detail) in enumerate(recs):
    with (rc1 if i % 2 == 0 else rc2):
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title"><span class="rec-icon">{icon}</span>{title}</div>
            <div style="font-size:.85rem;color:#555;line-height:1.55;">{detail}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ROW 4 — Global Feature Importance  (colorful, labeled)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)

importances  = model.feature_importances_
feat_display = [display_labels.get(col, col) for col in feature_order]

# Clip to safe length
min_fi       = min(len(importances), len(feat_display))
importances  = importances[:min_fi]
feat_display = feat_display[:min_fi]

fi_series = pd.Series(importances, index=feat_display).sort_values(ascending=True)
n_fi      = len(fi_series)

# Color by rank: top 3 red, next 3 orange, rest blue-grey
fi_colors = []
for rank in range(n_fi):
    if rank >= n_fi - 3:
        fi_colors.append("#e53935")
    elif rank >= n_fi - 6:
        fi_colors.append("#fb8c00")
    else:
        fi_colors.append("#90a4ae")

fig1, ax1 = plt.subplots(figsize=(8, max(3.5, n_fi * 0.46)))
bars1 = ax1.barh(fi_series.index, fi_series.values,
                 color=fi_colors, height=0.62, edgecolor="white", linewidth=0.4)

for bar, val in zip(bars1, fi_series.values):
    ax1.text(val + 0.0015, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left", fontsize=8.5, color="#333")

ax1.set_xlabel("Mean Decrease in Impurity (Importance Score)", fontsize=9, color="#555")
ax1.set_title("Feature Importance — Random Forest Model", fontsize=10.5,
              color="#1a1a2e", pad=10, fontweight="bold")
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
], fontsize=8.5, loc="lower right", framealpha=0.88, edgecolor="#dde3ed")
plt.tight_layout()
st.pyplot(fig1)
st.markdown(
    "<div style='font-size:.82rem;color:#888;margin-top:-4px;'>"
    "Mean decrease in impurity across all trees. Higher values indicate stronger contribution "
    "to severity classification.</div>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
#  ROW 5 — Input Feature Encoding  (colorful, shows ALL features including Cause)
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Input Feature Encoding</div>', unsafe_allow_html=True)

enc_values  = [int(v) for v in input_df.iloc[0].values]
enc_display = [display_labels.get(col, col) for col in feature_order]

# Clip safely
min_enc     = min(len(enc_values), len(enc_display))
enc_values  = enc_values[:min_enc]
enc_display = enc_display[:min_enc]

# Color each bar by encoded value magnitude — 0 = grey baseline, else gradient from blue to red
max_enc_val = max(enc_values) if max(enc_values) > 0 else 1
enc_colors  = []
for v in enc_values:
    if v == 0:
        enc_colors.append("#b0bec5")
    elif v / max_enc_val >= 0.67:
        enc_colors.append("#e53935")
    elif v / max_enc_val >= 0.34:
        enc_colors.append("#fb8c00")
    else:
        enc_colors.append("#1e88e5")

fig2, ax2 = plt.subplots(figsize=(8, max(3.5, min_enc * 0.46)))
bars2 = ax2.barh(enc_display, enc_values,
                 color=enc_colors, height=0.58, edgecolor="white", linewidth=0.4)

for bar, val, raw_col in zip(bars2, enc_values, feature_order[:min_enc]):
    # Show both encoded number AND the original selected label
    original_label = user_inputs.get(raw_col, "")
    display_text   = f"{val}  ({original_label})" if original_label else str(val)
    ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
             display_text, va="center", ha="left", fontsize=7.8, color="#333")

ax2.set_xlabel("Label-encoded category index", fontsize=9, color="#555")
ax2.set_title("Current Input — Encoded Feature Values with Selected Labels",
              fontsize=10.5, color="#1a1a2e", pad=10, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color("#dde3ed")
ax2.tick_params(axis="y", labelsize=9, colors="#333")
ax2.tick_params(axis="x", labelsize=8, colors="#666")
ax2.set_facecolor("#fafbfc")
fig2.patch.set_facecolor("#fafbfc")
ax2.legend(handles=[
    mpatches.Patch(color="#e53935", label="High-index category (more adverse)"),
    mpatches.Patch(color="#fb8c00", label="Mid-index category"),
    mpatches.Patch(color="#1e88e5", label="Low-index category"),
    mpatches.Patch(color="#b0bec5", label="Baseline / first category (0)"),
], fontsize=8, loc="lower right", framealpha=0.88, edgecolor="#dde3ed")
plt.tight_layout()
st.pyplot(fig2)

# ─────────────────────────────────────────────
#  DEBUG EXPANDER  (no DataFrame — no length mismatch)
# ─────────────────────────────────────────────
with st.expander("🔬 Debug — Raw model output"):
    st.markdown("**Encoded input row:**")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("**Raw class probabilities:**")
    for i in range(n_classes):
        lbl = _sev_labels[i] if i < len(_sev_labels) else f"Class {i}"
        pct = int(round(float(probs[i]) * 100))
        st.write(f"- **{lbl}**: `{float(probs[i]):.4f}` ({pct}%)")

    st.markdown("**Resolved feature values used for logic:**")
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
        st.write({
            "cause": cause, "time_of_day": time_of_day,
            "day_type": day_type, "road_geometry": road_geometry,
        })

    st.markdown(f"**Final prediction:** `{result}` &nbsp;|&nbsp; **Risk score:** `{risk_pct}%`")

# ─────────────────────────────────────────────
#  PANEL DEMO GUIDE
# ─────────────────────────────────────────────
with st.expander("📋 Panel Demo — Suggested Input Scenarios"):
    st.markdown("""
**Run these in sequence to demonstrate all four severity classes and the feature importance chart.**

---
**Scenario 1 — Fatal (worst case)**
Cause: *Over Speeding* · Victim Vehicle: *Two Wheeler* · Offending Vehicle: *Truck/Lorry*
Type of Accident: *Head on* · Road Geometry: *Curved Road* · Time of Day: *Night* · Day Type: *Weekend*
Victim Manoeuvre: *Going Straight* · Offender Manoeuvre: *Over Taking*

*Shows: overspeeding + unprotected road user + heavy vehicle + night/weekend = Fatal + Very High risk.*

---
**Scenario 2 — Grievous Injuries**
Cause: *Drunken Driving* · Victim Vehicle: *Two Wheeler* · Offending Vehicle: *Car*
Type of Accident: *Front side* · Road Geometry: *T - Junction* · Time of Day: *Night* · Day Type: *Weekday*
Victim Manoeuvre: *Crossing* · Offender Manoeuvre: *Going Straight*

*Shows: impaired driving + junction conflict + two-wheeler = Grievous Injuries.*

---
**Scenario 3 — Simple Injuries (moderate)**
Cause: *Loss of Control* · Victim Vehicle: *Car* · Offending Vehicle: *Car*
Type of Accident: *Front back* · Road Geometry: *Straight Road* · Time of Day: *Afternoon* · Day Type: *Weekday*
Victim Manoeuvre: *Going Straight* · Offender Manoeuvre: *Going Straight*

*Shows: moderate rear-end scenario — model does not always predict extreme outcomes.*

---
**Scenario 4 — No Injury (baseline)**
Cause: *Bad Road Condition* · Victim Vehicle: *Car* · Offending Vehicle: *Car*
Type of Accident: *Front back* · Road Geometry: *Straight Road* · Time of Day: *Morning* · Day Type: *Weekday*
Victim Manoeuvre: *Going Straight* · Offender Manoeuvre: *Stationary*

*Shows: baseline low-risk condition → No Injury + Low risk score.*

---
**Scenario 5 — Live feature importance demo**
Start with Scenario 1 (Fatal). Change only Cause from *Over Speeding* to *Bad Road Condition*.
Observe the severity drop — this demonstrates that Cause is among the highest-ranked features.
    """)
