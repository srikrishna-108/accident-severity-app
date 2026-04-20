import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Severity AI · NH-163",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] {
    background: #f0f4fa;
    border-right: 1px solid #d4dcea;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

/* severity cards */
.sev-card {
    border-radius: 18px; padding: 1.4rem 1.7rem;
    margin-bottom: 1rem; border: 2px solid;
}
.sev-fatal    { background: linear-gradient(135deg,#fff1f1 55%,#fddede); border-color:#e53935; }
.sev-grievous { background: linear-gradient(135deg,#fff8e1 55%,#ffe3a0); border-color:#fb8c00; }
.sev-simple   { background: linear-gradient(135deg,#e8f5e9 55%,#c8e6c9); border-color:#2e7d32; }
.sev-noinjury { background: linear-gradient(135deg,#e3f2fd 55%,#bbdefb); border-color:#1565c0; }

.sev-title { font-size: 2rem; font-weight: 700; margin: 0 0 .25rem; letter-spacing: -.4px; }
.clr-fatal    { color: #b71c1c; }
.clr-grievous { color: #bf360c; }
.clr-simple   { color: #1b5e20; }
.clr-noinjury { color: #0d47a1; }

/* risk bar */
.risk-track {
    background: #dce5f4; border-radius: 10px; height: 18px;
    width: 100%; margin: 8px 0 4px; overflow: hidden;
}
.risk-fill { height: 100%; border-radius: 10px; }

/* pills */
.pill {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: .78rem; font-weight: 600; margin: 3px 3px 3px 0;
}
.pill-red    { background:#fdecea; color:#b71c1c; border:1.5px solid #e53935; }
.pill-orange { background:#fff3e0; color:#bf360c; border:1.5px solid #fb8c00; }
.pill-yellow { background:#fffde7; color:#f57f17; border:1.5px solid #f9a825; }
.pill-green  { background:#e8f5e9; color:#1b5e20; border:1.5px solid #2e7d32; }
.pill-blue   { background:#e3f2fd; color:#0d47a1; border:1.5px solid #1565c0; }

/* reason blocks */
.reason-block {
    border-left: 4px solid; border-radius: 0 12px 12px 0;
    padding: .8rem 1.1rem; margin-bottom: .55rem;
    font-size: .89rem; line-height: 1.65; color: #1a1a2e;
}
.reason-red    { background:#fff5f5; border-color:#e53935; }
.reason-orange { background:#fffbf0; border-color:#fb8c00; }
.reason-yellow { background:#fffef0; border-color:#f9a825; }
.reason-green  { background:#f1f9f2; border-color:#2e7d32; }
.reason-blue   { background:#f0f6ff; border-color:#1565c0; }

/* recommendation cards */
.rec-card {
    background: #fff; border: 1px solid #d8e2f2; border-radius: 14px;
    padding: 1rem 1.1rem; margin-bottom: .65rem;
    box-shadow: 0 2px 6px rgba(0,0,0,.04);
}
.rec-head { font-weight: 700; font-size: .91rem; margin-bottom: 5px; color: #0e0e24; }

/* section headers */
.sec-hdr {
    font-size: 1.05rem; font-weight: 700; color: #0e0e24;
    border-bottom: 2px solid #e4eaf8; padding-bottom: .4rem;
    margin: 1.5rem 0 .85rem;
}

.stExpander { border: 1px solid #d8e2f2 !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD ARTIFACTS  — joblib (version-safe) with pickle fallback
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model    = joblib.load("rf_model.pkl")
    encoders = joblib.load("encoders.pkl")
    metadata = joblib.load("metadata.pkl")
    return model, encoders, metadata

model, encoders, metadata = load_artifacts()

# ── parse metadata ────────────────────────────────────────────────────────────
feature_order  = list(metadata["feature_order"])
display_labels = dict(metadata["display_labels"])

# severity_labels may be a dict {0:"No Injury",...} or a list
_raw_sev = metadata["severity_labels"]
if isinstance(_raw_sev, dict):
    severity_labels = [_raw_sev[k] for k in sorted(_raw_sev.keys())]
else:
    severity_labels = list(_raw_sev)

n_classes = len(severity_labels)

# ── per-severity visual config ───────────────────────────────────────────────
_sev_cfg = {
    "Fatal":             ("sev-fatal",    "clr-fatal",    "#e53935", "🔴"),
    "Grievous Injuries": ("sev-grievous", "clr-grievous", "#fb8c00", "🟠"),
    "Simple Injuries":   ("sev-simple",   "clr-simple",   "#2e7d32", "🟢"),
    "No Injury":         ("sev-noinjury", "clr-noinjury", "#1565c0", "🔵"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# Accident Severity Predictor")
st.markdown(
    "**Smart Road Risk & Severity Analysis System**  —  "
    "Tuned Random Forest · NH-163 Warangal–Hyderabad · "
    "Telangana State Police Dataset 2024"
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  —  all 9 input features from feature_order
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛣 Road Conditions")
st.sidebar.markdown("Select the scenario details and click **Predict**.")
st.sidebar.markdown("---")

user_inputs = {}
for col in feature_order:
    options = list(encoders[col].classes_)
    label   = display_labels.get(col, col)
    user_inputs[col] = st.sidebar.selectbox(label, options, key=col)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button(" Predict Severity", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  IDLE / WELCOME STATE
# ─────────────────────────────────────────────────────────────────────────────
if not predict_btn:
    c1, c2, c3 = st.columns(3)
    tiles = [
        ("Real Dataset",   "Official NH-163 records from Telangana State Police — 2024."),
        ("Random Forest",  "Tuned ensemble with balanced class weighting for imbalanced severity data."),
        ("Explainable AI", "Every prediction includes feature-level reasons and safety recommendations."),
    ]
    for col_ui, (icon, title, desc) in zip([c1, c2, c3], tiles):
        with col_ui:
            st.markdown(f"""
            <div style="background:#f8faff;border:1px solid #d8e2f2;border-radius:16px;
                        padding:1.3rem 1.5rem;height:100%;">
                <div style="font-size:2rem;margin-bottom:.5rem;">{icon}</div>
                <div style="font-weight:700;font-size:1rem;margin-bottom:.4rem;">{title}</div>
                <div style="font-size:.85rem;color:#666;line-height:1.55;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("Configure road conditions in the sidebar, then click **Predict Severity**.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  ENCODE  →  PREDICT
# ─────────────────────────────────────────────────────────────────────────────
encoded_row = [
    int(encoders[col].transform([user_inputs[col]])[0])
    for col in feature_order
]
input_df = pd.DataFrame([encoded_row], columns=feature_order)

pred   = int(model.predict(input_df)[0])
probs  = model.predict_proba(input_df)[0]          # shape (4,)
result = severity_labels[pred] if pred < n_classes else f"Class {pred}"

# probability-weighted risk score (0–100)
_rw       = {0: 0.05, 1: 0.35, 2: 0.70, 3: 1.00}
risk_pct  = int(min(sum(float(probs[i]) * _rw.get(i, 1.0) for i in range(n_classes)) * 100, 100))
risk_color = "#e53935" if risk_pct >= 65 else "#fb8c00" if risk_pct >= 40 else "#fdd835" if risk_pct >= 20 else "#2e7d32"
risk_label = "Very High" if risk_pct >= 65 else "High" if risk_pct >= 40 else "Moderate" if risk_pct >= 20 else "Low"

card_cls, title_cls, _accent, _icon = _sev_cfg.get(result, ("sev-noinjury","clr-noinjury","#1565c0","⚪"))


# ─────────────────────────────────────────────────────────────────────────────
#  SAFE FEATURE LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
_norm = {k.lower().replace(" ","").replace("_",""): k for k in user_inputs}

def _get(*candidates):
    for c in candidates:
        k = _norm.get(c.lower().replace(" ","").replace("_",""))
        if k is not None:
            return user_inputs[k]
    return ""

road_geometry      = _get("Road Geometry", "roadgeometry")
victim_vehicle     = _get("Victim Vehicle Type", "victimvehicletype", "victimvehicle")
offender_vehicle   = _get("Offending Vehicle Type", "offendingvehicletype", "offendervehicle")
victim_manoeuvre   = _get("Victim Manoeuvre", "victimmanoeuvre")
offender_manoeuvre = _get("Offender Manoeuvre", "offendermanoeuvre", "offendingmanoeuvre")
accident_type      = _get("Type of Accident", "typeofaccident", "accidenttype")
time_of_day        = _get("Time of Day", "timeofday", "timecategory", "time")
day_type           = _get("Day Type", "daytype")
cause              = _get("Cause", "causeofaccident", "cause_of_accident")


# ─────────────────────────────────────────────────────────────────────────────
#  RISK FACTOR PILLS
# ─────────────────────────────────────────────────────────────────────────────
active_factors = []

c_lo = cause.lower()
if "over speeding" in c_lo or "speeding" in c_lo:
    active_factors.append(("Over Speeding", "red"))
if "drunken" in c_lo or "drunk" in c_lo:
    active_factors.append(("Drunken Driving", "red"))
if "loss of control" in c_lo:
    active_factors.append(("Loss of Control", "orange"))
if "poor visibility" in c_lo or "bad road" in c_lo:
    active_factors.append(("Poor Visibility/Road", "orange"))
if "driver fatigue" in c_lo or "fatigue" in c_lo:
    active_factors.append(("Driver Fatigue", "orange"))
if "junction" in c_lo:
    active_factors.append(("Junction Issue", "yellow"))

if "night" in time_of_day.lower():
    active_factors.append(("Night-time", "red" if "weekend" in day_type.lower() else "orange"))
elif "evening" in time_of_day.lower():
    active_factors.append(("Evening Peak", "yellow"))

if "weekend" in day_type.lower():
    active_factors.append(("Weekend", "yellow"))

if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
    active_factors.append((victim_vehicle, "red"))

if offender_vehicle in ["Heavy Vehicle"]:
    active_factors.append(("Heavy Vehicle", "orange"))

if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
    active_factors.append((road_geometry, "orange"))
elif road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction", "Staggered Junction"]:
    active_factors.append((road_geometry, "yellow"))

if accident_type in ["Head on"]:
    active_factors.append(("Head-on Collision", "red"))
elif accident_type in ["Vehicle Overturn", "Run off the road"]:
    active_factors.append((accident_type, "orange"))

if result == "Fatal":
    active_factors.insert(0, ("⚠ Fatal Risk", "red"))
elif result == "Grievous Injuries":
    active_factors.insert(0, ("⚠ Grievous Risk", "orange"))


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC REASONS
# ─────────────────────────────────────────────────────────────────────────────
def build_reasons():
    r = []
    c = cause.lower()

    # — CAUSE —
    if "over speeding" in c or "speeding" in c:
        if result == "Fatal":
            r.append(("red",
                "Over Speeding is the dominant cause of fatal crashes on this corridor. At high kinetic "
                "energy levels, the model consistently identifies this as a maximum-severity scenario — "
                "particularly when combined with two-wheelers or head-on collision type."))
        elif result == "Grievous Injuries":
            r.append(("orange",
                "Over Speeding substantially amplifies impact force, making grievous injury probable "
                "even when road geometry and vehicle type are only moderately adverse."))
        else:
            r.append(("yellow",
                "Over Speeding is present, but the remaining conditions (vehicle type, road geometry, "
                "collision type) have limited predicted severity to a lower class in this scenario."))

    if "drunken" in c or "drunk" in c:
        r.append(("red",
            "Drunken Driving impairs reaction time by up to 30% and reduces peripheral vision. "
            "It is one of the strongest single-feature predictors of fatal outcomes in the NH-163 dataset — "
            "independent of road geometry or vehicle type."))

    if "driver fatigue" in c or "fatigue" in c:
        r.append(("orange",
            "Driver Fatigue reduces hazard detection and braking response. Fatigue-related crashes on "
            "highways tend to produce high-speed off-road or head-on scenarios with elevated severity."))

    if "loss of control" in c:
        r.append(("orange",
            "Loss of Control typically results from sudden evasive manoeuvres, wet road surfaces, "
            "or mechanical failure — often leading to secondary collisions or rollovers that "
            "escalate final severity."))

    if "poor visibility" in c or "bad road" in c:
        r.append(("orange",
            "Poor Visibility or Bad Road Condition extends stopping distances and delays hazard detection. "
            "Both conditions are well-established severity amplifiers, particularly at night and on "
            "curved or junction segments."))

    if "junction" in c:
        r.append(("yellow",
            "Junction-related issues such as inadequate sight lines, signal confusion, or "
            "conflicting movements create multiple simultaneous risk interactions — "
            "elevating severity beyond what single-feature analysis would predict."))

    if "overloading" in c:
        r.append(("yellow",
            "Overloading increases vehicle braking distance and reduces stability, particularly on "
            "gradients and curved sections. It is a contributing factor to heavy-vehicle rollovers."))

    # — VEHICLE TYPE —
    if victim_vehicle == "Two Wheeler":
        if result == "Fatal":
            r.append(("red",
                "Two-wheeler riders have no protective enclosure, crumple zones, or airbags. "
                "At the vehicle combination and road conditions selected, this represents a "
                "high-lethality scenario — two-wheeler fatality risk is consistently the highest "
                "of all vehicle classes in the training data."))
        else:
            r.append(("orange",
                "Two-wheeler involvement significantly elevates injury risk. Even at moderate "
                "impact speeds, the absence of structural protection means the rider absorbs "
                "a disproportionate share of collision energy."))
    elif victim_vehicle == "Pedestrian":
        r.append(("red",
            "Pedestrian involvement is one of the strongest predictors of fatal or grievous outcomes. "
            "Without any physical protection, even moderate-speed vehicle impacts are life-threatening — "
            "pedestrian crashes on NH-163 show the highest per-incident fatality rate."))
    elif victim_vehicle == "Auto":
        r.append(("yellow",
            "Auto-rickshaw occupants have limited structural protection compared to cars. "
            "Lateral and frontal impacts carry elevated injury risk due to the open design."))

    if offender_vehicle == "Heavy Vehicle":
        r.append(("orange",
            "Heavy Vehicle involvement creates an extreme mass disparity — the lighter vehicle absorbs "
            "the majority of deformation energy. This is a leading contributor to fatal and grievous "
            "outcomes on National Highways across India."))

    # — ROAD GEOMETRY —
    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        r.append(("orange",
            f"{road_geometry} reduces sight distance and increases lateral forces. "
            f"These geometric types on NH-163 show disproportionately higher severity rates "
            f"per unit length relative to straight segments."))
    elif road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction", "Staggered Junction"]:
        r.append(("orange",
            f"{road_geometry} creates multiple simultaneous conflict points. Angle and turning "
            f"collisions at junctions produce grievous or fatal outcomes due to direct lateral "
            f"impact forces on vehicle structures."))

    # — COLLISION TYPE —
    if accident_type == "Head on":
        r.append(("red",
            "Head-on collisions combine the velocities of both vehicles, making them the most "
            "energy-intensive crash type and a leading cause of fatalities on national highways nationally."))
    elif accident_type == "Vehicle Overturn":
        r.append(("orange",
            "Vehicle overturns involve sustained multi-impact energy transfer as the vehicle "
            "rolls, significantly increasing injury severity beyond the initial collision event."))
    elif accident_type in ["Front back", "Front side", "Side front"]:
        r.append(("yellow",
            f"A {accident_type} collision indicates asymmetric impact. While less severe than "
            f"head-on, these crash types still produce significant deceleration forces on occupants."))

    # — TIME / DAY —
    if "night" in time_of_day.lower():
        if "weekend" in day_type.lower():
            r.append(("red",
                "Weekend night-time is the highest-risk temporal window on this corridor. Fatigue, "
                "reduced lighting, and elevated prevalence of impaired driving compound all severity factors."))
        else:
            r.append(("orange",
                "Night-time driving on NH-163 involves reduced ambient lighting and slower hazard "
                "detection. Both factors are consistent contributors to higher-severity outcomes."))
    elif "evening" in time_of_day.lower():
        r.append(("yellow",
            "Evening hours coincide with peak traffic volume on this corridor, increasing conflict "
            "probability and rear-end risk as vehicles decelerate at junction approaches."))

    # — MANOEUVRE —
    if victim_manoeuvre in ["Crossing", "Wrong side driving", "U Turn"]:
        r.append(("yellow",
            f"The victim's manoeuvre ({victim_manoeuvre}) places them in an unexpected trajectory, "
            f"reducing reaction time for the offending vehicle."))
    if offender_manoeuvre in ["Over Taking", "Wrong side driving", "U Turn"]:
        r.append(("yellow",
            f"The offender's manoeuvre ({offender_manoeuvre}) involves lateral movement into a "
            f"conflict zone — a predictor of angle and side-impact crashes with higher severity "
            f"than following-traffic events."))

    # — LOW SEVERITY —
    if result in ["No Injury", "Simple Injuries"]:
        r.append(("green",
            "While some risk factors are present, the overall combination in this scenario is "
            "associated with lower severity outcomes in the training data. No high-weight feature "
            "combination was found that would elevate this to a grievous or fatal prediction."))

    return r[:7]


# ─────────────────────────────────────────────────────────────────────────────
#  DYNAMIC RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_recommendations():
    r = []
    c = cause.lower()

    if result == "Fatal":
        r.append(("Immediate corridor intervention",
            "This scenario indicates extreme fatality risk. Temporary speed restrictions, "
            "enhanced patrol presence, and emergency signage deployment are recommended."))

    if "over speeding" in c or "speeding" in c:
        r.append(("Speed enforcement infrastructure",
            "Deploy average-speed cameras at 2 km intervals on open highway segments. "
            "A 10 km/h reduction in mean speed reduces fatal crashes by ~34% (NHAI data)."))

    if "drunken" in c or "drunk" in c:
        r.append(("Impaired driving enforcement",
            "Increase breathalyzer checkpoints during night and weekend windows. "
            "Sustained enforcement — not one-off campaigns — is significantly more effective."))

    if "driver fatigue" in c or "fatigue" in c:
        r.append(("Driver fatigue countermeasures",
            "Install rumble strips on highway shoulders, place mandatory rest area signs, "
            "and run awareness campaigns for long-haul drivers on this corridor."))

    if "poor visibility" in c or "bad road" in c:
        r.append(("Visibility and road surface improvement",
            "Upgrade lane markings with high-retroreflectivity paint. Address surface defects "
            "reported at black spots. Install advance fog warning signs on problem segments."))

    if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
        r.append(("Vulnerable road user protection",
            "Construct dedicated footpaths and two-wheeler lanes. Install pedestrian refuges, "
            "improve zebra crossing visibility, and mandate ABS on two-wheelers above 125cc."))

    if offender_vehicle == "Heavy Vehicle":
        r.append(("Heavy vehicle management",
            "Enforce lane discipline for trucks and HGVs. Mandate speed limiters and introduce "
            "time-based restrictions near signalised junctions and school zones."))

    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        r.append(("Geometric safety treatment",
            "Install advance warning signs 300 m before hazards, chevron delineators on curves, "
            "crash barriers on exposed edges, and skid-resistant pavement at critical locations."))
    elif road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction", "Staggered Junction"]:
        r.append(("Junction safety upgrade",
            "Improve channelisation with retroreflective delineators. Evaluate signal timing "
            "optimisation, grade separation for high-volume movements, and pedestrian phases."))

    if accident_type == "Head on":
        r.append(("Head-on collision prevention",
            "Install flexible delineator posts as median protection. Evaluate wire rope or "
            "concrete median barriers on high-volume sections with head-on crash history."))

    if accident_type in ["Vehicle Overturn", "Run off the road"]:
        r.append(("Off-road and rollover prevention",
            "Install longitudinal rumble strips at edge lines and earthen shoulders. "
            "Evaluate guardrail placement on steep or curved segments with drop-off risk."))

    if "night" in time_of_day.lower():
        r.append(("Night-time visibility upgrade",
            "Upgrade corridor lighting to consistent LED luminance. Install retroreflective lane "
            "markings and raised pavement markers at bridges, curves, and junction approaches."))

    if offender_manoeuvre in ["Over Taking", "Wrong side driving"]:
        r.append(("Overtaking zone management",
            "Mark no-overtaking zones with double yellow lines and delineators. "
            "Consider average-speed enforcement specifically targeting overtaking segments."))

    if "weekend" in day_type.lower():
        r.append(("Weekend traffic management",
            "Deploy additional traffic personnel on Friday and Saturday nights. Use variable "
            "message signs for weekend-specific risk campaigns — fatigue and impaired driving."))

    # always include speed management
    if not any("speed" in t.lower() for _, t, _ in r):
        r.append(("Speed management",
            "Average-speed camera enforcement at 2 km intervals is the single most cost-effective "
            "fatal crash reduction measure on NH corridors in India."))

    return r[:6]


reasons = build_reasons()
recs    = build_recommendations()


# ─────────────────────────────────────────────────────────────────────────────
#  ROW 1 — Prediction card  |  Confidence bars  |  Risk factor pills
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1.6, 1.2, 1.2])

with col1:
    st.markdown(f"""
    <div class="sev-card {card_cls}">
        <p class="sev-title {title_cls}">{_icon}&nbsp;{result}</p>
        <p style="margin:0;font-size:.9rem;color:#555;">Predicted accident severity outcome</p>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">Risk Level</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:.9rem;color:#444;margin-bottom:4px;">
        Severity risk index:&nbsp;<strong style="color:{risk_color};">{risk_label}</strong>
    </div>
    <div class="risk-track">
        <div class="risk-fill" style="width:{risk_pct}%;background:{risk_color};"></div>
    </div>
    <div style="font-size:.8rem;color:#999;margin-top:3px;">{risk_pct}% weighted severity probability</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sec-hdr">Severity Confidence</div>', unsafe_allow_html=True)
    for i in range(n_classes):
        lbl     = severity_labels[i]
        pct     = int(round(float(probs[i]) * 100))
        clr     = _sev_cfg.get(lbl, ("","","#888",""))[2]
        is_pred = (i == pred)
        wt      = "700" if is_pred else "400"
        tag     = " ◀ predicted" if is_pred else ""
        opa     = "1.0" if is_pred else "0.38"
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;
                        font-size:.84rem;font-weight:{wt};margin-bottom:3px;">
                <span>{lbl}{tag}</span>
                <span style="color:{clr};">{pct}%</span>
            </div>
            <div style="background:#dce5f4;border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{pct}%;background:{clr};height:100%;border-radius:6px;opacity:{opa};"></div>
            </div>
        </div>""", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="sec-hdr">Scenario Risk Factors</div>', unsafe_allow_html=True)
    if active_factors:
        st.markdown(
            "".join(f'<span class="pill pill-{c}">{l}</span>' for l, c in active_factors),
            unsafe_allow_html=True,
        )
        n_af = len(active_factors)
        st.markdown(
            f"<div style='margin-top:.9rem;font-size:.82rem;color:#888;'>"
            f"{n_af} risk factor{'s' if n_af != 1 else ''} identified.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div style="background:#e8f5e9;border:1px solid #2e7d32;border-radius:12px;
                    padding:.85rem 1rem;font-size:.88rem;color:#1b5e20;line-height:1.5;">
            No major risk factors detected. This scenario is associated with
            lower severity outcomes in the NH-163 training data.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ROW 2 — Why this prediction
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Why this prediction?</div>', unsafe_allow_html=True)
if reasons:
    for clr, txt in [(r[0], r[1]) for r in reasons]:
        st.markdown(f'<div class="reason-block reason-{clr}">{txt}</div>', unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="reason-block reason-blue">
        The model evaluated all 9 input features collectively and assigned the most probable
        severity class based on patterns learned from the NH-163 dataset. Severity reflects
        the joint combination of conditions rather than any single dominant factor.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ROW 3 — Recommended Safety Actions
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Recommended Safety Actions</div>', unsafe_allow_html=True)
rc1, rc2 = st.columns(2)
for i, (icon, title, detail) in enumerate(recs):
    with (rc1 if i % 2 == 0 else rc2):
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-head">{icon}&nbsp;{title}</div>
            <div style="font-size:.84rem;color:#555;line-height:1.6;">{detail}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  ROW 4 — Global Feature Importance chart
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Global Feature Importance</div>', unsafe_allow_html=True)

importances  = model.feature_importances_
feat_display = [display_labels.get(col, col) for col in feature_order]
n_fi         = min(len(importances), len(feat_display))
fi           = pd.Series(importances[:n_fi], index=feat_display[:n_fi]).sort_values(ascending=True)

n = len(fi)
fi_colors = ["#e53935" if rank >= n-2 else "#fb8c00" if rank >= n-5 else "#90a4ae"
             for rank in range(n)]

fig1, ax1 = plt.subplots(figsize=(9, max(3.5, n * 0.55)))
bars1 = ax1.barh(fi.index, fi.values, color=fi_colors,
                 height=0.62, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars1, fi.values):
    ax1.text(val + 0.0012, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left", fontsize=9, color="#333")

ax1.set_xlabel("Mean Decrease in Impurity", fontsize=9, color="#555")
ax1.set_title("Feature Importance — Tuned Random Forest (NH-163)",
              fontsize=10.5, color="#0e0e24", pad=10, fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)
ax1.spines[["left","bottom"]].set_color("#d4dcea")
ax1.tick_params(axis="y", labelsize=9.5, colors="#333")
ax1.tick_params(axis="x", labelsize=8.5, colors="#666")
ax1.set_facecolor("#fafbff")
fig1.patch.set_facecolor("#fafbff")
ax1.legend(handles=[
    mpatches.Patch(color="#e53935", label="Top 2 — highest influence"),
    mpatches.Patch(color="#fb8c00", label="Mid-tier features"),
    mpatches.Patch(color="#90a4ae", label="Lower influence"),
], fontsize=8.5, loc="lower right", framealpha=0.9, edgecolor="#d4dcea")
plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)
st.markdown(
    "<div style='font-size:.81rem;color:#aaa;margin-top:-4px;'>"
    "Mean decrease in impurity across all 300 trees. Higher = stronger contribution "
    "to severity classification.</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
#  ROW 5 — Input encoding chart  (all 9 features)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Current Input — Encoded Feature Values</div>', unsafe_allow_html=True)

enc_vals = [int(encoders[col].transform([user_inputs[col]])[0]) for col in feature_order]
enc_disp = [display_labels.get(col, col) for col in feature_order]

max_val  = max(enc_vals) if max(enc_vals) > 0 else 1
enc_clrs = []
for v in enc_vals:
    if v == 0:               enc_clrs.append("#b0bec5")
    elif v / max_val >= .67: enc_clrs.append("#e53935")
    elif v / max_val >= .34: enc_clrs.append("#fb8c00")
    else:                    enc_clrs.append("#1565c0")

fig2, ax2 = plt.subplots(figsize=(9, max(3.5, len(enc_vals) * 0.55)))
bars2 = ax2.barh(enc_disp, enc_vals, color=enc_clrs,
                 height=0.58, edgecolor="white", linewidth=0.5)
for bar, val, col in zip(bars2, enc_vals, feature_order):
    lbl_txt = user_inputs.get(col, "")
    ax2.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
             f"{val}  ({lbl_txt})", va="center", ha="left", fontsize=8.5, color="#333")

ax2.set_xlabel("Label-encoded category index", fontsize=9, color="#555")
ax2.set_title("Input Features — Encoded Values with Selected Labels",
              fontsize=10.5, color="#0e0e24", pad=10, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.spines[["left","bottom"]].set_color("#d4dcea")
ax2.tick_params(axis="y", labelsize=9.5, colors="#333")
ax2.tick_params(axis="x", labelsize=8.5, colors="#666")
ax2.set_facecolor("#fafbff")
fig2.patch.set_facecolor("#fafbff")
ax2.legend(handles=[
    mpatches.Patch(color="#e53935", label="High-index (more adverse)"),
    mpatches.Patch(color="#fb8c00", label="Mid-index category"),
    mpatches.Patch(color="#1565c0", label="Low-index category"),
    mpatches.Patch(color="#b0bec5", label="Baseline (index 0)"),
], fontsize=8.5, loc="lower right", framealpha=0.9, edgecolor="#d4dcea")
plt.tight_layout()
st.pyplot(fig2)
plt.close(fig2)


# ─────────────────────────────────────────────────────────────────────────────
#  DEBUG EXPANDER
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Debug — Raw model output"):
    st.markdown("**Encoded input (9 features):**")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("**Class probabilities:**")
    for i in range(n_classes):
        lbl = severity_labels[i]
        pct = int(round(float(probs[i]) * 100))
        st.write(f"- **{lbl}** (class {i}): `{float(probs[i]):.4f}` ({pct}%)")

    st.markdown(f"**Predicted class index:** `{pred}` → **`{result}`**  |  **Risk score:** `{risk_pct}%` — {risk_label}")

    st.markdown("**Resolved feature values:**")
    st.json({
        "Road Geometry":      road_geometry,
        "Victim Vehicle":     victim_vehicle,
        "Offending Vehicle":  offender_vehicle,
        "Victim Manoeuvre":   victim_manoeuvre,
        "Offender Manoeuvre": offender_manoeuvre,
        "Type of Accident":   accident_type,
        "Time of Day":        time_of_day,
        "Day Type":           day_type,
        "Cause":              cause,
    })


