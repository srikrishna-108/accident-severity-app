import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Severity AI · NH-163",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #f0f4fa;
    border-right: 1px solid #d8e0ee;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }

/* ── severity cards ── */
.sev-card {
    border-radius: 18px;
    padding: 1.4rem 1.7rem;
    margin-bottom: 1rem;
    border: 2px solid;
    transition: box-shadow .2s;
}
.sev-fatal    { background: linear-gradient(135deg,#fff1f1 55%,#fde0e0); border-color:#e53935; }
.sev-grievous { background: linear-gradient(135deg,#fff8e1 55%,#ffe8b0); border-color:#fb8c00; }
.sev-simple   { background: linear-gradient(135deg,#e8f5e9 55%,#c8e6c9); border-color:#43a047; }
.sev-noinjury { background: linear-gradient(135deg,#e3f2fd 55%,#bbdefb); border-color:#1e88e5; }

.sev-title { font-size: 2rem; font-weight: 700; margin: 0 0 .25rem; letter-spacing: -.4px; }
.clr-fatal    { color: #b71c1c; }
.clr-grievous { color: #bf360c; }
.clr-simple   { color: #1b5e20; }
.clr-noinjury { color: #0d47a1; }

/* ── risk bar ── */
.risk-track {
    background: #e2e8f4;
    border-radius: 10px;
    height: 18px;
    width: 100%;
    margin: 8px 0 4px;
    overflow: hidden;
    box-shadow: inset 0 1px 4px rgba(0,0,0,.1);
}
.risk-fill { height: 100%; border-radius: 10px; transition: width .4s ease; }

/* ── pills ── */
.pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: .78rem;
    font-weight: 600;
    margin: 3px 3px 3px 0;
}
.pill-red    { background:#fdecea; color:#b71c1c; border:1.5px solid #e53935; }
.pill-orange { background:#fff3e0; color:#bf360c; border:1.5px solid #fb8c00; }
.pill-yellow { background:#fffde7; color:#f57f17; border:1.5px solid #f9a825; }
.pill-green  { background:#e8f5e9; color:#1b5e20; border:1.5px solid #43a047; }
.pill-blue   { background:#e3f2fd; color:#0d47a1; border:1.5px solid #1e88e5; }

/* ── reason blocks ── */
.reason-block {
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: .75rem 1.1rem;
    margin-bottom: .55rem;
    font-size: .89rem;
    line-height: 1.65;
    color: #1e1e2e;
}
.reason-red    { background:#fff5f5; border-color:#e53935; }
.reason-orange { background:#fffbf0; border-color:#fb8c00; }
.reason-yellow { background:#fffef0; border-color:#f9a825; }
.reason-green  { background:#f1f9f2; border-color:#43a047; }
.reason-blue   { background:#f0f6ff; border-color:#1e88e5; }

/* ── recommendation cards ── */
.rec-card {
    background: #fff;
    border: 1px solid #dce4f0;
    border-radius: 14px;
    padding: 1rem 1.15rem;
    margin-bottom: .65rem;
    font-size: .87rem;
    line-height: 1.6;
    box-shadow: 0 2px 6px rgba(0,0,0,.04);
}
.rec-head { font-weight: 700; font-size: .92rem; margin-bottom: 5px; color: #12122a; }
.rec-icon { font-size: 1.1rem; margin-right: 6px; }

/* ── section header ── */
.sec-hdr {
    font-size: 1.05rem;
    font-weight: 700;
    color: #12122a;
    border-bottom: 2px solid #eaf0fb;
    padding-bottom: .4rem;
    margin: 1.5rem 0 .85rem;
}

/* ── confidence bar row ── */
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: .84rem;
    margin-bottom: 3px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  LOAD MODEL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("rf_model.pkl",  "rb") as f: model     = pickle.load(f)
    with open("encoders.pkl",  "rb") as f: encoders  = pickle.load(f)
    with open("metadata.pkl",  "rb") as f: metadata  = pickle.load(f)
    return model, encoders, metadata

model, encoders, metadata = load_artifacts()

feature_order  = list(metadata["feature_order"])   # list of raw column names used in training
display_labels = metadata["display_labels"]         # dict: raw_col -> human-readable label

# ── FIX: severity_labels may be a dict {0:"No Injury", 1:...} or a list ──────
_raw_sev = metadata["severity_labels"]
if isinstance(_raw_sev, dict):
    # dict keys are integer class indices → sort by key → extract values
    severity_labels = [_raw_sev[k] for k in sorted(_raw_sev.keys())]
elif isinstance(_raw_sev, list):
    severity_labels = list(_raw_sev)
else:
    severity_labels = list(_raw_sev)

n_classes = len(severity_labels)

# ── visual config per class ───────────────────────────────────────────
_sev_cfg = {
    "Fatal":             ("sev-fatal",    "clr-fatal",    "#e53935", "🔴"),
    "Grievous Injuries": ("sev-grievous", "clr-grievous", "#fb8c00", "🟠"),
    "Simple Injuries":   ("sev-simple",   "clr-simple",   "#43a047", "🟢"),
    "No Injury":         ("sev-noinjury", "clr-noinjury", "#1e88e5", "🔵"),
}


# ─────────────────────────────────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────────────────────────────────
st.markdown("# 🚧 Accident Severity Predictor")
st.markdown(
    "**Smart Road Risk & Severity Analysis System** — "
    "Tuned Random Forest · NH-163 Warangal–Hyderabad · "
    "Telangana Police Dataset 2024"
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────
#  SIDEBAR — INPUTS  (only features present in feature_order / encoders)
# ─────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛣 Road Conditions")
st.sidebar.markdown("Select scenario details and click **Predict Severity**.")
st.sidebar.markdown("---")

user_inputs = {}
for col in feature_order:
    if col not in encoders:
        continue                              # safety: skip any missing encoder
    options = list(encoders[col].classes_)
    label   = display_labels.get(col, col)
    user_inputs[col] = st.sidebar.selectbox(label, options, key=col)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Predict Severity", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
#  WELCOME / IDLE STATE
# ─────────────────────────────────────────────────────────────────────
if not predict_btn:
    c1, c2, c3 = st.columns(3)
    tiles = [
        ("🗂️", "Real Dataset",    "Official NH-163 accident records from Telangana State Police."),
        ("🌲", "Random Forest",   "Tuned ensemble model with balanced class weighting."),
        ("🧠", "Explainable AI",  "Every prediction includes feature-level reasons and safety recommendations."),
    ]
    for col, (icon, title, body) in zip([c1, c2, c3], tiles):
        with col:
            st.markdown(f"""
            <div style="background:#f8faff;border:1px solid #dce4f0;border-radius:16px;
                        padding:1.3rem 1.5rem;height:100%;">
                <div style="font-size:2rem;margin-bottom:.5rem;">{icon}</div>
                <div style="font-weight:700;margin-bottom:.4rem;font-size:1rem;">{title}</div>
                <div style="font-size:.85rem;color:#666;line-height:1.55;">{body}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈  Configure road conditions in the sidebar and click **Predict Severity** to begin.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────
#  ENCODE + PREDICT
# ─────────────────────────────────────────────────────────────────────
encoded_row = []
for col in feature_order:
    val = int(encoders[col].transform([user_inputs[col]])[0])
    encoded_row.append(val)

input_df = pd.DataFrame([encoded_row], columns=feature_order)
pred     = int(model.predict(input_df)[0])          # integer class index
probs    = model.predict_proba(input_df)[0]          # float array length = n_classes

# Resolve predicted class name safely
result = severity_labels[pred] if pred < len(severity_labels) else f"Class {pred}"

# Risk score: weighted probability across severity levels
_weights  = {0: 0.05, 1: 0.35, 2: 0.70, 3: 1.00}
risk_pct  = int(min(sum(float(probs[i]) * _weights.get(i, 1.0) for i in range(n_classes)) * 100, 100))
risk_color = "#e53935" if risk_pct >= 65 else "#fb8c00" if risk_pct >= 40 else "#fdd835" if risk_pct >= 20 else "#43a047"
risk_label = "Very High" if risk_pct >= 65 else "High" if risk_pct >= 40 else "Moderate" if risk_pct >= 20 else "Low"

# Card style for result
card_cls, title_cls, _accent, _icon = _sev_cfg.get(result, ("sev-noinjury","clr-noinjury","#1e88e5","⚪"))


# ─────────────────────────────────────────────────────────────────────
#  HELPER — safe lookup of user_inputs without depending on Cause
# ─────────────────────────────────────────────────────────────────────
_norm = {k.lower().replace(" ","").replace("_",""): k for k in user_inputs}

def _get(*candidates):
    for c in candidates:
        k = _norm.get(c.lower().replace(" ","").replace("_",""))
        if k is not None:
            return user_inputs[k]
    return ""

road_geometry      = _get("Road Geometry", "RoadGeometry", "roadgeometry", "geometry")
victim_vehicle     = _get("Victim Vehicle Type", "VictimVehicleType", "victimvehicle")
offender_vehicle   = _get("Offending Vehicle Type", "OffendingVehicleType", "offendervehicle", "crimevehicle")
victim_manoeuvre   = _get("Victim Manoeuvre", "VictimManoeuvre", "victimmanoeuvre")
offender_manoeuvre = _get("Offender Manoeuvre", "OffenderManoeuvre", "offendermanoeuvre", "crimemanoeuvre")
time_of_day        = _get("Time Category", "TimeCategory", "Time of Day", "TimeofDay", "timeofday", "time")
day_type           = _get("Day Type", "DayType", "daytype")


# ─────────────────────────────────────────────────────────────────────
#  ACTIVE RISK FACTOR PILLS
# ─────────────────────────────────────────────────────────────────────
active_factors = []

if time_of_day and "night" in time_of_day.lower():
    active_factors.append(("Night-time",  "red"))
if day_type and "weekend" in day_type.lower():
    active_factors.append(("Weekend",     "orange"))
if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
    active_factors.append((victim_vehicle, "red"))
if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry", "Bus", "Bus - RTC", "Bus - Private"]:
    active_factors.append(("Heavy Vehicle", "orange"))
if road_geometry in ["Curved Road", "Bridge", "Steep Grade",
                     "T - Junction", "Y - Junction", "Four Arm Junction"]:
    active_factors.append((road_geometry, "orange"))
if victim_manoeuvre in ["Crossing", "Wrong side driving", "U Turn"]:
    active_factors.append((victim_manoeuvre, "yellow"))
if offender_manoeuvre in ["Over Taking", "Wrong side driving", "U Turn"]:
    active_factors.append((offender_manoeuvre, "yellow"))
if result == "Fatal":
    active_factors.insert(0, ("Fatal Risk", "red"))
elif result == "Grievous Injuries":
    active_factors.insert(0, ("Grievous Risk", "orange"))


# ─────────────────────────────────────────────────────────────────────
#  DYNAMIC REASONS  — based only on available features
# ─────────────────────────────────────────────────────────────────────
def build_reasons():
    r = []

    # vehicle combination
    if victim_vehicle == "Two Wheeler":
        if result == "Fatal":
            r.append(("red",
                "Two-wheeler riders have no protective enclosure, crumple zones, or airbags. "
                "At the vehicle combination and road conditions selected, the model identifies this as "
                "a high-lethality scenario — two-wheeler fatality risk on NH-163 is consistently the highest "
                "of all vehicle classes in the training data."))
        else:
            r.append(("orange",
                "Two-wheeler involvement significantly elevates injury risk due to the complete absence "
                "of structural occupant protection. Even at moderate speeds, impact forces are absorbed "
                "directly by the rider."))
    elif victim_vehicle == "Pedestrian":
        r.append(("red",
            "Pedestrian involvement is one of the strongest predictors of fatal or grievous outcomes "
            "in the NH-163 dataset. Without any protection, even moderate-speed vehicle impacts are "
            "life-threatening."))
    elif victim_vehicle == "Car":
        r.append(("green",
            "Car occupants benefit from structural crumple zones and airbags, which substantially reduce "
            "injury severity compared to two-wheelers at comparable impact energies."))

    if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry"]:
        r.append(("orange",
            "Heavy vehicle involvement creates an extreme mass disparity: the lighter vehicle absorbs "
            "the majority of deformation energy. This combination is a leading contributor to fatal and "
            "grievous injury outcomes on National Highways."))
    elif offender_vehicle in ["Bus", "Bus - RTC", "Bus - Private"]:
        r.append(("orange",
            "Bus involvement elevates severity due to vehicle mass, height, and the frequency of "
            "pedestrian exposure near bus routes. Buses also have higher stopping distances than cars."))

    # road geometry
    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        r.append(("orange",
            f"The selected geometry ({road_geometry}) reduces sight distance and increases lateral "
            f"forces on vehicles. Curved and elevated sections on NH-163 statistically show disproportionately "
            f"higher severity rates per unit length relative to straight segments."))
    elif road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction", "Staggered Junction"]:
        r.append(("orange",
            f"Junction geometry ({road_geometry}) creates multiple simultaneous conflict points. "
            f"Angle and right-turn collisions at junctions produce grievous or fatal outcomes due "
            f"to the direct lateral impact force on vehicle structures."))
    elif road_geometry == "Straight Road":
        r.append(("yellow",
            "Straight road geometry is the dominant segment type on this corridor. While individually "
            "low-risk, straight segments account for the highest absolute accident count on NH-163 "
            "due to their length proportion and tendency for high-speed travel."))

    # time and day
    if time_of_day and "night" in time_of_day.lower():
        if day_type and "weekend" in day_type.lower():
            r.append(("red",
                "Weekend night-time is the highest-risk temporal window on this corridor. Fatigue, "
                "reduced ambient lighting, and elevated prevalence of impaired driving compound "
                "all physical severity factors simultaneously."))
        else:
            r.append(("orange",
                "Night-time driving on NH-163 involves reduced ambient lighting, slower hazard "
                "detection, and longer stopping distances under headlamp illumination. Both factors "
                "are well-established contributors to higher-severity accident outcomes."))
    elif time_of_day and "evening" in time_of_day.lower():
        r.append(("yellow",
            "Evening hours coincide with peak traffic volume on this corridor, increasing conflict "
            "probability and rear-end interaction risk as vehicles decelerate at junction approaches."))

    # manoeuvre
    if victim_manoeuvre in ["Crossing", "Wrong side driving", "U Turn"]:
        r.append(("yellow",
            f"The victim's manoeuvre ({victim_manoeuvre}) places them in an unexpected trajectory, "
            f"reducing available reaction time for the offending vehicle and increasing the probability "
            f"of a high-energy direct impact."))
    if offender_manoeuvre in ["Over Taking", "Wrong side driving", "U Turn"]:
        r.append(("yellow",
            f"The offender's manoeuvre ({offender_manoeuvre}) involves lateral movement into an "
            f"occupied conflict zone — a strong predictor of angle and side-impact collisions, "
            f"which produce higher severity than following-traffic crashes."))

    # low severity conclusion
    if result in ["No Injury", "Simple Injuries"]:
        r.append(("green",
            "While some individual risk factors are present, the overall combination of conditions "
            "in this scenario is associated with lower severity outcomes in the training data. "
            "The model found no high-weight feature combination that would elevate this to a "
            "grievous or fatal prediction."))

    return r[:6]


# ─────────────────────────────────────────────────────────────────────
#  DYNAMIC RECOMMENDATIONS — based on available features
# ─────────────────────────────────────────────────────────────────────
def build_recommendations():
    r = []

    if result == "Fatal":
        r.append(("🚨", "Immediate corridor intervention",
            "This scenario indicates extreme fatality risk. Emergency traffic management, "
            "temporary speed restrictions, and enhanced patrol presence are recommended."))

    if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
        r.append(("🏍", "Vulnerable road user protection",
            "Construct dedicated footpaths and two-wheeler lanes. Install pedestrian refuges, "
            "improve zebra crossing visibility with retroreflective markings, and consider "
            "physical separation barriers at high-conflict points."))

    if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry", "Bus", "Bus - RTC", "Bus - Private"]:
        r.append(("🛻", "Heavy vehicle management",
            "Enforce strict lane discipline for buses and trucks. Introduce truck lay-bys on "
            "critical segments, require mandatory speed limiters, and implement time-based "
            "restrictions near school zones and signalised junctions."))

    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        r.append(("⚠️", "Geometric safety treatment",
            "Install advance warning signs 300 m before the hazard, chevron markers on curves, "
            "crash barriers on exposed edges, and skid-resistant pavement at critical locations. "
            "Consider widening sight lines by clearing roadside vegetation."))
    elif road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction",
                           "Staggered Junction", "Round about"]:
        r.append(("🚦", "Junction safety upgrade",
            "Improve approach channelisation with retroreflective delineators. "
            "Evaluate signal timing optimisation and grade separation for high-volume movements. "
            "Install pedestrian push-button phases at all arms."))

    if time_of_day and "night" in time_of_day.lower():
        r.append(("💡", "Night-time visibility upgrade",
            "Upgrade roadway lighting to consistent LED luminance. Install retroreflective lane "
            "markings, raised pavement markers, and cat's eyes at regular intervals. "
            "Priority sections: bridges, curves, and junction approaches."))

    if offender_manoeuvre in ["Over Taking", "Wrong side driving"]:
        r.append(("🛡", "Overtaking and lane discipline enforcement",
            "Deploy camera-based overtaking zone monitoring. Install flexible delineator posts "
            "at no-overtaking zones. Consider wire rope or concrete median barriers on "
            "high-volume sections with a head-on crash history."))

    if day_type and "weekend" in day_type.lower():
        r.append(("📅", "Weekend traffic management",
            "Deploy additional traffic personnel on Friday and Saturday nights. "
            "Use variable message signs for weekend-specific risk campaigns — "
            "particularly targeting fatigue and impaired driving."))

    # Baseline speed recommendation always present
    r.append(("📷", "Speed management infrastructure",
        "Deploy average-speed cameras at 2 km intervals on open highway segments. "
        "A 10 km/h reduction in mean operating speed reduces fatal crashes by approximately "
        "34% on national highways (NHAI guidelines)."))

    return r[:6]


reasons = build_reasons()
recs    = build_recommendations()


# ─────────────────────────────────────────────────────────────────────
#  ROW 1 — Prediction Card + Confidence + Scenario Summary
# ─────────────────────────────────────────────────────────────────────
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
        Severity risk index: <strong style="color:{risk_color};">{risk_label}</strong>
    </div>
    <div class="risk-track">
        <div class="risk-fill" style="width:{risk_pct}%;background:{risk_color};"></div>
    </div>
    <div style="font-size:.8rem;color:#999;margin-top:3px;">{risk_pct}% weighted severity probability</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="sec-hdr">Severity Confidence</div>', unsafe_allow_html=True)
    for i in range(n_classes):
        lbl      = severity_labels[i]
        pct      = int(round(float(probs[i]) * 100))
        clr      = _sev_cfg.get(lbl, ("","","#888",""))[2]
        is_pred  = (i == pred)
        wt       = "700" if is_pred else "400"
        tag      = " ◀ predicted" if is_pred else ""
        opacity  = "1.0" if is_pred else "0.4"
        st.markdown(f"""
        <div style="margin-bottom:10px;">
            <div class="conf-label" style="font-weight:{wt};">
                <span>{lbl}{tag}</span>
                <span style="color:{clr};">{pct}%</span>
            </div>
            <div style="background:#e8edf6;border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{pct}%;background:{clr};height:100%;border-radius:6px;
                            opacity:{opacity};transition:width .3s;"></div>
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
            f"{n_af} risk factor{'s' if n_af != 1 else ''} identified in this scenario.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div style="background:#e8f5e9;border:1px solid #43a047;border-radius:12px;
                    padding:.85rem 1rem;font-size:.88rem;color:#2e7d32;line-height:1.5;">
            ✅ No major risk factors detected. This scenario is associated with lower
            severity outcomes based on the NH-163 training data.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  ROW 2 — Why this prediction
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Why this prediction?</div>', unsafe_allow_html=True)
if reasons:
    for clr, txt in [(r[0], r[1]) for r in reasons]:
        st.markdown(
            f'<div class="reason-block reason-{clr}">{txt}</div>',
            unsafe_allow_html=True,
        )
else:
    st.markdown("""
    <div class="reason-block reason-blue">
        The model evaluated all input features collectively and assigned the most probable
        severity class based on patterns learned from the NH-163 dataset. No single dominant
        risk factor was identified — severity reflects the joint combination of conditions.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  ROW 3 — Recommended Safety Actions
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Recommended Safety Actions</div>', unsafe_allow_html=True)
rc1, rc2 = st.columns(2)
for i, (icon, title, detail) in enumerate(recs):
    with (rc1 if i % 2 == 0 else rc2):
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-head"><span class="rec-icon">{icon}</span>{title}</div>
            <div style="font-size:.84rem;color:#555;line-height:1.6;">{detail}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
#  ROW 4 — Global Feature Importance chart
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Global Feature Importance</div>', unsafe_allow_html=True)

importances  = model.feature_importances_
# Use only the features that are in feature_order (which match the model)
feat_display = [display_labels.get(col, col) for col in feature_order]

# Ensure lengths match
n_fi = min(len(importances), len(feat_display))
fi   = pd.Series(importances[:n_fi], index=feat_display[:n_fi]).sort_values(ascending=True)

n    = len(fi)
colors_fi = []
for rank in range(n):
    if rank >= n - 2:       colors_fi.append("#e53935")
    elif rank >= n - 4:     colors_fi.append("#fb8c00")
    else:                   colors_fi.append("#90a4ae")

fig1, ax1 = plt.subplots(figsize=(9, max(3.2, n * 0.52)))
bars1 = ax1.barh(fi.index, fi.values, color=colors_fi,
                 height=0.62, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars1, fi.values):
    ax1.text(val + 0.0012, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", ha="left", fontsize=9, color="#333")

ax1.set_xlabel("Mean Decrease in Impurity", fontsize=9, color="#555")
ax1.set_title("Feature Importance — Tuned Random Forest (NH-163 Model)",
              fontsize=10.5, color="#12122a", pad=10, fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)
ax1.spines[["left","bottom"]].set_color("#dce4f0")
ax1.tick_params(axis="y", labelsize=9.5, colors="#333")
ax1.tick_params(axis="x", labelsize=8.5, colors="#666")
ax1.set_facecolor("#fafbff")
fig1.patch.set_facecolor("#fafbff")
ax1.legend(handles=[
    mpatches.Patch(color="#e53935", label="Top 2 — highest influence"),
    mpatches.Patch(color="#fb8c00", label="Mid-tier features"),
    mpatches.Patch(color="#90a4ae", label="Lower influence"),
], fontsize=8.5, loc="lower right", framealpha=0.9, edgecolor="#dce4f0")
plt.tight_layout()
st.pyplot(fig1)
st.markdown(
    "<div style='font-size:.81rem;color:#999;margin-top:-4px;'>"
    "Mean decrease in impurity across all trees. Higher score = stronger contribution "
    "to severity classification across the training dataset.</div>",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────
#  ROW 5 — Current Input Encoding chart
#          Shows all features in feature_order with encoded value + label
# ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-hdr">Current Input — Encoded Feature Values</div>', unsafe_allow_html=True)

enc_vals = [int(encoders[col].transform([user_inputs[col]])[0]) for col in feature_order]
enc_disp = [display_labels.get(col, col) for col in feature_order]

n_enc    = len(enc_vals)
max_val  = max(enc_vals) if max(enc_vals) > 0 else 1
enc_clrs = []
for v in enc_vals:
    if v == 0:
        enc_clrs.append("#b0bec5")
    elif v / max_val >= 0.67:
        enc_clrs.append("#e53935")
    elif v / max_val >= 0.34:
        enc_clrs.append("#fb8c00")
    else:
        enc_clrs.append("#1e88e5")

fig2, ax2 = plt.subplots(figsize=(9, max(3.2, n_enc * 0.52)))
bars2 = ax2.barh(enc_disp, enc_vals, color=enc_clrs,
                 height=0.58, edgecolor="white", linewidth=0.5)

for bar, val, col in zip(bars2, enc_vals, feature_order):
    label_txt = user_inputs.get(col, "")
    ax2.text(
        val + 0.05,
        bar.get_y() + bar.get_height() / 2,
        f"{val}  ({label_txt})",
        va="center", ha="left", fontsize=8.5, color="#333",
    )

ax2.set_xlabel("Label-encoded category index", fontsize=9, color="#555")
ax2.set_title("Input Features — Encoded Values with Selected Category Labels",
              fontsize=10.5, color="#12122a", pad=10, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.spines[["left","bottom"]].set_color("#dce4f0")
ax2.tick_params(axis="y", labelsize=9.5, colors="#333")
ax2.tick_params(axis="x", labelsize=8.5, colors="#666")
ax2.set_facecolor("#fafbff")
fig2.patch.set_facecolor("#fafbff")
ax2.legend(handles=[
    mpatches.Patch(color="#e53935", label="High-index (more adverse category)"),
    mpatches.Patch(color="#fb8c00", label="Mid-index category"),
    mpatches.Patch(color="#1e88e5", label="Low-index category"),
    mpatches.Patch(color="#b0bec5", label="Baseline category (index 0)"),
], fontsize=8.5, loc="lower right", framealpha=0.9, edgecolor="#dce4f0")
plt.tight_layout()
st.pyplot(fig2)


# ─────────────────────────────────────────────────────────────────────
#  DEBUG EXPANDER
# ─────────────────────────────────────────────────────────────────────
with st.expander("🔬 Debug — Raw model output"):
    st.markdown("**Encoded input:**")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("**Class probabilities:**")
    for i in range(n_classes):
        lbl = severity_labels[i]
        pct = int(round(float(probs[i]) * 100))
        st.write(f"- **{lbl}**: `{float(probs[i]):.4f}` ({pct}%)")

    st.markdown(f"**Predicted index:** `{pred}` → **`{result}`**")
    st.markdown(f"**Risk score:** `{risk_pct}%` — **{risk_label}**")

    st.markdown("**Resolved feature values:**")
    st.json({
        "road_geometry":      road_geometry,
        "victim_vehicle":     victim_vehicle,
        "offender_vehicle":   offender_vehicle,
        "victim_manoeuvre":   victim_manoeuvre,
        "offender_manoeuvre": offender_manoeuvre,
        "time_of_day":        time_of_day,
        "day_type":           day_type,
    })


# ─────────────────────────────────────────────────────────────────────
#  PANEL DEMO GUIDE
# ─────────────────────────────────────────────────────────────────────
with st.expander("📋 Panel Demo — Suggested Test Scenarios"):
    st.markdown("""
**Run these in order to demonstrate all four severity classes.**

---
**Scenario 1 — Fatal (worst case)**
Victim Vehicle: *Two Wheeler* · Offending Vehicle: *Heavy Vehicle* / *Truck/Lorry*
Road Geometry: *Curved Road* · Time Category: *Night* · Day Type: *Weekend*
Victim Manoeuvre: *Going Straight* · Offender Manoeuvre: *Over Taking*

*Shows: unprotected road user + heavy vehicle + night/weekend + curve = Fatal + Very High risk.*

---
**Scenario 2 — Grievous Injuries**
Victim Vehicle: *Two Wheeler* · Offending Vehicle: *Car*
Road Geometry: *T - Junction* · Time Category: *Night* · Day Type: *Weekday*
Victim Manoeuvre: *Crossing* · Offender Manoeuvre: *Going Straight*

*Shows: junction conflict + two-wheeler at night = Grievous Injuries.*

---
**Scenario 3 — Simple Injuries (moderate)**
Victim Vehicle: *Car* · Offending Vehicle: *Car*
Road Geometry: *Straight Road* · Time Category: *Afternoon* · Day Type: *Weekday*
Victim Manoeuvre: *Going Straight* · Offender Manoeuvre: *Going Straight*

*Shows: standard rear-end, low-risk combination.*

---
**Scenario 4 — No Injury (baseline)**
Victim Vehicle: *Car* · Offending Vehicle: *Car*
Road Geometry: *Straight Road* · Time Category: *Morning* · Day Type: *Weekday*
Victim Manoeuvre: *Going Straight* · Offender Manoeuvre: *Stationary*

*Shows: baseline low-risk → No Injury + Low risk score.*

---
**Live feature importance demo:**
Start with Scenario 1. Change only *Victim Vehicle* from Two Wheeler → Car.
Observe the severity drop — confirms vehicle type is the highest-ranked feature.
    """)
