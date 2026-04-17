import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Accident Severity AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: #f8f9fb;
        border-right: 1px solid #e0e4ea;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    .severity-card {
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        border: 1.5px solid;
    }
    .card-fatal    { background:#fff1f1; border-color:#e53935; }
    .card-grievous { background:#fff8e1; border-color:#fb8c00; }
    .card-minor    { background:#e8f5e9; border-color:#43a047; }
    .card-property { background:#e3f2fd; border-color:#1e88e5; }

    .severity-title { font-size:2rem; font-weight:700; margin:0 0 .25rem; }
    .severity-sub   { font-size:.95rem; color:#555; margin:0; }

    .fatal-title    { color:#b71c1c; }
    .grievous-title { color:#e65100; }
    .minor-title    { color:#2e7d32; }
    .property-title { color:#1565c0; }

    .risk-bar-bg {
        background:#e0e4ea; border-radius:8px;
        height:18px; width:100%; margin:8px 0 4px;
        overflow:hidden;
    }
    .risk-bar-fill {
        height:100%; border-radius:8px;
        transition: width 0.5s ease;
    }

    .factor-pill {
        display:inline-block;
        padding:3px 12px;
        border-radius:20px;
        font-size:.82rem;
        font-weight:600;
        margin:3px 4px 3px 0;
    }
    .pill-red    { background:#fdecea; color:#b71c1c; border:1px solid #e53935; }
    .pill-orange { background:#fff3e0; color:#e65100; border:1px solid #fb8c00; }
    .pill-yellow { background:#fffde7; color:#f57f17; border:1px solid #f9a825; }
    .pill-green  { background:#e8f5e9; color:#2e7d32; border:1px solid #43a047; }

    .reason-block {
        background:#fafbfc;
        border-left:4px solid #e53935;
        border-radius:0 8px 8px 0;
        padding:.7rem 1rem;
        margin-bottom:.6rem;
        font-size:.92rem;
    }
    .reason-block.orange { border-color:#fb8c00; }
    .reason-block.yellow { border-color:#fdd835; }
    .reason-block.green  { border-color:#43a047; background:#f1f8f1; }

    .rec-card {
        background:#fff;
        border:1px solid #dde2ea;
        border-radius:10px;
        padding:.8rem 1rem;
        margin-bottom:.6rem;
        font-size:.9rem;
    }
    .rec-icon { font-size:1.2rem; margin-right:8px; }

    .section-header {
        font-size:1.1rem;
        font-weight:700;
        color:#1a1a2e;
        border-bottom:2px solid #eef0f5;
        padding-bottom:.4rem;
        margin:1.4rem 0 .9rem;
    }

    .stExpander { border:1px solid #e0e4ea !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL + ENCODERS + METADATA
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

feature_order = metadata["feature_order"]
severity_labels = metadata["severity_labels"]
display_labels = metadata["display_labels"]

severity_style = {
    "Fatal":           ("card-fatal",    "fatal-title",    "#e53935"),
    "Grievous Injuries": ("card-grievous", "grievous-title", "#fb8c00"),
    "Simple Injuries": ("card-minor",    "minor-title",    "#43a047"),
    "No Injury":       ("card-property", "property-title", "#1e88e5"),
}

severity_icon = {
    "Fatal": "🔴",
    "Grievous Injuries": "🟠",
    "Simple Injuries": "🟢",
    "No Injury": "🔵",
}

# ─────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────
st.sidebar.markdown("## Road Conditions")
st.sidebar.markdown("Fill in the scenario details below and click **Predict**.")
st.sidebar.markdown("---")

user_inputs = {}
for col in feature_order:
    options = list(encoders[col].classes_)
    label = display_labels.get(col, col)
    user_inputs[col] = st.sidebar.selectbox(label, options)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🚀 Predict Severity", use_container_width=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🚧 Accident Severity Predictor")
st.markdown("Smart Road Risk & Severity Analysis System — powered by Tuned Random Forest")
st.markdown("---")

if not predict_btn:
    st.info("👈 Configure road conditions in the sidebar and click **Predict Severity** to begin.")
    st.stop()

# ─────────────────────────────────────────────
# ENCODE INPUT
# ─────────────────────────────────────────────
encoded_row = []
for col in feature_order:
    encoded_val = encoders[col].transform([user_inputs[col]])[0]
    encoded_row.append(encoded_val)

input_df = pd.DataFrame([encoded_row], columns=feature_order)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
pred = int(model.predict(input_df)[0])
probs = model.predict_proba(input_df)[0]

result = severity_labels[pred]

# ─────────────────────────────────────────────
# RISK SCORE (Probability-based)
# ─────────────────────────────────────────────
risk_weights = {
    0: 0.10,   # No Injury
    1: 0.40,   # Simple Injuries
    2: 0.75,   # Grievous Injuries
    3: 1.00    # Fatal
}

risk_percent = int(sum(probs[i] * risk_weights[i] for i in range(len(probs))) * 100)

risk_color = (
    "#e53935" if risk_percent >= 70 else
    "#fb8c00" if risk_percent >= 45 else
    "#fdd835" if risk_percent >= 25 else
    "#43a047"
)
risk_label = (
    "Very High" if risk_percent >= 70 else
    "High"      if risk_percent >= 45 else
    "Moderate"  if risk_percent >= 25 else
    "Low"
)

# ─────────────────────────────────────────────
# DYNAMIC REASONS
# ─────────────────────────────────────────────
def build_reasons(inp, result):
    reasons = []

    cause = inp.get("Cause", "")
    time_of_day = inp.get("Time of Day", "")
    day_type = inp.get("Day Type", "")
    road_geometry = inp.get("Road Geometry", "")
    victim_vehicle = inp.get("Victim Vehicle Type", "")
    offender_vehicle = inp.get("Offending Vehicle Type", "")
    victim_manoeuvre = inp.get("Victim Manoeuvre", "")
    offender_manoeuvre = inp.get("Offender Manoeuvre", "")
    accident_type = inp.get("Type of Accident", "")

    if "Over Speeding" in cause:
        reasons.append(("red",
            "Overspeeding is a dominant crash driver in this corridor and significantly increases impact severity."))

    if "Drunken" in cause:
        reasons.append(("red",
            "Drunken driving severely reduces reaction time and decision-making, increasing the probability of severe outcomes."))

    if "Loss of Control" in cause:
        reasons.append(("orange",
            "Loss of control indicates unstable driving conditions and often leads to high-energy crashes."))

    if "Bad Road" in cause or "Visibility" in cause:
        reasons.append(("orange",
            "Environmental and roadway conditions appear to have contributed to this scenario."))

    if time_of_day == "Night":
        reasons.append(("orange",
            "Night-time driving typically increases risk because of fatigue, poor visibility, and delayed hazard detection."))

    if day_type == "Weekend" and time_of_day == "Night":
        reasons.append(("red",
            "Weekend night conditions are associated with higher risk due to nightlife traffic and impaired driving probability."))

    if road_geometry in ["Curved Road", "Bridge", "Steep Grade", "T - Junction", "Y - Junction", "Four Arm Junction"]:
        reasons.append(("orange",
            "This road geometry is operationally more complex than a straight segment and can elevate crash severity."))

    if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
        reasons.append(("red",
            "Unprotected road users such as two-wheelers and pedestrians are more vulnerable to severe injuries."))

    if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry", "Bus - RTC", "Bus", "Bus - Private"]:
        reasons.append(("orange",
            "Heavy vehicle involvement can increase crash force and injury severity due to mass disparity."))

    if accident_type in ["Head on", "Front back", "Front side"]:
        reasons.append(("red",
            "This accident type is generally associated with stronger impact forces and higher injury risk."))

    if victim_manoeuvre in ["Crossing", "Wrong side driving", "U Turn"]:
        reasons.append(("yellow",
            "The victim manoeuvre indicates conflict-prone movement, which may increase severity."))

    if offender_manoeuvre in ["Over Taking", "Wrong side driving", "U Turn"]:
        reasons.append(("yellow",
            "The offending manoeuvre suggests risky lateral or directional movement, contributing to crash escalation."))

    if result in ["No Injury", "Simple Injuries"] and cause == "Over Speeding":
        reasons.append(("green",
            "Although overspeeding is a major risk, the current combination of other factors may have limited the predicted severity."))

    return reasons[:7]

reasons = build_reasons(user_inputs, result)

# ─────────────────────────────────────────────
# DYNAMIC RECOMMENDATIONS
# ─────────────────────────────────────────────
def build_recommendations(inp, result):
    recs = []

    cause = inp.get("Cause", "")
    time_of_day = inp.get("Time of Day", "")
    day_type = inp.get("Day Type", "")
    road_geometry = inp.get("Road Geometry", "")
    victim_vehicle = inp.get("Victim Vehicle Type", "")
    offender_vehicle = inp.get("Offending Vehicle Type", "")
    accident_type = inp.get("Type of Accident", "")

    if result == "Fatal":
        recs.append(("🚨", "Immediate high-risk intervention",
            "This scenario indicates extreme crash severity risk. Temporary traffic control and speed enforcement should be prioritized."))

    if "Over Speeding" in cause:
        recs.append(("📷", "Speed enforcement",
            "Install speed cameras, rumble strips, and warning signage to reduce high-speed crashes on this stretch."))

    if "Drunken" in cause:
        recs.append(("🍺", "Impaired driving control",
            "Increase breathalyzer checks and police patrols, especially during weekend evenings and nights."))

    if road_geometry in ["Curved Road", "Bridge", "Steep Grade"]:
        recs.append(("⚠️", "Geometry-based safety treatment",
            "Provide chevron signs, reflective markers, crash barriers, and curve-speed warnings in high-risk geometric sections."))

    if road_geometry in ["T - Junction", "Y - Junction", "Four Arm Junction", "Staggered Junction", "Round about"]:
        recs.append(("🚦", "Junction safety improvement",
            "Improve junction channelization, signage, and visibility. Consider signal optimization or conflict reduction measures."))

    if victim_vehicle in ["Two Wheeler", "Pedestrian"]:
        recs.append(("🏍", "Vulnerable road user protection",
            "Strengthen pedestrian and two-wheeler safety through crossings, separators, reflectors, and visibility improvements."))

    if offender_vehicle in ["Heavy Vehicle", "Truck/Lorry", "Bus - RTC", "Bus", "Bus - Private"]:
        recs.append(("🛻", "Heavy vehicle management",
            "Introduce lane discipline enforcement and targeted controls for buses and trucks in mixed-traffic environments."))

    if time_of_day == "Night":
        recs.append(("💡", "Night safety measures",
            "Improve lighting, reflective markings, and late-night enforcement coverage for better hazard visibility."))

    if day_type == "Weekend":
        recs.append(("📅", "Weekend traffic control",
            "Weekend-specific patrols and traffic calming should be considered due to behavioral variation in traffic flow."))

    if accident_type in ["Head on", "Front back", "Front side"]:
        recs.append(("🛡", "Impact severity mitigation",
            "Introduce speed moderation, median protection, and lane discipline strategies to reduce severe collision outcomes."))

    return recs[:6]

recs = build_recommendations(user_inputs, result)

# ─────────────────────────────────────────────
# LAYOUT — ROW 1
# ─────────────────────────────────────────────
card_cls, title_cls, bar_color = severity_style[result]

col1, col2, col3 = st.columns([1.6, 1.2, 1.2])

with col1:
    st.markdown(f"""
    <div class="severity-card {card_cls}">
        <p class="severity-title {title_cls}">
            {severity_icon[result]} {result}
        </p>
        <p class="severity-sub">Predicted accident severity outcome</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Risk Level</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:.3rem; font-size:.9rem; color:#555;">
        Overall risk: <strong style="color:{risk_color}">{risk_label}</strong>
    </div>
    <div class="risk-bar-bg">
        <div class="risk-bar-fill" style="width:{risk_percent}%;background:{risk_color};"></div>
    </div>
    <div style="font-size:.8rem;color:#888;margin-top:2px;">{risk_percent}% estimated severity risk</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header">Severity Confidence</div>', unsafe_allow_html=True)
    for i in range(len(probs)):
        label = severity_labels[i]
        pct = int(round(probs[i] * 100))
        color = severity_style[label][2]
        is_pred = (i == pred)
        weight = "700" if is_pred else "400"
        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;
                        font-size:.85rem;font-weight:{weight};margin-bottom:3px;">
                <span>{label}</span><span style="color:{color}">{pct}%</span>
            </div>
            <div style="background:#eef0f5;border-radius:6px;height:8px;overflow:hidden;">
                <div style="width:{pct}%;background:{color};height:100%;
                            border-radius:6px;opacity:{'1' if is_pred else '0.5'};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-header">Scenario Summary</div>', unsafe_allow_html=True)

    active_factors = []

    if "Over Speeding" in user_inputs.get("Cause", ""):
        active_factors.append(("Over Speeding", "red"))
    if "Drunken" in user_inputs.get("Cause", ""):
        active_factors.append(("Drunken Driving", "red"))
    if user_inputs.get("Time of Day", "") == "Night":
        active_factors.append(("Night-time", "orange"))
    if user_inputs.get("Day Type", "") == "Weekend":
        active_factors.append(("Weekend", "yellow"))
    if user_inputs.get("Victim Vehicle Type", "") in ["Two Wheeler", "Pedestrian"]:
        active_factors.append((user_inputs.get("Victim Vehicle Type", ""), "orange"))
    if user_inputs.get("Road Geometry", "") in ["Curved Road", "Bridge", "Steep Grade"]:
        active_factors.append((user_inputs.get("Road Geometry", ""), "yellow"))

    if active_factors:
        pills_html = ""
        for label, color in active_factors:
            pills_html += f'<span class="factor-pill pill-{color}">{label}</span>'
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.markdown("No major risk factors detected.")

# ─────────────────────────────────────────────
# WHY THIS PREDICTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Why this prediction?</div>', unsafe_allow_html=True)

if reasons:
    for text, color in [(r[1], r[0]) for r in reasons]:
        st.markdown(f'<div class="reason-block {color}">{text}</div>', unsafe_allow_html=True)
else:
    st.info("No significant risk factors identified for this scenario.")

# ─────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Recommended Actions</div>', unsafe_allow_html=True)

rec_cols = st.columns(2)
for i, (icon, title, detail) in enumerate(recs):
    with rec_cols[i % 2]:
        st.markdown(f"""
        <div class="rec-card">
            <div style="font-weight:700;font-size:.95rem;margin-bottom:4px;">
                <span class="rec-icon">{icon}</span>{title}
            </div>
            <div style="font-size:.85rem;color:#555;line-height:1.5;">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)

importances = model.feature_importances_
feat_imp = pd.Series(importances, index=feature_order).sort_values(ascending=True)

fig1, ax1 = plt.subplots(figsize=(8, 4))
feat_imp.plot(kind="barh", ax=ax1, color="#1e88e5")
ax1.set_title("Feature Importance (Random Forest)")
ax1.set_xlabel("Importance Score")
ax1.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig1)

# ─────────────────────────────────────────────
# INPUT ENCODING VIEW
# ─────────────────────────────────────────────
st.markdown('<div class="section-header">Feature Input Encoding</div>', unsafe_allow_html=True)

values = input_df.iloc[0].values
colors = ["#fb8c00" if v > 0 else "#b0bec5" for v in values]

fig2, ax2 = plt.subplots(figsize=(8, 4))
bars = ax2.barh(feature_order, values, color=colors, height=0.55, edgecolor="white")
ax2.set_xlabel("Encoded Input Value", fontsize=9, color="#666")
ax2.tick_params(axis='y', labelsize=9)
ax2.tick_params(axis='x', labelsize=8)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color('#dde2ea')
ax2.set_facecolor("#fafbfc")
fig2.patch.set_facecolor("#fafbfc")

for bar, val in zip(bars, values):
    ax2.text(val + 0.03, bar.get_y() + bar.get_height()/2,
             str(int(val)), va='center', fontsize=8, color="#444")

legend_patches = [
    mpatches.Patch(color="#fb8c00", label="Selected encoded category"),
    mpatches.Patch(color="#b0bec5", label="Baseline / low code"),
]
ax2.legend(handles=legend_patches, fontsize=8, loc="lower right",
           framealpha=0.7, edgecolor="#dde2ea")

plt.tight_layout()
st.pyplot(fig2)

# ─────────────────────────────────────────────
# DEBUG EXPANDER
# ─────────────────────────────────────────────
with st.expander("🔬 Debug — Raw model output"):
    st.markdown("**Encoded input row:**")
    st.write(input_df)

    st.markdown("**Raw probabilities:**")
    for i in range(len(probs)):
        st.write(f"{severity_labels[i]}: `{probs[i]:.4f}` ({int(round(probs[i]*100))}%)")

    st.markdown(f"**Final prediction:** `{result}`")
