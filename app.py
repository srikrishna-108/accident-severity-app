import streamlit as st
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8f9fb;
        border-right: 1px solid #e0e4ea;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* Cards */
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

    /* Risk bar */
    .risk-bar-bg {
        background:#e0e4ea; border-radius:8px;
        height:18px; width:100%; margin:8px 0 4px;
        overflow:hidden;
    }
    .risk-bar-fill {
        height:100%; border-radius:8px;
        transition: width 0.5s ease;
    }

    /* Factor pills */
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

    /* Reason cards */
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

    /* Recommendation cards */
    .rec-card {
        background:#fff;
        border:1px solid #dde2ea;
        border-radius:10px;
        padding:.8rem 1rem;
        margin-bottom:.6rem;
        font-size:.9rem;
    }
    .rec-icon { font-size:1.2rem; margin-right:8px; }

    /* Section headers */
    .section-header {
        font-size:1.1rem;
        font-weight:700;
        color:#1a1a2e;
        border-bottom:2px solid #eef0f5;
        padding-bottom:.4rem;
        margin:1.4rem 0 .9rem;
    }

    /* Metric tiles */
    .metric-tile {
        background:#f4f6fa;
        border-radius:10px;
        padding:.9rem 1rem;
        text-align:center;
    }
    .metric-tile .val { font-size:1.7rem; font-weight:700; color:#1a1a2e; }
    .metric-tile .lbl { font-size:.78rem; color:#888; margin-top:2px; }

    /* Debug expander clean */
    .stExpander { border:1px solid #e0e4ea !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return xgb.Booster(model_file="xgb_model.json")

model = load_model()


# ─────────────────────────────────────────────
#  CATEGORY MAPS
# ─────────────────────────────────────────────
weather_map   = {"Clear": 0, "Rainy": 1, "Foggy": 2}
vehicle_map   = {"Two-wheeler": 0, "Car": 1, "Truck": 2, "Bus": 3}
collision_map = {"Rear-end": 0, "Side": 1, "Head-on": 2, "Pedestrian": 3}
time_map      = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
lighting_map  = {"Good": 0, "Poor": 1}
speed_map     = {"Low": 0, "Medium": 1, "High": 2}
lanes_map     = {"2": 0, "4": 1, "6": 2}
traffic_map   = {"Low": 0, "Medium": 1, "High": 2}
geometry_map  = {"Straight": 0, "Moderate Curve": 1, "Sharp Curve": 2}
gradient_map  = {"Flat": 0, "Moderate": 1, "Steep": 2}

feature_names = [
    "Weather", "Vehicle_Type", "Collision_Type", "Time_of_Day",
    "Lighting", "Speed_Category", "Number_of_Lanes",
    "Traffic_Volume", "Road_Geometry", "Gradient"
]

severity_map = {
    0: "Property Damage",
    1: "Minor Injury",
    2: "Grievous Injury",
    3: "Fatal"
}

severity_style = {
    "Fatal":           ("card-fatal",    "fatal-title",    "#e53935"),
    "Grievous Injury": ("card-grievous", "grievous-title", "#fb8c00"),
    "Minor Injury":    ("card-minor",    "minor-title",    "#43a047"),
    "Property Damage": ("card-property", "property-title", "#1e88e5"),
}

severity_icon = {
    "Fatal":           "🔴",
    "Grievous Injury": "🟠",
    "Minor Injury":    "🟢",
    "Property Damage": "🔵",
}


# ─────────────────────────────────────────────
#  SIDEBAR — INPUT UI
# ─────────────────────────────────────────────
st.sidebar.markdown("## Road Conditions")
st.sidebar.markdown("Fill in the scenario details below and click **Predict**.")
st.sidebar.markdown("---")

weather   = st.sidebar.selectbox("🌦 Weather",          list(weather_map.keys()))
lighting  = st.sidebar.selectbox("💡 Lighting",         list(lighting_map.keys()))
time      = st.sidebar.selectbox("🕐 Time of Day",      list(time_map.keys()))
st.sidebar.markdown("---")
vehicle   = st.sidebar.selectbox("🚗 Vehicle Type",     list(vehicle_map.keys()))
speed     = st.sidebar.selectbox("⚡ Speed Category",   list(speed_map.keys()))
collision = st.sidebar.selectbox("💥 Collision Type",   list(collision_map.keys()))
st.sidebar.markdown("---")
geometry  = st.sidebar.selectbox("🛣 Road Geometry",    list(geometry_map.keys()))
gradient  = st.sidebar.selectbox("⛰ Road Gradient",    list(gradient_map.keys()))
lanes     = st.sidebar.selectbox("🔢 Number of Lanes",  list(lanes_map.keys()))
traffic   = st.sidebar.selectbox("🚦 Traffic Volume",   list(traffic_map.keys()))
st.sidebar.markdown("---")

predict_btn = st.sidebar.button("🚀 Predict Severity", use_container_width=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("# 🚧 Accident Severity Predictor")
st.markdown("Smart Road Risk & Severity Analysis System — powered by XGBoost")
st.markdown("---")

if not predict_btn:
    st.info("👈 Configure road conditions in the sidebar and click **Predict Severity** to begin.")
    st.stop()


# ─────────────────────────────────────────────
#  BUILD INPUT VECTOR
# ─────────────────────────────────────────────
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
probs   = model.predict(dmatrix)[0]
pred    = int(np.argmax(probs))


# ─────────────────────────────────────────────
#  SMART OVERRIDE — SCORE-BASED (FIXED)
# ─────────────────────────────────────────────
override_score = 0
if speed == "High":                              override_score += 3
if collision in ["Head-on", "Pedestrian"]:       override_score += 3
if lighting == "Poor":                           override_score += 2
if weather in ["Foggy", "Rainy"]:               override_score += 2
if geometry == "Sharp Curve":                    override_score += 2
if gradient == "Steep":                          override_score += 2
if vehicle == "Two-wheeler":                     override_score += 1
if time in ["Night", "Evening"]:                 override_score += 1
if traffic == "High":                            override_score += 1

# Hard fatal — absolute worst case
if override_score >= 12:
    pred = 3
# Soft fatal — high risk + model partially agrees
elif override_score >= 9 and probs[3] > 0.05:
    pred = 3

result = severity_map[pred]


# ─────────────────────────────────────────────
#  RISK SCORE (for display bar)
# ─────────────────────────────────────────────
risk_score = 0
if speed == "High":                              risk_score += 3
if collision in ["Head-on", "Pedestrian"]:       risk_score += 3
if weather != "Clear":                           risk_score += 2
if lighting == "Poor":                           risk_score += 2
if traffic == "High":                            risk_score += 2
if geometry == "Sharp Curve":                    risk_score += 2
if gradient == "Steep":                          risk_score += 2
if vehicle in ["Two-wheeler"]:                   risk_score += 2
if time in ["Night"]:                            risk_score += 1
if vehicle in ["Truck", "Bus"]:                  risk_score += 1

risk_percent = min(int((risk_score / 20) * 100), 100)

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
#  DYNAMIC REASONS (context-aware)
# ─────────────────────────────────────────────
def build_reasons(speed, collision, weather, lighting, geometry,
                  gradient, vehicle, time, traffic, result):
    reasons = []

    # Speed
    if speed == "High":
        if collision == "Head-on":
            reasons.append(("red",
                "High speed + head-on collision: kinetic energy is squared with velocity — "
                "impact force at high speed is 4× that of medium speed. Survival odds drop sharply."))
        elif collision == "Pedestrian":
            reasons.append(("red",
                "High-speed pedestrian strike: at speeds above 60 km/h, pedestrian fatality "
                "risk exceeds 85%. No protective structure absorbs the impact."))
        else:
            reasons.append(("orange",
                f"High speed with {collision.lower()} collision increases severity. "
                "Braking distance at high speed can exceed 3× that at medium speed."))
    elif speed == "Medium":
        reasons.append(("yellow",
            "Medium speed still carries moderate injury risk, especially on curves or wet roads."))

    # Collision type
    if collision == "Head-on":
        reasons.append(("red",
            "Head-on collisions are the deadliest collision type — combined closing speed doubles "
            "impact force. Even at medium speed, head-on crashes cause disproportionate fatalities."))
    elif collision == "Pedestrian":
        if vehicle in ["Truck", "Bus"]:
            reasons.append(("red",
                f"{vehicle} vs pedestrian: large vehicle height means the torso, not the legs, "
                "takes primary impact — dramatically raising fatal injury risk."))
        else:
            reasons.append(("orange",
                "Pedestrian collision: unprotected road users have no crumple zone or airbag. "
                "Severe injuries are common even at moderate speeds."))
    elif collision == "Rear-end" and traffic == "High":
        reasons.append(("yellow",
            "Rear-end collision in high traffic: chain-reaction risk is elevated. "
            "Multiple vehicle pile-ups can escalate a minor initial impact."))

    # Lighting
    if lighting == "Poor":
        if time == "Night":
            reasons.append(("red",
                "Poor lighting at night: driver reaction time increases 2–3× in darkness. "
                "Pedestrians and hazards are detected far later, leaving no time to brake."))
        else:
            reasons.append(("orange",
                "Poor lighting (dusk/artificial): reduced contrast and glare extend stopping "
                "distances and raise the chance of missed hazard detection."))

    # Weather
    if weather == "Foggy":
        reasons.append(("red",
            "Fog reduces visibility to under 50m in dense conditions. Drivers often "
            "maintain unsafe speeds, leading to multi-vehicle chain crashes."))
    elif weather == "Rainy":
        if geometry == "Sharp Curve":
            reasons.append(("red",
                "Rain + sharp curve: wet road friction drops by ~30%. Vehicles take curves "
                "at speeds safe in dry conditions but lose grip on wet asphalt."))
        else:
            reasons.append(("orange",
                "Rainy conditions: wet roads increase stopping distance by 50–70%. "
                "Hydroplaning risk rises with vehicle speed."))

    # Road geometry
    if geometry == "Sharp Curve":
        if gradient == "Steep":
            reasons.append(("red",
                "Sharp curve + steep gradient: compounded risk. Centrifugal force on the curve "
                "combined with gravity on the slope dramatically raises rollover and run-off risk."))
        else:
            reasons.append(("orange",
                "Sharp curve: lateral forces on vehicles exceed safe limits if speed isn't reduced. "
                "Trucks and buses are at rollover risk; two-wheelers at skid risk."))
    elif geometry == "Moderate Curve" and speed == "High":
        reasons.append(("yellow",
            "Moderate curve at high speed: can become dangerous, especially in adverse weather "
            "or with loaded trucks."))

    # Gradient
    if gradient == "Steep" and vehicle in ["Truck", "Bus"]:
        reasons.append(("orange",
            f"Steep gradient with {vehicle}: heavy vehicles experience brake fade on long descents. "
            "Brake failure on steep slopes is a leading cause of fatal truck accidents."))
    elif gradient == "Steep":
        reasons.append(("yellow",
            "Steep gradient: increases braking distance going downhill and reduces "
            "vehicle control at higher speeds."))

    # Vehicle
    if vehicle == "Two-wheeler":
        reasons.append(("orange",
            "Two-wheeler: motorcyclists have no protective shell, airbag, or crumple zone. "
            "Injury severity is consistently higher than for car occupants at the same impact speed."))
    if vehicle in ["Truck", "Bus"] and collision == "Head-on":
        reasons.append(("red",
            f"{vehicle} in head-on collision: mass disparity means the lighter vehicle "
            "absorbs most of the deformation energy. Occupant survival is severely compromised."))

    # Time
    if time == "Night" and lighting == "Good":
        reasons.append(("yellow",
            "Night-time driving with adequate lighting still raises fatigue-related risk. "
            "Driver alertness drops significantly after midnight."))

    # Traffic
    if traffic == "High" and collision == "Side":
        reasons.append(("yellow",
            "Side collision in high-traffic: secondary impacts from surrounding vehicles "
            "are likely. Injury can compound from multiple collision events."))

    # Positive factor — something working in their favour
    if result != "Fatal" and speed == "Low":
        reasons.append(("green",
            "Low speed: significantly limits injury severity. At speeds under 30 km/h, "
            "most collisions are survivable with minor injuries."))
    if result != "Fatal" and weather == "Clear" and lighting == "Good":
        reasons.append(("green",
            "Clear weather and good lighting: optimal visibility gives drivers maximum "
            "reaction time to detect and respond to hazards."))

    return reasons[:7]  # cap at 7 for readability


reasons = build_reasons(
    speed, collision, weather, lighting, geometry,
    gradient, vehicle, time, traffic, result
)


# ─────────────────────────────────────────────
#  DYNAMIC RECOMMENDATIONS
# ─────────────────────────────────────────────
def build_recommendations(speed, collision, weather, lighting,
                           geometry, gradient, vehicle, time, traffic, result):
    recs = []

    if result == "Fatal":
        recs.append(("🚨", "Immediate action required",
            "This combination of factors creates extreme fatality risk. "
            "Traffic authorities should consider temporary road closure or mandatory speed restrictions."))

    if speed == "High":
        recs.append(("📷", "Enforce speed limits electronically",
            "Deploy average-speed cameras across this road segment. "
            "Research shows a 10 km/h reduction in average speed cuts fatalities by ~34%."))

    if lighting == "Poor":
        if time == "Night":
            recs.append(("💡", "Install adaptive street lighting",
                "Sensor-activated LED lighting at this location would improve visibility 5–10× "
                "and has been shown to reduce night-time accidents by up to 30%."))
        else:
            recs.append(("💡", "Improve roadway lighting",
                "Upgrade to high-lumen LED fixtures at dusk-prone segments. "
                "Reflective road markings should also be renewed."))

    if geometry == "Sharp Curve":
        recs.append(("⚠️", "Install curve warning infrastructure",
            "Place dynamic speed advisory signs 200m before the curve. "
            "Add chevron markers, Armco barriers, and skid-resistant surface on the curve itself."))

    if weather in ["Rainy", "Foggy"]:
        recs.append(("🌧", "Deploy weather-responsive signage",
            "Connect variable message signs to real-time weather sensors. "
            "Automatically lower posted speed limits when rainfall or fog is detected."))

    if collision in ["Pedestrian"]:
        recs.append(("🚶", "Pedestrian safety infrastructure",
            "Install raised crossings, pedestrian refuge islands, and countdown timers. "
            "Consider pedestrian detection systems at this junction."))

    if collision == "Head-on":
        recs.append(("🛡", "Install central road dividers",
            "Physical separation (concrete barrier or wire rope median) eliminates "
            "head-on risk entirely. This single intervention reduces fatal head-on crashes by ~90%."))

    if gradient == "Steep" and vehicle in ["Truck", "Bus"]:
        recs.append(("🔧", "Heavy vehicle gradient controls",
            "Mandate pre-descent brake checks for heavy vehicles. "
            "Install escape ramps (runaway truck ramps) on long descents."))

    if vehicle == "Two-wheeler":
        recs.append(("🏍", "Motorcycle safety measures",
            "Ensure road surface has anti-skid treatment. "
            "Mandate ABS on all motorcycles above 125cc — ABS reduces fatal crashes by 31%."))

    if traffic == "High":
        recs.append(("🚦", "Intelligent traffic management",
            "Adaptive signal control at junctions reduces stop-and-go conflicts. "
            "Consider ramp metering or dynamic lane assignment during peak hours."))

    if time == "Night":
        recs.append(("🌙", "Night-time safety patrols",
            "Increase police visibility during 10 PM – 4 AM window. "
            "Alcohol checkpoints at this location reduce DUI incidents by up to 20%."))

    return recs[:6]


recs = build_recommendations(
    speed, collision, weather, lighting,
    geometry, gradient, vehicle, time, traffic, result
)


# ─────────────────────────────────────────────
#  LAYOUT — ROW 1: Severity + Risk + Metrics
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

    st.markdown(f'<div class="section-header">Risk Level</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:.3rem; font-size:.9rem; color:#555;">
        Overall risk: <strong style="color:{risk_color}">{risk_label}</strong>
    </div>
    <div class="risk-bar-bg">
        <div class="risk-bar-fill" style="width:{risk_percent}%;background:{risk_color};"></div>
    </div>
    <div style="font-size:.8rem;color:#888;margin-top:2px;">{risk_percent}% of maximum risk</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="section-header">Model Confidence</div>', unsafe_allow_html=True)
    for i, label in severity_map.items():
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
    st.markdown(f'<div class="section-header">Scenario Summary</div>', unsafe_allow_html=True)

    active_factors = []
    if speed == "High":                             active_factors.append(("High speed",    "red"))
    if collision in ["Head-on", "Pedestrian"]:      active_factors.append((collision,        "red"))
    if lighting == "Poor":                          active_factors.append(("Poor lighting", "orange"))
    if weather != "Clear":                          active_factors.append((weather,          "orange"))
    if geometry == "Sharp Curve":                   active_factors.append(("Sharp curve",   "orange"))
    if gradient == "Steep":                         active_factors.append(("Steep gradient","yellow"))
    if vehicle == "Two-wheeler":                    active_factors.append(("Two-wheeler",   "yellow"))
    if time == "Night":                             active_factors.append(("Night-time",    "yellow"))
    if speed == "Low" and weather == "Clear":       active_factors.append(("Low risk env.", "green"))

    if active_factors:
        pills_html = ""
        for label, color in active_factors:
            pills_html += f'<span class="factor-pill pill-{color}">{label}</span>'
        st.markdown(pills_html, unsafe_allow_html=True)
    else:
        st.markdown("No major risk factors detected.")

    st.markdown(f"""
    <div style="margin-top:1rem;font-size:.82rem;color:#888;">
        Override score: <strong>{override_score}</strong> / 17 &nbsp;|&nbsp;
        Model pred: <strong>{severity_map[int(np.argmax(probs))]}</strong>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  ROW 2: Why this prediction
# ─────────────────────────────────────────────
st.markdown(f'<div class="section-header">Why this prediction?</div>', unsafe_allow_html=True)

if reasons:
    for text, color in [(r[1], r[0]) for r in reasons]:
        st.markdown(f'<div class="reason-block {color}">{text}</div>', unsafe_allow_html=True)
else:
    st.info("No significant risk factors identified for this scenario.")


# ─────────────────────────────────────────────
#  ROW 3: Recommendations
# ─────────────────────────────────────────────
st.markdown(f'<div class="section-header">Recommended Actions</div>', unsafe_allow_html=True)

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
#  ROW 4: Feature Impact Chart
# ─────────────────────────────────────────────
st.markdown(f'<div class="section-header">Feature Input Encoding</div>', unsafe_allow_html=True)

values = input_data.flatten()
colors = []
for i, v in enumerate(values):
    fname = feature_names[i]
    if v == 0:
        colors.append("#b0bec5")
    elif fname in ["Speed_Category", "Collision_Type", "Lighting"] and v >= 1:
        colors.append("#e53935")
    elif v >= 1:
        colors.append("#fb8c00")
    else:
        colors.append("#b0bec5")

fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.barh(feature_names, values, color=colors, height=0.55, edgecolor="white")
ax.set_xlabel("Encoded Input Value", fontsize=9, color="#666")
ax.set_xlim(0, max(values) + 0.8)
ax.tick_params(axis='y', labelsize=9)
ax.tick_params(axis='x', labelsize=8)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color('#dde2ea')
ax.set_facecolor("#fafbfc")
fig.patch.set_facecolor("#fafbfc")

for bar, val in zip(bars, values):
    if val > 0:
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                str(int(val)), va='center', fontsize=8, color="#444")

legend_patches = [
    mpatches.Patch(color="#e53935", label="High risk input"),
    mpatches.Patch(color="#fb8c00", label="Moderate risk input"),
    mpatches.Patch(color="#b0bec5", label="Low / baseline input"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right",
          framealpha=0.7, edgecolor="#dde2ea")

plt.tight_layout()
st.pyplot(fig)


# ─────────────────────────────────────────────
#  DEBUG EXPANDER
# ─────────────────────────────────────────────
with st.expander("🔬 Debug — Raw model output"):
    st.markdown("**Raw probability outputs from XGBoost:**")
    for i, label in severity_map.items():
        st.write(f"  {label}: `{probs[i]:.4f}` ({int(round(probs[i]*100))}%)")
    st.markdown(f"**Override score:** `{override_score}` / 17")
    st.markdown(f"**Model's raw prediction:** `{severity_map[int(np.argmax(probs))]}`")
    st.markdown(f"**Final prediction (after override):** `{result}`")
    st.markdown(f"**Input vector:** `{input_data.flatten().tolist()}`")
