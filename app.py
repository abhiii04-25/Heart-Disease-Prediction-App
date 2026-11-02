from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction | ML Demo",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
      --bg: #08111f; --line: rgba(161,186,221,.16); --muted: #9babc4;
      --text: #f4f7ff; --mint: #71e5d1; --coral: #ff9e9e;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, [data-testid="stMetricValue"] { font-family: 'Space Grotesk','DM Sans',sans-serif !important; }

    [data-testid="stAppViewContainer"] {
      color: var(--text);
      background:
        radial-gradient(ellipse at 78% -8%, rgba(78,109,194,.20), transparent 36%),
        radial-gradient(ellipse at 0% 35%, rgba(62,210,190,.08), transparent 32%),
        var(--bg);
    }
    [data-testid="stHeader"] { background: rgba(8,17,31,.35); }
    main .block-container { max-width: 1320px; padding-top: 1.7rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg,#111d2f 0%,#0b1423 100%);
      border-right: 1px solid var(--line);
    }

    .hero {
      position: relative; overflow: hidden;
      padding: clamp(24px,4vw,43px); margin: 5px 0 27px;
      border: 1px solid rgba(139,184,232,.23); border-radius: 26px;
      background:
        radial-gradient(circle at 91% 12%, rgba(255,158,158,.13), transparent 28%),
        radial-gradient(circle at 7% 110%, rgba(113,229,209,.13), transparent 37%),
        linear-gradient(135deg, rgba(22,37,59,.97), rgba(10,20,35,.98));
      box-shadow: 0 24px 70px rgba(0,0,0,.19);
    }
    .hero:after {
      content: "♡"; position: absolute; right: 4%; bottom: -75px;
      color: rgba(255,170,179,.07); font-size: clamp(180px,24vw,310px);
      line-height: 1; pointer-events: none; transform: rotate(-12deg);
    }
    .eyebrow {
      position: relative; z-index: 1; display: flex; align-items: center; gap: 9px;
      color: #a9f4e9; font-size: .73rem; font-weight: 700;
      letter-spacing: .14em; text-transform: uppercase; margin-bottom: 16px;
    }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 13px rgba(113,229,209,.7); }
    .hero h1 {
      position: relative; z-index: 1; max-width: 820px; color: #f5f7ff;
      font-size: clamp(2.2rem,5vw,4.15rem); line-height: 1.05;
      letter-spacing: -.055em; margin: 0 0 13px; font-weight: 700;
    }
    .hero h1 span {
      background: linear-gradient(90deg, var(--mint), #9dbdff 68%, #d4acff);
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero-sub { position: relative; z-index: 1; max-width: 760px; color: #bac7dd; font-size: 1.04rem; line-height: 1.7; margin-bottom: 21px; }
    .pills { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 9px; }
    .pills span {
      padding: 7px 11px; border: 1px solid rgba(165,188,229,.19); border-radius: 999px;
      background: rgba(255,255,255,.045); color: #dce6f8; font-size: .78rem;
    }
    .section-kicker { color: var(--mint); text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 700; margin: 4px 0 5px; }
    .section-copy { color: var(--muted); margin: 0 0 18px; font-size: .94rem; }

    [data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--line) !important; border-radius: 19px !important;
      background: linear-gradient(145deg, rgba(19,31,50,.91), rgba(11,21,36,.95));
    }
    [data-testid="stMetric"] {
      padding: 15px 17px; border: 1px solid rgba(151,179,225,.16); border-radius: 15px;
      background: linear-gradient(145deg, rgba(25,42,65,.93), rgba(14,26,43,.94));
    }
    [data-testid="stMetricLabel"] { color: #acbad1 !important; font-size: .82rem; }
    [data-testid="stMetricValue"] { color: #f4f7ff !important; font-weight: 700; }

    .stButton > button, [data-testid="stFormSubmitButton"] button {
      border-radius: 12px; font-weight: 700; min-height: 45px;
      transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); filter: brightness(1.06); }
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
      color: #071620; background: linear-gradient(100deg,#71e5d1,#9cebdc); border: 0;
    }

    .result-card {
      padding: 24px; border-radius: 18px; border: 1px solid rgba(139,184,232,.20);
      background: linear-gradient(140deg, rgba(24,39,61,.96), rgba(12,22,38,.98)); margin: 6px 0 15px;
    }
    .result-card.alert { border-color: rgba(255,158,158,.43); }
    .result-card.calm  { border-color: rgba(113,229,209,.35); }
    .result-card .overline { color: var(--muted); font-size: .72rem; letter-spacing: .13em; text-transform: uppercase; font-weight: 700; margin-bottom: 10px; }
    .result-card h2 { color: #f5f7ff; font-size: clamp(1.4rem,3vw,2rem); margin: 0 0 10px; }
    .result-card p { color: #b8c5db; line-height: 1.65; margin: 0; }
    .notice {
      padding: 13px 15px; border-radius: 12px; border: 1px solid rgba(255,198,132,.25);
      background: rgba(255,185,90,.07); color: #f0d6b3; font-size: .84rem; line-height: 1.55;
    }
    .steps { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 10px; margin: 12px 0; }
    .step { padding: 15px 13px; border: 1px solid rgba(151,179,225,.16); border-radius: 14px; background: rgba(255,255,255,.025); }
    .step b { display: block; color: #e7eeff; font-size: .85rem; margin-bottom: 5px; }
    .step small { color: #92a2bd; line-height: 1.5; font-size: .76rem; }
    @media(max-width: 900px) { main .block-container { padding: 1rem 1rem 2rem; } .steps { grid-template-columns: 1fr; } }
    @media(max-width: 600px) { .hero { border-radius: 19px; padding: 23px 19px; } .hero h1 { font-size: 2.25rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Categorical encodings ----------
# Display label -> dataset code. Training data (UCI/Kaggle heart dataset) uses
# one-hot columns named "<Feature>_<Code>", e.g. "ChestPainType_ATA".
SEX = {"Male": "M", "Female": "F"}
CHEST_PAIN = {"Typical Angina": "TA", "Atypical Angina": "ATA", "Non-Anginal Pain": "NAP", "Asymptomatic": "ASY"}
RESTING_ECG = {"Normal": "Normal", "ST-T abnormality": "ST", "LVH": "LVH"}
EXERCISE_ANGINA = {"No": "N", "Yes": "Y"}
ST_SLOPE = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}

DEFAULTS = {
    "age": 40, "sex": "Male", "cp": "Atypical Angina", "bp": 120, "chol": 200, "fbs": 0,
    "ecg": "Normal", "maxhr": 150, "exang": "No", "oldpeak": 1.0, "slope": "Upsloping",
}
EXAMPLES = {
    "Example 1": {"age": 35, "sex": "Female", "cp": "Atypical Angina", "bp": 115, "chol": 190, "fbs": 0,
                  "ecg": "Normal", "maxhr": 175, "exang": "No", "oldpeak": 0.0, "slope": "Upsloping"},
    "Example 2": {"age": 60, "sex": "Male", "cp": "Asymptomatic", "bp": 150, "chol": 260, "fbs": 1,
                  "ecg": "ST", "maxhr": 110, "exang": "Yes", "oldpeak": 2.5, "slope": "Flat"},
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_values(values: dict) -> None:
    for key, value in values.items():
        st.session_state[key] = value


# ---------- Load saved artifacts safely ----------
@st.cache_resource
def load_artifacts():
    root = Path(__file__).resolve().parent
    files = {
        "model": root / "LogisticRegression_heart.pkl",
        "scaler": root / "scaler.pkl",
        "columns": root / "columns.pkl",
    }
    missing = [p.name for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))
    return (
        joblib.load(files["model"]),
        joblib.load(files["scaler"]),
        list(joblib.load(files["columns"])),
    )


try:
    model, scaler, expected_columns = load_artifacts()
except Exception as exc:
    st.error("The prediction model could not be loaded.")
    st.code(str(exc))
    st.stop()


def build_features(values: dict) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode inputs and align them to the training columns.

    Every candidate dummy for the selected category is written, then the frame is
    re-indexed to `expected_columns`. This works whether the model was trained with
    drop_first=True or False. Returns the aligned frame and a list of problems.
    """
    row = {
        "Age": values["age"], "RestingBP": values["bp"], "Cholesterol": values["chol"],
        "FastingBS": values["fbs"], "MaxHR": values["maxhr"], "Oldpeak": values["oldpeak"],
        f"Sex_{SEX[values['sex']]}": 1,
        f"ChestPainType_{CHEST_PAIN[values['cp']]}": 1,
        f"RestingECG_{RESTING_ECG[values['ecg']]}": 1,
        f"ExerciseAngina_{EXERCISE_ANGINA[values['exang']]}": 1,
        f"ST_Slope_{ST_SLOPE[values['slope']]}": 1,
    }
    problems = []
    numeric = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
    problems += [f"Numeric column not found in model: {c}" for c in numeric if c not in expected_columns]
    for prefix in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
        if not any(c.startswith(prefix + "_") for c in expected_columns):
            problems.append(f"No '{prefix}_*' columns in the saved model; naming may differ.")
    frame = pd.DataFrame([row]).reindex(columns=expected_columns, fill_value=0)
    return frame, problems


# ---------- Hero ----------
st.markdown(
    """
    <section class="hero">
      <div class="eyebrow"><span class="dot"></span> MACHINE LEARNING · HEALTH DATA EXPLORER</div>
      <h1>Heart Disease <span>Prediction</span></h1>
      <div class="hero-sub">
        Explore how a trained Logistic Regression model classifies a set of selected clinical
        parameters. Enter example values, review the model output, and see which inputs
        pushed the result.
      </div>
      <div class="pills">
        <span>♡ Logistic Regression</span><span>⌁ 11 clinical inputs</span><span>◈ Interactive ML demo</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🫀 CardioLens")
    st.caption("Heart Disease Prediction · ML demo")
    st.divider()
    st.markdown("### Quick start")
    st.button("Load Example 1", on_click=apply_values, args=(EXAMPLES["Example 1"],), use_container_width=True)
    st.button("Load Example 2", on_click=apply_values, args=(EXAMPLES["Example 2"],), use_container_width=True)
    st.button("Reset to defaults", on_click=apply_values, args=(DEFAULTS,), use_container_width=True)
    st.divider()
    st.markdown("### Model pipeline")
    st.markdown("**Inputs** → one-hot alignment → scaling → Logistic Regression")
    st.divider()
    st.markdown("**Educational use only.** Not a medical device. A reassuring result never rules out heart disease.")

# ---------- Workspace ----------
st.markdown('<div class="section-kicker">01 / Prediction workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Use plausible example values, or load one from the sidebar.</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1.2, 0.8], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown("### ◈ Clinical parameters")
        with st.form("heart_prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.slider("Age", 1, 100, key="age", help="Age in years.")
                sex = st.selectbox("Sex", list(SEX), key="sex")
                chest_pain_type = st.selectbox("Chest pain type", list(CHEST_PAIN), key="cp",
                                               help="Dataset category; this does not diagnose chest pain.")
                resting_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 200, step=1, key="bp")
                cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, step=1, key="chol")
                fasting_blood_sugar = st.selectbox(
                    "Fasting blood sugar > 120 mg/dL", [0, 1], key="fbs",
                    format_func=lambda v: "Yes (1)" if v == 1 else "No (0)",
                )
            with c2:
                resting_ecg = st.selectbox("Resting ECG result", list(RESTING_ECG), key="ecg")
                max_hr = st.slider("Maximum heart rate achieved", 60, 220, key="maxhr",
                                   help="A model feature, not a fitness target.")
                exercise_angina = st.selectbox("Exercise-induced angina", list(EXERCISE_ANGINA), key="exang")
                oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.0, step=0.1, key="oldpeak")
                st_slope = st.selectbox("Peak exercise ST slope", list(ST_SLOPE), key="slope")
            submitted = st.form_submit_button("♡  Run prediction", type="primary", use_container_width=True)

with right_col:
    with st.container(border=True):
        st.markdown("### What happens next?")
        st.markdown(
            """
            <div class="steps">
              <div class="step"><b>01 · Inputs</b><small>Numeric and categorical values.</small></div>
              <div class="step"><b>02 · Prepare</b><small>One-hot alignment and saved scaler.</small></div>
              <div class="step"><b>03 · Classify</b><small>Logistic Regression predicts a class.</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("Input fields", "11")
        m2.metric("Model features", str(len(expected_columns)))
        st.markdown(
            '<div class="notice">Demonstration only. This is not a validated personal risk estimate '
            "and cannot replace advice from a health professional.</div>",
            unsafe_allow_html=True,
        )

# ---------- Output ----------
st.divider()
st.markdown('<div class="section-kicker">02 / Model output</div>', unsafe_allow_html=True)

if submitted:
    values = {
        "age": age, "sex": sex, "cp": chest_pain_type, "bp": resting_bp, "chol": cholesterol,
        "fbs": fasting_blood_sugar, "ecg": resting_ecg, "maxhr": max_hr,
        "exang": exercise_angina, "oldpeak": oldpeak, "slope": st_slope,
    }
    input_df, problems = build_features(values)

    if problems:
        st.error("Input encoding does not match the saved model, so the result would be unreliable:")
        for p in problems:
            st.write("• " + p)
        with st.expander("Columns the model expects"):
            st.code("\n".join(expected_columns))
        st.stop()

    if max_hr > 220 - age:
        st.warning("Maximum heart rate is above the common age-based estimate (220 − age). Please double-check it.")

    scaled_input = scaler.transform(input_df)
    predicted_class = int(model.predict(scaled_input)[0])
    probability = float(model.predict_proba(scaled_input)[0][1]) if hasattr(model, "predict_proba") else None

    result_col, details_col = st.columns([1.2, 0.8], gap="large")
    with result_col:
        positive = predicted_class == 1
        heading = "Model output: positive class" if positive else "Model output: negative class"
        copy = (
            "The classifier assigned these inputs to its positive class (label 1). This does not "
            "establish that anyone has heart disease. Seek a qualified professional for real concerns."
            if positive else
            "The classifier assigned these inputs to its negative class (label 0). This cannot rule "
            "out heart disease. Seek medical advice for symptoms, whatever this demo says."
        )
        st.markdown(
            f"""
            <div class="result-card {'alert' if positive else 'calm'}">
              <div class="overline">Classification completed</div>
              <h2>{heading}</h2><p>{copy}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if probability is not None:
            st.write(f"**Model score for class 1: {probability:.0%}**")
            st.progress(probability)
            st.caption("An uncalibrated model score on a small public dataset, not a medical probability.")

        # Which inputs pushed the result? (coefficient x scaled value)
        if hasattr(model, "coef_"):
            contrib = pd.Series(model.coef_[0] * scaled_input[0], index=expected_columns)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
            st.markdown("#### What pushed the result")
            st.bar_chart(top.sort_values())
            st.caption("Positive bars push toward class 1, negative toward class 0. This describes the model, not causes.")

    with details_col:
        with st.container(border=True):
            st.markdown("### Submitted input summary")
            st.metric("Predicted class label", str(predicted_class))
            summary_df = pd.DataFrame(
                [
                    ("Age", age, "years"),
                    ("Resting blood pressure", resting_bp, "mm Hg"),
                    ("Cholesterol", cholesterol, "mg/dL"),
                    ("Fasting blood sugar > 120", "Yes" if fasting_blood_sugar == 1 else "No", ""),
                    ("Maximum heart rate", max_hr, "bpm"),
                    ("Oldpeak", oldpeak, "ST depression"),
                    ("Sex", sex, ""),
                    ("Chest pain type", chest_pain_type, ""),
                    ("Resting ECG", resting_ecg, ""),
                    ("Exercise-induced angina", exercise_angina, ""),
                    ("ST slope", st_slope, ""),
                ],
                columns=["Input", "Value", "Unit / note"],
            ).astype({"Value": str})
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            with st.expander("Encoded vector sent to the model"):
                st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)
else:
    st.markdown(
        """
        <div class="result-card">
          <div class="overline">Awaiting submitted values</div>
          <h2>Your model output will appear here</h2>
          <p>Fill in the form (or load an example from the sidebar) and select <b>Run prediction</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("CardioLens · Heart Disease Prediction ML Demo · For educational use only, not a medical diagnosis.")
