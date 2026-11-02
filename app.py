from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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
    @keyframes pop { 0% { transform: scale(.96); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
    .verdict { padding: 30px 24px; border-radius: 22px; text-align: center; border: 2px solid; margin: 4px 0 14px; animation: pop .45s ease; }
    .verdict.positive { background: linear-gradient(145deg,#6a2132,#2b1019); border-color: #ff7b8a;
      box-shadow: 0 0 0 5px rgba(255,123,138,.13), 0 22px 60px rgba(255,90,110,.28); }
    .verdict.negative { background: linear-gradient(145deg,#0f5148,#0a2327); border-color: #71e5d1;
      box-shadow: 0 0 0 5px rgba(113,229,209,.12), 0 22px 60px rgba(60,220,190,.22); }
    .verdict .tag { font-size: .74rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; color: rgba(255,255,255,.72); margin-bottom: 10px; }
    .verdict .big { font-family: 'Space Grotesk','DM Sans',sans-serif; font-size: clamp(2rem,4.5vw,3rem); font-weight: 700; line-height: 1.05; color: #fff; margin-bottom: 6px; }
    .verdict .score { font-size: 1.05rem; color: rgba(255,255,255,.88); margin-bottom: 14px; }
    .verdict .score b { font-size: 1.5rem; }
    .verdict p { color: rgba(255,255,255,.80); font-size: .9rem; line-height: 1.6; margin: 0; }
    .verdict .meaning { font-size: 1rem; font-weight: 600; color: #fff; margin-bottom: 10px; }
    .chip { display: inline-block; padding: 4px 12px; border-radius: 999px; border: 1px solid rgba(255,255,255,.35);
      font-size: .78rem; font-weight: 600; color: #fff; background: rgba(255,255,255,.10); margin-bottom: 12px; }
    .legend { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 14px; margin: 8px 0 14px; }
    .legend > div { padding: 16px; border-radius: 14px; border: 1px solid; }
    .legend .c1 { border-color: rgba(255,123,138,.5); background: rgba(255,123,138,.07); }
    .legend .c0 { border-color: rgba(113,229,209,.5); background: rgba(113,229,209,.07); }
    .legend b { display: block; font-size: 1rem; color: #fff; margin-bottom: 6px; }
    .legend span { color: #b8c5db; font-size: .88rem; line-height: 1.6; }
    @media(max-width: 600px) { .legend { grid-template-columns: 1fr; } }
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

# ---------- Run the model (before drawing the right column so the result can sit beside the form) ----------
result = None
if submitted:
    values = {
        "age": age, "sex": sex, "cp": chest_pain_type, "bp": resting_bp, "chol": cholesterol,
        "fbs": fasting_blood_sugar, "ecg": resting_ecg, "maxhr": max_hr,
        "exang": exercise_angina, "oldpeak": oldpeak, "slope": st_slope,
    }
    input_df, problems = build_features(values)
    result = {"problems": problems, "input_df": input_df}
    if not problems:
        scaled_input = scaler.transform(input_df)
        result["scaled"] = scaled_input
        result["cls"] = int(model.predict(scaled_input)[0])
        result["proba"] = float(model.predict_proba(scaled_input)[0][1]) if hasattr(model, "predict_proba") else None

with right_col:
    st.markdown('<div id="result-anchor"></div>', unsafe_allow_html=True)

    if result is None:
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
                '<div class="notice">Your result will appear here, right next to the form, '
                "after you press <b>Run prediction</b>.</div>",
                unsafe_allow_html=True,
            )
    elif result["problems"]:
        st.error("Input encoding does not match the saved model, so the result would be unreliable:")
        for p in result["problems"]:
            st.write("• " + p)
        with st.expander("Columns the model expects"):
            st.code("\n".join(expected_columns))
    else:
        positive = result["cls"] == 1
        proba = result["proba"]
        score_html = f'<div class="score">Model score for class 1: <b>{proba:.0%}</b></div>' if proba is not None else ""
        if proba is None:
            chip = ""
        else:
            strength = abs(proba - 0.5) * 2  # 0 = coin flip, 1 = very sure
            level = "High" if strength >= 0.6 else "Moderate" if strength >= 0.3 else "Low"
            chip = f'<div class="chip">Model confidence: {level}</div>'
        meaning = (
            "Class 1: similar to patients with heart disease in the training data"
            if positive else
            "Class 0: similar to patients without heart disease in the training data"
        )
        st.markdown(
            f"""
            <div class="verdict {'positive' if positive else 'negative'}">
              <div class="tag">Prediction result</div>
              <div class="big">{'POSITIVE' if positive else 'NEGATIVE'} · Class {result['cls']}</div>
              <div class="meaning">{meaning}</div>
              {score_html}
              {chip}
              <p>{'This is a pattern match, not a diagnosis. Consider discussing it with a doctor.'
                  if positive else
                  'This does not rule out heart disease. See a doctor if you have symptoms.'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if proba is not None:
            st.progress(proba)
        if max_hr > 220 - age:
            st.warning("Maximum heart rate is above the common age-based estimate (220 − age). Please double-check it.")
        st.markdown(
            '<div class="notice">Educational demo only, not a diagnosis. Please see a qualified '
            "health professional for any real concern.</div>",
            unsafe_allow_html=True,
        )
        # Bring the result into view (matters on phones, where the column stacks below the form)
        components.html(
            """<script>
            const el = window.parent.document.getElementById('result-anchor');
            if (el) el.scrollIntoView({behavior: 'smooth', block: 'start'});
            </script>""",
            height=0,
        )

# ---------- Details (only after a successful run) ----------
if result is not None and not result["problems"]:
    st.divider()
    st.markdown('<div class="section-kicker">Details · why the model said this</div>', unsafe_allow_html=True)
    chart_col, table_col = st.columns([1.1, 0.9], gap="large")

    with chart_col:
        with st.container(border=True):
            if hasattr(model, "coef_"):
                contrib = pd.Series(model.coef_[0] * result["scaled"][0], index=expected_columns)
                top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(6)
                st.markdown("### What pushed the result")
                st.bar_chart(top.sort_values())
                st.caption("Positive bars push toward class 1, negative toward class 0. This describes the model, not causes.")
            else:
                st.info("This model does not expose coefficients.")

    with table_col:
        with st.container(border=True):
            st.markdown("### Submitted input summary")
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
                st.dataframe(result["input_df"].T.rename(columns={0: "value"}), use_container_width=True)

# ---------- How to read the result (always visible) ----------
st.divider()
st.markdown('<div class="section-kicker">Understanding the result</div>', unsafe_allow_html=True)
with st.container(border=True):
    st.markdown("### How to read this result")
    st.markdown(
        """
        <div class="legend">
          <div class="c1"><b>Class 1 · Positive</b>
            <span>The model thinks these values resemble patients who <b>had heart disease</b> in its training data.
            It is a flag to talk to a doctor, not a diagnosis.</span></div>
          <div class="c0"><b>Class 0 · Negative</b>
            <span>The model thinks these values resemble patients who <b>did not have heart disease</b> in its training data.
            It is <b>not</b> a clean bill of health.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **About the model score and confidence**
        - The score is how strongly the model leans toward class 1. Near **0%** means it leans class 0, near **100%** class 1, and around **50%** means it is unsure.
        - Confidence only describes how far the score is from 50%. It says how decisive the model is, **not** how likely you are to have the disease.
        - The score is **not a medical risk percentage**. The model was trained on a small public dataset and has not been clinically validated.

        **What it cannot do**
        - It cannot see your symptoms, family history, medicines, scans or blood tests beyond these 11 numbers.
        - A wrong answer is possible in both directions, so a negative result can miss real disease.

        **When to get real medical help**
        - Chest pain or pressure, shortness of breath, fainting, or pain spreading to the arm or jaw need urgent care. Call your local emergency number (112 in India) instead of using this app.
        - For routine concerns such as blood pressure, cholesterol or family history, see a doctor for proper tests.
        """
    )

st.divider()
st.caption("CardioLens · Heart Disease Prediction ML Demo · For educational use only, not a medical diagnosis.")
