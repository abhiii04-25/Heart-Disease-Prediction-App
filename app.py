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
      --bg: #08111f;
      --panel: #111e30;
      --line: rgba(161, 186, 221, .16);
      --muted: #9babc4;
      --text: #f4f7ff;
      --mint: #71e5d1;
      --blue: #8bc8ff;
      --coral: #ff9e9e;
    }

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, [data-testid="stMetricValue"] {
      font-family: 'Space Grotesk', 'DM Sans', sans-serif !important;
    }

    [data-testid="stAppViewContainer"] {
      color: var(--text);
      background:
        radial-gradient(ellipse at 78% -8%, rgba(78, 109, 194, .20), transparent 36%),
        radial-gradient(ellipse at 0% 35%, rgba(62, 210, 190, .08), transparent 32%),
        var(--bg);
    }

    [data-testid="stHeader"] { background: rgba(8, 17, 31, .35); }
    main .block-container { max-width: 1320px; padding-top: 1.7rem; padding-bottom: 3rem; }

    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #111d2f 0%, #0b1423 100%);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b8c5dc; }

    .hero {
      position: relative;
      overflow: hidden;
      padding: clamp(24px, 4vw, 43px);
      margin: 5px 0 27px;
      border: 1px solid rgba(139, 184, 232, .23);
      border-radius: 26px;
      background:
        radial-gradient(circle at 91% 12%, rgba(255, 158, 158, .13), transparent 28%),
        radial-gradient(circle at 7% 110%, rgba(113, 229, 209, .13), transparent 37%),
        linear-gradient(135deg, rgba(22, 37, 59, .97), rgba(10, 20, 35, .98));
      box-shadow: 0 24px 70px rgba(0, 0, 0, .19);
    }
    .hero:after {
      content: "♡";
      position: absolute;
      right: 4%;
      bottom: -75px;
      color: rgba(255, 170, 179, .07);
      font-size: clamp(180px, 24vw, 310px);
      line-height: 1;
      pointer-events: none;
      transform: rotate(-12deg);
    }
    .eyebrow {
      position: relative;
      z-index: 1;
      display: flex;
      align-items: center;
      gap: 9px;
      color: #a9f4e9;
      font-size: .73rem;
      font-weight: 700;
      letter-spacing: .14em;
      text-transform: uppercase;
      margin-bottom: 16px;
    }
    .dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--mint); box-shadow: 0 0 13px rgba(113, 229, 209, .7);
    }
    .hero h1 {
      position: relative;
      z-index: 1;
      max-width: 820px;
      color: #f5f7ff;
      font-size: clamp(2.2rem, 5vw, 4.15rem);
      line-height: 1.05;
      letter-spacing: -.055em;
      margin: 0 0 13px;
      font-weight: 700;
    }
    .hero h1 span {
      background: linear-gradient(90deg, var(--mint), #9dbdff 68%, #d4acff);
      -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .hero-sub {
      position: relative; z-index: 1;
      max-width: 760px;
      color: #bac7dd;
      font-size: 1.04rem;
      line-height: 1.7;
      margin-bottom: 21px;
    }
    .pills { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 9px; }
    .pills span {
      padding: 7px 11px;
      border: 1px solid rgba(165, 188, 229, .19);
      border-radius: 999px;
      background: rgba(255,255,255,.045);
      color: #dce6f8;
      font-size: .78rem;
    }
    .section-kicker {
      color: var(--mint); text-transform: uppercase; letter-spacing: .15em;
      font-size: .72rem; font-weight: 700; margin: 4px 0 5px;
    }
    .section-copy { color: var(--muted); margin: 0 0 18px; font-size: .94rem; }
    .section-heading {
      color: #f3f6ff; font-size: 1.2rem; font-weight: 700;
      margin: 0 0 5px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--line) !important;
      border-radius: 19px !important;
      background: linear-gradient(145deg, rgba(19, 31, 50, .91), rgba(11, 21, 36, .95));
      box-shadow: 0 12px 34px rgba(0,0,0,.10);
    }
    [data-testid="stMetric"] {
      padding: 15px 17px;
      border: 1px solid rgba(151, 179, 225, .16);
      border-radius: 15px;
      background: linear-gradient(145deg, rgba(25, 42, 65, .93), rgba(14, 26, 43, .94));
    }
    [data-testid="stMetricLabel"] { color: #acbad1 !important; font-size: .82rem; }
    [data-testid="stMetricValue"] { color: #f4f7ff !important; font-weight: 700; }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] [data-baseweb="input"] > div {
      border-radius: 11px; border-color: rgba(153, 179, 226, .20);
      background: rgba(7, 15, 28, .70);
    }
    [data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .35rem; }
    [data-testid="stRadio"] label { color: #dbe5f6; }
    .stButton > button, [data-testid="stFormSubmitButton"] button {
      border-radius: 12px; font-weight: 700; min-height: 45px;
      transition: transform .15s ease, filter .15s ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
      transform: translateY(-1px); filter: brightness(1.06);
    }
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
      color: #071620; background: linear-gradient(100deg, #71e5d1, #9cebdc);
      border: 0; box-shadow: 0 7px 22px rgba(72, 212, 190, .14);
    }
    .result-card {
      padding: 24px;
      border-radius: 18px;
      border: 1px solid rgba(139, 184, 232, .20);
      background: linear-gradient(140deg, rgba(24, 39, 61, .96), rgba(12, 22, 38, .98));
      margin: 6px 0 15px;
    }
    .result-card.alert { border-color: rgba(255, 158, 158, .43); }
    .result-card.calm { border-color: rgba(113, 229, 209, .35); }
    .result-card .overline {
      color: var(--muted); font-size: .72rem; letter-spacing: .13em;
      text-transform: uppercase; font-weight: 700; margin-bottom: 10px;
    }
    .result-card h2 { color: #f5f7ff; font-size: clamp(1.4rem, 3vw, 2rem); margin: 0 0 10px; }
    .result-card p { color: #b8c5db; line-height: 1.65; margin: 0; }
    .notice {
      padding: 13px 15px; border-radius: 12px;
      border: 1px solid rgba(255, 198, 132, .25);
      background: rgba(255, 185, 90, .07);
      color: #f0d6b3; font-size: .84rem; line-height: 1.55;
    }
    .steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 12px 0; }
    .step {
      padding: 15px 13px; border: 1px solid rgba(151,179,225,.16); border-radius: 14px;
      background: rgba(255,255,255,.025);
    }
    .step b { display: block; color: #e7eeff; font-size: .85rem; margin-bottom: 5px; }
    .step small { color: #92a2bd; line-height: 1.5; font-size: .76rem; }
    [data-testid="stAlert"] { border-radius: 12px; }
    @media(max-width: 900px) {
      main .block-container { padding: 1rem 1rem 2rem; }
      .steps { grid-template-columns: 1fr; }
    }
    @media(max-width: 600px) {
      .hero { border-radius: 19px; padding: 23px 19px; }
      .hero h1 { font-size: 2.25rem; }
      .hero-sub { font-size: .96rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Load saved artifacts safely ----------
@st.cache_resource
def load_artifacts():
    root = Path(__file__).resolve().parent
    required_files = {
        "model": root / "LogisticRegression_heart.pkl",
        "scaler": root / "scaler.pkl",
        "columns": root / "columns.pkl",
    }
    missing = [path.name for path in required_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing model files: " + ", ".join(missing))
    model = joblib.load(required_files["model"])
    scaler = joblib.load(required_files["scaler"])
    expected_columns = joblib.load(required_files["columns"])
    return model, scaler, expected_columns


try:
    model, scaler, expected_columns = load_artifacts()
except Exception as exc:
    st.error("The prediction model could not be loaded.")
    st.code(str(exc))
    st.stop()


# ---------- Hero ----------
st.markdown(
    """
    <section class="hero">
      <div class="eyebrow"><span class="dot"></span> MACHINE LEARNING · HEALTH DATA EXPLORER</div>
      <h1>Heart Disease <span>Prediction</span></h1>
      <div class="hero-sub">
        Explore how a trained Logistic Regression model classifies a set of selected clinical
        parameters. Enter the example values, review the model output, and learn how the
        prediction pipeline works.
      </div>
      <div class="pills">
        <span>♡ Logistic Regression</span>
        <span>⌁ 11 clinical inputs</span>
        <span>◈ Interactive ML demo</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🫀 CardioLens")
    st.caption("Heart Disease Prediction · ML demo")
    st.divider()
    st.markdown("### How to use")
    st.markdown(
        "1. Enter the example clinical values.\n"
        "2. Select **Run prediction**.\n"
        "3. Read the model output and its limitations."
    )
    st.divider()
    st.markdown("### Model pipeline")
    st.markdown("**Clinical inputs** → feature alignment → scaling → Logistic Regression")
    st.divider()
    st.markdown(
        """
        **Educational use only**

        This model is not a medical device. Its output is not a diagnosis, and a reassuring
        result must not be used to rule out heart disease.
        """
    )

st.markdown('<div class="section-kicker">01 / Prediction workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">Use plausible example values for a demonstration. These controls do not assess symptoms or replace clinical advice.</div>', unsafe_allow_html=True)

left_col, right_col = st.columns([1.2, 0.8], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown("### ◈ Clinical parameters")
        st.caption("All fields are required for the model input.")
        with st.form("heart_prediction_form"):
            top1, top2 = st.columns(2)
            with top1:
                age = st.slider("Age", 1, 100, 40, help="Age in years.")
                sex = st.selectbox("Sex", ["Male", "Female"], help="Category used by the trained model.")
                chest_pain_type = st.selectbox(
                    "Chest pain type",
                    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
                    help="Use the dataset category for this demonstration; it does not diagnose chest pain.",
                )
                resting_bp = st.number_input(
                    "Resting blood pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1
                )
                cholesterol = st.number_input(
                    "Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1
                )
                fasting_blood_sugar = st.selectbox(
                    "Fasting blood sugar > 120 mg/dL", [0, 1],
                    format_func=lambda value: "Yes (1)" if value == 1 else "No (0)",
                )
            with top2:
                resting_ecg = st.selectbox(
                    "Resting ECG result", ["Normal", "ST", "LVH"],
                    help="Select the category expected by the training dataset.",
                )
                max_hr = st.slider(
                    "Maximum heart rate achieved", 60, 220, 150,
                    help="Value entered as a model feature, not a fitness target.",
                )
                exercise_angina = st.selectbox(
                    "Exercise-induced angina", ["Y", "N"],
                    format_func=lambda value: "Yes" if value == "Y" else "No",
                )
                oldpeak = st.slider(
                    "Oldpeak (ST depression)", 0.0, 6.0, 1.0, 0.1,
                    help="Value from the dataset's exercise/rest comparison.",
                )
                st_slope = st.selectbox(
                    "Peak exercise ST slope", ["Upsloping", "Flat", "Downsloping"]
                )
                st.markdown(
                    '<div class="notice">Enter sample values for demonstration. Only a qualified health professional can interpret clinical measurements in context.</div>',
                    unsafe_allow_html=True,
                )
            submitted = st.form_submit_button("♡  Run prediction", type="primary", use_container_width=True)

with right_col:
    with st.container(border=True):
        st.markdown("### What happens next?")
        st.markdown(
            """
            The app formats the values to match the model's saved feature columns, applies the
            saved scaler, and passes the result to Logistic Regression.
            """
        )
        st.markdown(
            """
            <div class="steps">
              <div class="step"><b>01 · Inputs</b><small>Age, blood pressure, cholesterol and categorical values.</small></div>
              <div class="step"><b>02 · Prepare</b><small>One-hot feature alignment and saved scaling step.</small></div>
              <div class="step"><b>03 · Classify</b><small>Saved Logistic Regression model produces a class.</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### Model information")
        m1, m2 = st.columns(2)
        m1.metric("Input fields", "11")
        m2.metric("Algorithm", "Logistic Regression")
        st.caption("This interface demonstrates an ML workflow. It does not estimate a medically validated personal risk.")

st.divider()
st.markdown('<div class="section-kicker">02 / Model output</div>', unsafe_allow_html=True)
st.markdown('<div class="section-copy">The output below appears after you submit the form. Treat it as an educational model result only.</div>', unsafe_allow_html=True)

if submitted:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_blood_sugar,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex" + sex: 1,
        "ChestPainType" + chest_pain_type: 1,
        "RestingECG" + resting_ecg: 1,
        "ExerciseAngina" + exercise_angina: 1,
        "ST_Slope" + st_slope: 1,
    }
    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    predicted_class = int(prediction[0])

    result_col, details_col = st.columns([1.2, 0.8], gap="large")
    with result_col:
        is_positive = predicted_class == 1
        card_class = "alert" if is_positive else "calm"
        result_heading = (
            "Model output: positive class"
            if is_positive
            else "Model output: negative class"
        )
        result_copy = (
            "The saved classifier assigned these inputs to its positive class (label 1). "
            "This does not establish that the person has heart disease. Please seek advice "
            "from a qualified healthcare professional for real health concerns."
            if is_positive
            else "The saved classifier assigned these inputs to its negative class (label 0). "
            "This cannot rule out heart disease. Seek professional medical advice for symptoms "
            "or concerns, regardless of this demo result."
        )
        st.markdown(
            f"""
            <div class="result-card {card_class}">
              <div class="overline">Classification completed</div>
              <h2>{result_heading}</h2>
              <p>{result_copy}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="notice">Safety note: this is an educational machine-learning demonstration, not a clinical assessment or emergency service.</div>',
            unsafe_allow_html=True,
        )
    with details_col:
        with st.container(border=True):
            st.markdown("### Submitted input summary")
            st.metric("Predicted class label", str(predicted_class))
            st.write("**Values passed to the model**")
            summary_df = pd.DataFrame(
                [
                    ("Age", age, "years"),
                    ("Resting blood pressure", resting_bp, "mm Hg"),
                    ("Cholesterol", cholesterol, "mg/dL"),
                    ("Fasting blood sugar > 120", "Yes" if fasting_blood_sugar == 1 else "No", ""),
                    ("Maximum heart rate", max_hr, "model input"),
                    ("Oldpeak", oldpeak, "model input"),
                    ("Sex", sex, ""),
                    ("Chest pain type", chest_pain_type, ""),
                    ("Resting ECG", resting_ecg, ""),
                    ("Exercise-induced angina", "Yes" if exercise_angina == "Y" else "No", ""),
                    ("ST slope", st_slope, ""),
                ],
                columns=["Input", "Value", "Unit / note"],
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
else:
    with st.container(border=True):
        st.markdown(
            """
            <div class="result-card">
              <div class="overline">Awaiting submitted values</div>
              <h2>Your model output will appear here</h2>
              <p>Complete the parameter form and select <b>Run prediction</b>. The interface will display the classifier's output and a summary of the inputs you submitted.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "CardioLens · Heart Disease Prediction ML Demo · Logistic Regression. "
    "For educational use only — not a medical diagnosis."
)
