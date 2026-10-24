# 🫀 Heart Disease Prediction — ML Demo

An interactive Streamlit app that demonstrates how a saved Logistic Regression model classifies selected clinical parameters. The interface includes grouped inputs, a model-output panel, an input summary, and a clear educational-use disclaimer.

> **Medical disclaimer:** This app is for learning and demonstration only. It is not a medical device, does not provide a diagnosis, and must not be used to rule out disease or make healthcare decisions. Consult a qualified healthcare professional for personal medical concerns.

## ✨ Features

- **Polished Streamlit interface** with a dark, clinical dashboard theme
- **11 input fields** covering age, sex, chest-pain category, blood pressure, cholesterol, fasting blood sugar, ECG category, maximum heart rate, exercise-induced angina, Oldpeak, and ST slope
- **Saved Logistic Regression model** loaded from a pickle artifact
- **Feature alignment and scaling** using the saved column schema and fitted scaler
- **Model output panel** with a plain-language explanation and a summary of submitted values
- **Input guidance** for the example clinical features

## 🧠 Prediction pipeline

```text
Example clinical inputs
       ↓
DataFrame feature assembly
       ↓
Align to saved one-hot feature columns
       ↓
Saved StandardScaler
       ↓
Saved Logistic Regression model
       ↓
Predicted class label (0 or 1)
```

The app preserves the saved model, scaler, feature column order, and input-to-feature mapping from the repository. It does not calculate a medically validated individual risk score.

## 🛠️ Technology

- Python
- Streamlit
- pandas
- scikit-learn
- joblib

## 📁 Project structure

```text
Heart-Disease-Prediction-App/
├── app.py                       # Streamlit UI and inference pipeline
├── LogisticRegression_heart.pkl # Saved classifier
├── scaler.pkl                   # Fitted scaler
├── columns.pkl                  # Expected feature order
├── requirements.txt             # Python dependencies
└── README.md
```

## 🚀 Run locally

Requires Python 3.10+ and pip.

```powershell
# Clone the repository
git clone https://github.com/abhiii04-25/Heart-Disease-Prediction-App.git
cd Heart-Disease-Prediction-App

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
python -m pip install -r requirements.txt

# Start the app
python -m streamlit run app.py
```

Streamlit normally opens the local application at `http://localhost:8501`.

## ☁️ Deploy to Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub.
2. Choose **Create app** and select this repository.
3. Use branch `main` and main file `app.py`.
4. Click **Deploy** and review the app logs if startup fails.

The repository contains the model, scaler, and expected feature-column artifacts needed by the app. If the saved model was produced with an older scikit-learn version, deployment may require matching that version in `requirements.txt`.

## ⚠️ Important limitations

- The model's predicted class is a machine-learning output, not a clinical diagnosis.
- The app's saved model and scaler should be validated and versioned together.
- A negative model output does not rule out heart disease, and a positive output does not confirm it.
- Do not use this demonstration for emergency triage, treatment choices, or personal medical decisions.

## 👨‍💻 Author

Built by Abhishek Sontakke as an end-to-end machine-learning learning project.

Source: [Heart-Disease-Prediction-App on GitHub](https://github.com/abhiii04-25/Heart-Disease-Prediction-App)
