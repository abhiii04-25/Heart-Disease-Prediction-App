# Heart-Disease-Prediction-App
A machine learning project that predicts the likelihood of heart disease from clinical parameters using Logistic Regression, deployed as an interactive Streamlit web app. Built as part of my journey learning ML — covering data preprocessing, feature encoding, model training, and deployment.
# ❤️ Heart Disease Prediction using Machine Learning

A machine learning project that predicts the likelihood of heart disease from a patient's clinical parameters using **Logistic Regression**, deployed as an interactive **Streamlit** web app. This project was built as part of my journey learning machine learning — applying core ML concepts end-to-end, from raw data to a working, deployable app.

## 📌 Overview

This app takes standard clinical inputs (age, blood pressure, cholesterol, ECG results, etc.), preprocesses them to match the model's expected feature format, scales them, and returns a prediction of whether the patient is likely or unlikely to have heart disease.

> ⚠️ **Disclaimer:** This tool is for educational/demonstration purposes only and is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional.

## 🎯 What I Learned / Practiced

This project helped me apply several core machine learning concepts hands-on:

- **Data preprocessing** — handling categorical variables, encoding (one-hot encoding), and feature alignment
- **Feature scaling** — using `StandardScaler` to normalize numerical inputs before feeding them to the model
- **Classification modeling** — training and evaluating a **Logistic Regression** classifier for a binary classification problem
- **Model persistence** — saving/loading trained models, scalers, and column schemas with `joblib`
- **Model deployment** — turning a trained model into a usable, interactive web app with **Streamlit**
- **End-to-end ML workflow** — going from raw tabular data to a deployed prediction tool, not just a notebook

## 🚀 Demo

Run locally with Streamlit (see [Getting Started](#-getting-started) below).

## 🧠 How It Works

1. The user enters clinical details through interactive sliders and dropdowns.
2. Inputs are assembled into a single-row DataFrame and one-hot encoded to match the model's training feature set.
3. Any missing expected columns are filled with `0` to align with the format the model was trained on.
4. Features are scaled using a pre-fitted `StandardScaler`.
5. The trained Logistic Regression model predicts the outcome (`1` = likely heart disease, `0` = unlikely).
6. The result is displayed with a clear success/error message.

## 🩺 Input Features

| Feature | Description |
|---|---|
| Age | Patient's age in years |
| Sex | Male / Female |
| Chest Pain Type | Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic |
| Resting Blood Pressure | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| Fasting Blood Sugar | Whether fasting blood sugar > 120 mg/dl (0/1) |
| Resting ECG | Normal, ST, LVH |
| Max Heart Rate | Maximum heart rate achieved |
| Exercise Angina | Exercise-induced angina (Y/N) |
| Oldpeak | ST depression induced by exercise relative to rest |
| ST Slope | Slope of the peak exercise ST segment (Upsloping, Flat, Downsloping) |

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — web app UI
- **scikit-learn** — model training, scaling, and inference
- **pandas** — data handling
- **joblib** — model/scaler/artifact serialization

## 📂 Project Structure

```
heart-disease-prediction/
├── app.py                       # Streamlit application
├── LogisticRegression_heart.pkl # Trained Logistic Regression model
├── scaler.pkl                   # Fitted StandardScaler
├── columns.pkl                  # Expected feature columns (post one-hot encoding)
├── requirements.txt             # Python dependencies
└── README.md
```

## ⚙️ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## 📦 requirements.txt

```
streamlit
pandas
scikit-learn
joblib
```

## 📊 Dataset

The model was trained on the **Heart Failure Prediction Dataset** (commonly sourced from Kaggle / UCI), which includes clinical attributes such as age, sex, chest pain type, blood pressure, cholesterol, ECG results, and more, labeled with the presence or absence of heart disease.

## 🔮 Future Improvements

As I continue learning, I plan to extend this project with:

- Model performance metrics (accuracy, precision, recall, ROC-AUC) displayed in the app
- Comparing Logistic Regression against other classifiers (Random Forest, XGBoost, SVM)
- Cross-validation and hyperparameter tuning
- Input validation and helpful tooltips for each field
- Deploying the app (e.g., Streamlit Community Cloud) and linking a live demo here

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙋‍♂️ Author

Built by Abhishek Sontakke — feel free to connect or raise an issue if you spot a bug!
