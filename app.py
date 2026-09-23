"""
app.py
------
Flask website for the project. Loads the artifacts saved by train.py and
serves a form where a user enters the 13 clinical values; returns the
final proposed system's prediction (stacked ensemble of the hybrid
CNN-BiLSTM-Attention model + Random Forest + SVM).

Run (AFTER train.py has finished at least once):
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import json
import joblib
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---- load all artifacts once at startup ----
scaler = joblib.load("saved_models/scaler.pkl")
rf_model = joblib.load("saved_models/rf_model.pkl")
svm_model = joblib.load("saved_models/svm_model.pkl")
meta_model = joblib.load("saved_models/meta_model.pkl")
hybrid_model = load_model("saved_models/hybrid_model.keras", safe_mode=False)
with open("saved_models/feature_order.json") as f:
    feat_info = json.load(f)
ALL_FEATURES = feat_info["all_features"]
SELECTED_FEATURES = feat_info["selected_features"]
SEL_IDX = [ALL_FEATURES.index(f) for f in SELECTED_FEATURES]

FIELD_META = {
    "age":      {"label": "Age (years)",                         "type": "number", "step": "1"},
    "sex":      {"label": "Sex (1 = male, 0 = female)",           "type": "select", "options": [("1", "Male"), ("0", "Female")]},
    "cp":       {"label": "Chest pain type (0-3)",                "type": "select", "options": [("0", "Typical angina"), ("1", "Atypical angina"), ("2", "Non-anginal pain"), ("3", "Asymptomatic")]},
    "trestbps": {"label": "Resting blood pressure (mm Hg)",       "type": "number", "step": "1"},
    "chol":     {"label": "Serum cholesterol (mg/dl)",            "type": "number", "step": "1"},
    "fbs":      {"label": "Fasting blood sugar > 120 mg/dl",      "type": "select", "options": [("1", "Yes"), ("0", "No")]},
    "restecg":  {"label": "Resting ECG result (0-2)",             "type": "select", "options": [("0", "Normal"), ("1", "ST-T abnormality"), ("2", "LV hypertrophy")]},
    "thalach":  {"label": "Max heart rate achieved",              "type": "number", "step": "1"},
    "exang":    {"label": "Exercise induced angina",              "type": "select", "options": [("1", "Yes"), ("0", "No")]},
    "oldpeak":  {"label": "ST depression (oldpeak)",              "type": "number", "step": "0.1"},
    "slope":    {"label": "Slope of peak exercise ST segment",    "type": "select", "options": [("0", "Upsloping"), ("1", "Flat"), ("2", "Downsloping")]},
    "ca":       {"label": "Number of major vessels (0-3)",        "type": "number", "step": "1"},
    "thal":     {"label": "Thalassemia (0=normal,1=fixed,2=reversible)", "type": "number", "step": "1"},
}


@app.route("/", methods=["GET"])
def index():
    fields = [{"name": f, **FIELD_META[f]} for f in ALL_FEATURES]
    return render_template("index.html", fields=fields, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    fields = [{"name": f, **FIELD_META[f]} for f in ALL_FEATURES]
    try:
        raw = np.array([[float(request.form[f]) for f in ALL_FEATURES]])
    except (KeyError, ValueError):
        return render_template("index.html", fields=fields,
                                result={"error": "Please fill in every field with a valid number."})

    scaled = scaler.transform(raw)
    sel = scaled[:, SEL_IDX]

    dl_prob = float(hybrid_model.predict(sel, verbose=0).ravel()[0])
    rf_prob = float(rf_model.predict_proba(sel)[:, 1][0])
    svm_prob = float(svm_model.predict_proba(sel)[:, 1][0])
    meta_X = np.array([[dl_prob, rf_prob, svm_prob]])
    final_prob = float(meta_model.predict_proba(meta_X)[:, 1][0])
    prediction = "High risk of heart disease" if final_prob >= 0.5 else "Low risk of heart disease"

    result = {
        "prediction": prediction,
        "probability": round(final_prob * 100, 2),
        "dl_prob": round(dl_prob * 100, 2),
        "rf_prob": round(rf_prob * 100, 2),
        "svm_prob": round(svm_prob * 100, 2),
    }
    return render_template("index.html", fields=fields, result=result)


if __name__ == "__main__":
    app.run(debug=True)
