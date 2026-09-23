"""
train.py
--------
End-to-end pipeline:
  1. Download + clean dataset            (data_loader.py)
  2. Train/test split + scaling + SMOTE balancing
  3. Hybrid feature optimization (RFE + GA)   (feature_optimization.py)
  4. Train BASELINE model (plain ANN, all features) - "previous work" style
  5. Train PROPOSED hybrid CNN-BiLSTM-Attention model on optimized features
  6. Train a stacking ensemble (hybrid DL + RF + SVM -> Gradient Boosting)
  7. Evaluate everything, print + save a comparison table and plots
  8. Save every artifact the Flask website needs to make live predictions

Run:
    python train.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping

from data_loader import load_and_prepare
from feature_optimization import optimize_features
from models import (build_baseline_ann, build_hybrid_cnn_bilstm_attention,
                     build_stacked_ensemble)

RANDOM_STATE = 42


def evaluate(y_true, y_pred, y_prob, name):
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }
    print(f"\n[{name}] " + " | ".join(f"{k}={v}" for k, v in metrics.items() if k != "model"))
    return metrics


def main():
    print("=" * 70)
    print("STEP 1: Loading dataset")
    print("=" * 70)
    df = load_and_prepare()
    feature_names = [c for c in df.columns if c != "target"]
    X_all = df[feature_names].values
    y_all = df["target"].values

    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )

    # scale on ALL features first (needed for both baseline + feature selection)
    scaler_full = StandardScaler().fit(X_train_full)
    X_train_full_s = scaler_full.transform(X_train_full)
    X_test_full_s = scaler_full.transform(X_test_full)

    print("\n" + "=" * 70)
    print("STEP 2: Balancing classes with SMOTE (train set only)")
    print("=" * 70)
    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_full_s, y_train)
    print(f"Train size before/after SMOTE: {len(y_train)} -> {len(y_train_bal)}")

    print("\n" + "=" * 70)
    print("STEP 3: BASELINE model (plain ANN on ALL features) - previous-work style")
    print("=" * 70)
    baseline = build_baseline_ann(X_train_bal.shape[1])
    baseline.fit(X_train_bal, y_train_bal, validation_split=0.15,
                 epochs=100, batch_size=16, verbose=0)
    base_prob = baseline.predict(X_test_full_s, verbose=0).ravel()
    base_pred = (base_prob >= 0.5).astype(int)
    baseline_metrics = evaluate(y_test, base_pred, base_prob, "Baseline ANN (all features)")

    print("\n" + "=" * 70)
    print("STEP 4: Hybrid feature optimization (RFE + Genetic Algorithm)")
    print("=" * 70)
    selected = optimize_features(X_train_bal, y_train_bal, feature_names)
    idx = [feature_names.index(f) for f in selected]

    X_train_sel = X_train_bal[:, idx]
    X_test_sel = X_test_full_s[:, idx]

    print("\n" + "=" * 70)
    print("STEP 5: PROPOSED hybrid CNN-BiLSTM-Attention model (optimized features)")
    print("=" * 70)
    hybrid = build_hybrid_cnn_bilstm_attention(len(idx))
    early_stop = EarlyStopping(monitor="val_auc", mode="max", patience=15,
                                restore_best_weights=True)
    history = hybrid.fit(X_train_sel, y_train_bal, validation_split=0.15,
                          epochs=200, batch_size=16, callbacks=[early_stop], verbose=0)
    hybrid_prob = hybrid.predict(X_test_sel, verbose=0).ravel()
    hybrid_pred = (hybrid_prob >= 0.5).astype(int)
    hybrid_metrics = evaluate(y_test, hybrid_pred, hybrid_prob,
                               "Proposed Hybrid CNN-BiLSTM-Attention (optimized features)")

    print("\n" + "=" * 70)
    print("STEP 6: Stacking ensemble (Hybrid-DL + RandomForest + SVM -> GBM meta-learner)")
    print("=" * 70)
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE).fit(X_train_sel, y_train_bal)
    svm = SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE).fit(X_train_sel, y_train_bal)

    rf_train_prob = rf.predict_proba(X_train_sel)[:, 1]
    svm_train_prob = svm.predict_proba(X_train_sel)[:, 1]
    dl_train_prob = hybrid.predict(X_train_sel, verbose=0).ravel()
    meta_X_train = np.column_stack([dl_train_prob, rf_train_prob, svm_train_prob])

    rf_test_prob = rf.predict_proba(X_test_sel)[:, 1]
    svm_test_prob = svm.predict_proba(X_test_sel)[:, 1]
    meta_X_test = np.column_stack([hybrid_prob, rf_test_prob, svm_test_prob])

    meta_model = build_stacked_ensemble().fit(meta_X_train, y_train_bal)
    ensemble_prob = meta_model.predict_proba(meta_X_test)[:, 1]
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    ensemble_metrics = evaluate(y_test, ensemble_pred, ensemble_prob,
                                 "Final Proposed System (Stacked Ensemble)")

    print("\n" + "=" * 70)
    print("STEP 7: Results comparison + plots")
    print("=" * 70)
    results = pd.DataFrame([baseline_metrics, hybrid_metrics, ensemble_metrics])
    results.to_csv("saved_models/results_comparison.csv", index=False)
    print(results.to_string(index=False))

    plt.figure(figsize=(7, 5))
    sns.barplot(data=results, x="model", y="accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.title("Accuracy comparison: baseline vs proposed hybrid system")
    plt.tight_layout()
    plt.savefig("saved_models/accuracy_comparison.png", dpi=150)

    cm = confusion_matrix(y_test, ensemble_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion matrix - Final Proposed System")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("saved_models/confusion_matrix.png", dpi=150)

    print("\n" + "=" * 70)
    print("STEP 8: Saving artifacts for the website")
    print("=" * 70)
    hybrid.save("saved_models/hybrid_model.keras")
    joblib.dump(scaler_full, "saved_models/scaler.pkl")
    joblib.dump(rf, "saved_models/rf_model.pkl")
    joblib.dump(svm, "saved_models/svm_model.pkl")
    joblib.dump(meta_model, "saved_models/meta_model.pkl")
    with open("saved_models/feature_order.json", "w") as f:
        json.dump({"all_features": feature_names, "selected_features": selected}, f, indent=2)

    print("\nAll done! Artifacts saved in saved_models/.")
    print("Now run:  python app.py   to launch the prediction website.")


if __name__ == "__main__":
    main()
