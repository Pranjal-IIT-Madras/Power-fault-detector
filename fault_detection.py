"""
fault_detection.py
──────────────────
Fault Detection in Industrial Power Systems using Supervised Classification.

Syllabus concepts applied:
  - Supervised Learning (Classification)
  - Intelligent Agent (rational agent monitoring sensors)
  - Probability & Statistics (class distributions, confusion matrix)
  - Overfitting/Underfitting, Bias-Variance Tradeoff
  - Hyperparameter Tuning via Grid Search (Informed Search Strategy)
  - Validation Sets, Cross-validation
  - Estimators: Random Forest, SVM, Logistic Regression
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.pipeline import Pipeline

FEATURES = ["voltage_v", "current_a", "power_factor",
            "frequency_hz", "temperature_c", "thd_pct"]
TARGET = "fault_label"
LABEL_MAP = {0: "Normal", 1: "Voltage Sag", 2: "Overcurrent",
             3: "Overheating", 4: "Harmonic Distortion"}


# ── 1. Rational Agent Interface ─────────────────────────────────────────────
class PowerFaultAgent:
    """
    A rational intelligent agent that:
      - Perceives: sensor readings from the environment (power grid)
      - Decides : classifies the system state into Normal or a fault type
      - Acts    : raises an alert with severity level
    Implements the Agent concept from the syllabus (Intelligent Agents & Environments).
    """

    SEVERITY = {0: "✅ NORMAL", 1: "⚠️  WARNING", 2: "🔴 CRITICAL",
                3: "🔴 CRITICAL", 4: "⚠️  WARNING"}

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def perceive(self, reading: dict) -> np.ndarray:
        """Convert raw sensor dict to feature vector."""
        return np.array([[reading[f] for f in FEATURES]])

    def decide(self, reading: dict) -> dict:
        """Classify a single sensor reading and return a structured alert."""
        X = self.perceive(reading)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = proba.max()
        return {
            "fault_code": int(pred),
            "fault_name": LABEL_MAP[pred],
            "severity": self.SEVERITY[pred],
            "confidence": round(float(confidence), 4),
            "probabilities": {LABEL_MAP[i]: round(float(p), 4)
                              for i, p in enumerate(proba)}
        }

    def act(self, reading: dict):
        """Full agent loop: perceive → decide → display alert."""
        result = self.decide(reading)
        print("\n" + "─" * 55)
        print("  ⚡ POWER FAULT AGENT — SENSOR ALERT")
        print("─" * 55)
        for feat in FEATURES:
            print(f"  {feat:<20}: {reading[feat]}")
        print("─" * 55)
        print(f"  Detected       : {result['fault_name']}")
        print(f"  Severity       : {result['severity']}")
        print(f"  Confidence     : {result['confidence']:.2%}")
        print("─" * 55)
        print("  Class Probabilities:")
        for name, prob in result["probabilities"].items():
            bar = "█" * int(prob * 30)
            print(f"    {name:<25} {prob:.2%}  {bar}")
        print("─" * 55)
        return result


# ── 2. Data Loading & Feature Engineering ───────────────────────────────────
def load_data(path="power_sensor_data.csv"):
    df = pd.read_csv(path)

    # Feature engineering: derived features
    df["apparent_power_kva"] = (df["voltage_v"] * df["current_a"]) / 1000
    df["reactive_power_kvar"] = df["apparent_power_kva"] * np.sqrt(
        np.clip(1 - df["power_factor"] ** 2, 0, 1))
    df["voltage_deviation"] = np.abs(df["voltage_v"] - 415) / 415 * 100

    feature_cols = FEATURES + ["apparent_power_kva",
                                "reactive_power_kvar", "voltage_deviation"]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y, feature_cols


# ── 3. Model Training & Hyperparameter Tuning ───────────────────────────────
def train_and_evaluate(X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("=" * 60)
    print("  ⚡ INDUSTRIAL POWER FAULT DETECTION")
    print("  Comparing Classifiers + Hyperparameter Grid Search")
    print("=" * 60)

    # ── Multiple estimators (Bias-Variance comparison) ───────────
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "SVM (RBF)":           SVC(probability=True, random_state=42),
    }

    results = {}
    for name, clf in estimators.items():
        cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5,
                                    scoring="f1_weighted")
        clf.fit(X_train_s, y_train)
        test_f1 = f1_score(clf.predict(X_test_s), y_test, average="weighted")
        results[name] = {"cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
                         "test_f1": test_f1}
        print(f"\n  {name}")
        print(f"    CV F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    Test F1       : {test_f1:.4f}")

    # ── Grid Search on best model (Random Forest) ────────────────
    print("\n" + "─" * 60)
    print("  🔍 Grid Search Hyperparameter Tuning — Random Forest")
    print("─" * 60)
    param_grid = {
        "n_estimators":      [50, 100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train_s, y_train)
    best_clf = grid_search.best_estimator_
    print(f"  Best params   : {grid_search.best_params_}")
    print(f"  Best CV F1    : {grid_search.best_score_:.4f}")

    # ── Final evaluation ─────────────────────────────────────────
    y_pred = best_clf.predict(X_test_s)
    print("\n" + "─" * 60)
    print("  📊 FINAL MODEL PERFORMANCE (Test Set)")
    print("─" * 60)
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\n  Classification Report:")
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df.to_string())

    # ── Feature Importances ──────────────────────────────────────
    print("\n  🔑 Feature Importances:")
    importances = pd.Series(best_clf.feature_importances_,
                             index=feature_cols).sort_values(ascending=False)
    for feat, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"    {feat:<28} {imp:.4f}  {bar}")

    # Save model
    joblib.dump(best_clf, "fault_model.pkl")
    joblib.dump(scaler,   "fault_scaler.pkl")
    print("\n💾 Model saved: fault_model.pkl, fault_scaler.pkl")

    return best_clf, scaler


# ── 4. Demo Agent ────────────────────────────────────────────────────────────
def run_agent_demo(model, scaler):
    agent = PowerFaultAgent(model, scaler)
    print("\n" + "=" * 60)
    print("  🤖 INTELLIGENT AGENT DEMO — Live Sensor Readings")
    print("=" * 60)

    test_readings = [
        {"label": "Normal Operation",
         "voltage_v": 415.2, "current_a": 79.5, "power_factor": 0.921,
         "frequency_hz": 50.01, "temperature_c": 44.8, "thd_pct": 2.9},
        {"label": "Voltage Sag Event",
         "voltage_v": 328.0, "current_a": 105.0, "power_factor": 0.76,
         "frequency_hz": 49.75, "temperature_c": 53.0, "thd_pct": 5.8},
        {"label": "Overcurrent Fault",
         "voltage_v": 411.0, "current_a": 172.0, "power_factor": 0.84,
         "frequency_hz": 49.9, "temperature_c": 78.0, "thd_pct": 4.1},
        {"label": "Overheating Fault",
         "voltage_v": 413.0, "current_a": 88.0, "power_factor": 0.87,
         "frequency_hz": 50.0, "temperature_c": 98.0, "thd_pct": 3.4},
        {"label": "Harmonic Distortion",
         "voltage_v": 402.0, "current_a": 91.0, "power_factor": 0.70,
         "frequency_hz": 50.1, "temperature_c": 60.0, "thd_pct": 19.5},
    ]

    for reading in test_readings:
        label = reading.pop("label")
        print(f"\n  📡 Scenario: {label}")
        agent.act(reading)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "power_sensor_data.csv"

    X, y, feature_cols = load_data(data_path)
    model, scaler = train_and_evaluate(X, y, feature_cols)
    run_agent_demo(model, scaler)
