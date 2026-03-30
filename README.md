# ⚡ Industrial Power Fault Detection & Load Forecasting

An AI-powered system that monitors industrial power grid sensor readings to **detect electrical faults** (classification) and **forecast energy load** (regression) — acting as a rational intelligent agent to prevent equipment damage and optimize energy use.

---

## 📌 Problem Statement

Industrial power systems face two critical challenges:

1. **Fault Detection** — Undetected faults like voltage sags, overcurrents, and overheating cause equipment damage, production downtime, and safety hazards costing industries lakhs to crores per incident.
2. **Load Forecasting** — Poor load prediction leads to energy wastage, peak demand penalties, and inefficient power procurement.

This project applies supervised machine learning to solve both problems using real-time sensor data.

---

## 🧠 Syllabus Concepts Applied

| Concept | Where Applied |
|---|---|
| Intelligent Agents & Rationality | `PowerFaultAgent` class — perceive, decide, act loop |
| Supervised Learning — Classification | Fault detection (5 classes) |
| Supervised Learning — Regression | Load forecasting (kW prediction) |
| Search Strategies (Informed) | Grid Search for hyperparameter tuning |
| Probability & Statistics | Class distributions, confidence scores |
| Overfitting & Underfitting | Polynomial degree comparison in regression |
| Bias-Variance Tradeoff | Demonstrated with degree-1 vs degree-3 polynomial |
| Estimators, Validation Sets | Cross-validation, train/test split |
| Feature Learning | Derived features (apparent power, reactive power, etc.) |

---

## 🗂️ Project Structure

```
power-fault-detector/
│
├── main.py               # Entry point — run full pipeline or individual tasks
├── generate_data.py      # Synthetic sensor data generator
├── fault_detection.py    # Classification: fault detection + intelligent agent
├── load_forecasting.py   # Regression: load forecasting + bias-variance demo
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/power-fault-detector.git
cd power-fault-detector
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Run Full Pipeline (recommended)

```bash
python main.py
```

This runs all three steps:
1. Generates 2000 synthetic sensor readings
2. Trains fault detection classifier + runs agent demo
3. Trains load forecasting regressor + shows bias-variance analysis

### Run Individual Modules

```bash
python main.py --task generate   # Generate dataset only
python main.py --task fault      # Fault detection only
python main.py --task load       # Load forecasting only
```

### Run Modules Directly

```bash
python generate_data.py
python fault_detection.py power_sensor_data.csv
python load_forecasting.py power_sensor_data.csv
```

---

## 📊 Sensor Features

The system monitors 6 real-time power sensor readings:

| Feature | Unit | Normal Range |
|---|---|---|
| Voltage | V | 410–420 V |
| Current | A | 60–100 A |
| Power Factor | — | 0.88–0.97 |
| Frequency | Hz | 49.9–50.1 Hz |
| Temperature | °C | 38–55 °C |
| Total Harmonic Distortion (THD) | % | 2–5% |

---

## 🔴 Fault Classes Detected

| Code | Fault Type | Key Indicator |
|---|---|---|
| 0 | Normal | All readings within spec |
| 1 | Voltage Sag | Voltage drops to ~330V |
| 2 | Overcurrent | Current exceeds 150A |
| 3 | Overheating | Temperature exceeds 85°C |
| 4 | Harmonic Distortion | THD exceeds 8% |

---

## 📈 Expected Results

**Fault Detection (Classification):**
- Accuracy: ~96–98%
- Weighted F1-Score: ~96–98%
- Models compared: Logistic Regression, Random Forest, Gradient Boosting, SVM

**Load Forecasting (Regression):**
- Best RMSE: ~2–4 kW
- R² Score: ~0.96–0.99
- Models compared: Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting

---

## 🤖 Intelligent Agent

The `PowerFaultAgent` class implements the agent architecture from the AI syllabus:

```python
from fault_detection import PowerFaultAgent
import joblib

model  = joblib.load("fault_model.pkl")
scaler = joblib.load("fault_scaler.pkl")
agent  = PowerFaultAgent(model, scaler)

reading = {
    "voltage_v": 328.0, "current_a": 105.0, "power_factor": 0.76,
    "frequency_hz": 49.75, "temperature_c": 53.0, "thd_pct": 5.8
}
agent.act(reading)
```

Output:
```
  Detected       : Voltage Sag
  Severity       : ⚠️  WARNING
  Confidence     : 97.32%
```

---

## 🏫 Course Context

Built as a BYOP (Bring Your Own Project) submission for the **Machine Learning / AI** course. Covers CO2–CO5 outcomes including problem solving, supervised learning, statistical decision theory, and machine learning applications in an industrial engineering context.

---

## 📝 License

MIT License — free to use and modify.
