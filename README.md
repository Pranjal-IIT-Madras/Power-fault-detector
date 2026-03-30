# ⚡ Industrial Power Fault Detection & Load Forecasting

An AI-powered system that monitors industrial power grid sensor readings to **detect electrical faults** (classification) and **forecast energy load** (regression).
It also implements a **rational intelligent agent** that continuously analyzes system conditions and raises alerts.

---

## 📌 Problem Statement

Industrial power systems face two critical challenges:

1. **Fault Detection**
   Faults like voltage sags, overcurrent, overheating, and harmonic distortion can damage equipment and cause downtime.

2. **Load Forecasting**
   Poor load prediction leads to energy wastage and higher operational costs.

This project uses **supervised machine learning** to solve both problems using sensor data.

---

## 🧠 Concepts Applied

| Concept                | Implementation                              |
| ---------------------- | ------------------------------------------- |
| Intelligent Agent      | `PowerFaultAgent` (Perceive → Decide → Act) |
| Classification         | Fault detection                             |
| Regression             | Load forecasting                            |
| Grid Search            | Hyperparameter tuning                       |
| Bias-Variance Tradeoff | Polynomial regression demo                  |
| Cross-validation       | Model evaluation                            |
| Feature Engineering    | Derived electrical features                 |

---

## 🗂️ Project Structure

```
Power-fault-detector/
│
├── main.py                # Entry point (runs full pipeline or tasks)
├── generate_data.py       # Synthetic dataset generator
├── fault_detection.py     # Fault classification + intelligent agent
├── load_forecasting.py    # Load prediction + bias-variance demo
├── categorize.py          # (Extra) NLP expense categorizer
├── evaluate.py            # (Extra) evaluation for categorizer
├── requirements.txt       # Dependencies
├── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Pranjal-IIT-Madras/Power-fault-detector.git
cd Power-fault-detector
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### ▶️ Run Full Pipeline (Recommended)

```bash
python main.py
```

This will:

1. Generate synthetic dataset (if not present)
2. Train fault detection model
3. Run intelligent agent demo
4. Train load forecasting model
5. Show bias-variance analysis

---

### ▶️ Run Individual Tasks

```bash
python main.py --task generate   # Generate dataset
python main.py --task fault      # Fault detection
python main.py --task load       # Load forecasting
```

---

## 📊 Sensor Features

| Feature       | Description             |
| ------------- | ----------------------- |
| voltage_v     | Voltage (V)             |
| current_a     | Current (A)             |
| power_factor  | Power efficiency        |
| frequency_hz  | System frequency        |
| temperature_c | Equipment temperature   |
| thd_pct       | Harmonic distortion (%) |
| load_kw       | Power consumption (kW)  |

---

## 🔴 Fault Classes

| Code | Fault               |
| ---- | ------------------- |
| 0    | Normal              |
| 1    | Voltage Sag         |
| 2    | Overcurrent         |
| 3    | Overheating         |
| 4    | Harmonic Distortion |

---

## 🤖 Intelligent Agent

The system includes a **rational intelligent agent**:

```python
from fault_detection import PowerFaultAgent
import joblib

model  = joblib.load("fault_model.pkl")
scaler = joblib.load("fault_scaler.pkl")

agent = PowerFaultAgent(model, scaler)

reading = {
    "voltage_v": 330,
    "current_a": 100,
    "power_factor": 0.80,
    "frequency_hz": 49.8,
    "temperature_c": 50,
    "thd_pct": 6
}

agent.act(reading)
```

### 🔄 Agent Workflow

* **Perceive** → Reads sensor data
* **Decide** → Predicts fault using ML model
* **Act** → Displays alert with severity & confidence

---

## 📈 Models Used

### Classification (Fault Detection)

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)

### Regression (Load Forecasting)

* Linear Regression
* Ridge / Lasso / ElasticNet
* Random Forest Regressor
* Gradient Boosting Regressor

---

## 📊 Output Highlights

* Fault classification with confidence scores
* Confusion matrix & F1-score
* Load prediction with RMSE, MAE, R²
* Bias-variance tradeoff demonstration

---

## ⚠️ Notes

* Dataset is **synthetic but realistic (~2000 samples)**
* Models are trained dynamically (no pre-trained dependency required)
* `categorize.py` and `evaluate.py` are **extra modules (not part of main pipeline)**

---

## 🏫 Course Context

This project demonstrates:

* Intelligent Agents
* Supervised Learning
* Feature Engineering
* Model Evaluation
* Bias-Variance Tradeoff

Aligned with **Machine Learning / AI coursework (CO2–CO5)**.

---

## 📝 License

MIT License — free to use and modify.
