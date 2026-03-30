"""
generate_data.py
────────────────
Generates synthetic industrial power system sensor data for:
  1. Fault Detection (classification)
  2. Load Forecasting (regression)

Simulates readings from a 3-phase industrial power distribution unit:
  - Voltage (V), Current (A), Power Factor, Frequency (Hz),
    Temperature (°C), Harmonic Distortion (THD%), Load (kW)

Fault types injected:
  0 = Normal
  1 = Voltage Sag
  2 = Overcurrent
  3 = Overheating
  4 = Harmonic Distortion Fault
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_NORMAL = 1400
N_FAULT_EACH = 150   # per fault type × 4 types = 600 fault samples
TOTAL = N_NORMAL + N_FAULT_EACH * 4  # 2000 samples


def generate_timestamps(n):
    start = datetime(2024, 1, 1, 0, 0, 0)
    return [start + timedelta(minutes=15 * i) for i in range(n)]


def normal_readings(n):
    return pd.DataFrame({
        "voltage_v":        np.random.normal(415, 5, n),        # 3-phase 415V nominal
        "current_a":        np.random.normal(80, 8, n),
        "power_factor":     np.clip(np.random.normal(0.92, 0.03, n), 0.70, 1.0),
        "frequency_hz":     np.random.normal(50.0, 0.1, n),
        "temperature_c":    np.random.normal(45, 4, n),
        "thd_pct":          np.abs(np.random.normal(3.0, 0.5, n)),
        "load_kw":          np.random.normal(55, 7, n),
        "fault_label":      np.zeros(n, dtype=int),
        "fault_name":       "Normal"
    })


def voltage_sag(n):
    return pd.DataFrame({
        "voltage_v":        np.random.normal(330, 15, n),        # sag: drops to ~330V
        "current_a":        np.random.normal(100, 10, n),        # current rises
        "power_factor":     np.clip(np.random.normal(0.78, 0.05, n), 0.60, 1.0),
        "frequency_hz":     np.random.normal(49.8, 0.2, n),
        "temperature_c":    np.random.normal(52, 5, n),
        "thd_pct":          np.abs(np.random.normal(5.5, 1.0, n)),
        "load_kw":          np.random.normal(60, 8, n),
        "fault_label":      np.ones(n, dtype=int),
        "fault_name":       "Voltage Sag"
    })


def overcurrent(n):
    return pd.DataFrame({
        "voltage_v":        np.random.normal(410, 6, n),
        "current_a":        np.random.normal(165, 12, n),        # overcurrent: >150A threshold
        "power_factor":     np.clip(np.random.normal(0.85, 0.04, n), 0.70, 1.0),
        "frequency_hz":     np.random.normal(49.9, 0.15, n),
        "temperature_c":    np.random.normal(72, 8, n),          # heat rises with current
        "thd_pct":          np.abs(np.random.normal(4.0, 0.8, n)),
        "load_kw":          np.random.normal(95, 10, n),
        "fault_label":      np.full(n, 2, dtype=int),
        "fault_name":       "Overcurrent"
    })


def overheating(n):
    return pd.DataFrame({
        "voltage_v":        np.random.normal(412, 7, n),
        "current_a":        np.random.normal(90, 10, n),
        "power_factor":     np.clip(np.random.normal(0.88, 0.04, n), 0.70, 1.0),
        "frequency_hz":     np.random.normal(50.0, 0.1, n),
        "temperature_c":    np.random.normal(95, 10, n),         # critical: >85°C threshold
        "thd_pct":          np.abs(np.random.normal(3.5, 0.7, n)),
        "load_kw":          np.random.normal(62, 9, n),
        "fault_label":      np.full(n, 3, dtype=int),
        "fault_name":       "Overheating"
    })


def harmonic_distortion(n):
    return pd.DataFrame({
        "voltage_v":        np.random.normal(405, 10, n),
        "current_a":        np.random.normal(88, 9, n),
        "power_factor":     np.clip(np.random.normal(0.72, 0.06, n), 0.55, 1.0),
        "frequency_hz":     np.random.normal(50.0, 0.2, n),
        "temperature_c":    np.random.normal(58, 6, n),
        "thd_pct":          np.abs(np.random.normal(18.0, 3.0, n)),  # high THD: >8% is fault
        "load_kw":          np.random.normal(58, 8, n),
        "fault_label":      np.full(n, 4, dtype=int),
        "fault_name":       "Harmonic Distortion"
    })


def main():
    frames = [
        normal_readings(N_NORMAL),
        voltage_sag(N_FAULT_EACH),
        overcurrent(N_FAULT_EACH),
        overheating(N_FAULT_EACH),
        harmonic_distortion(N_FAULT_EACH),
    ]

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    timestamps = generate_timestamps(len(df))
    df.insert(0, "timestamp", timestamps)

    # Round for realism
    for col in ["voltage_v", "current_a", "temperature_c", "load_kw"]:
        df[col] = df[col].round(2)
    for col in ["power_factor", "frequency_hz", "thd_pct"]:
        df[col] = df[col].round(3)

    df.to_csv("power_sensor_data.csv", index=False)
    print(f"✅ Dataset generated: power_sensor_data.csv")
    print(f"   Total samples : {len(df)}")
    print(f"\n📊 Fault distribution:")
    print(df["fault_name"].value_counts().to_string())
    print(f"\n📋 Sample rows:")
    print(df[["voltage_v", "current_a", "temperature_c", "thd_pct",
              "load_kw", "fault_name"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
