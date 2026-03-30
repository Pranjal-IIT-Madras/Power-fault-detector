"""
main.py
────────
Entry point for the Industrial Power Fault Detection & Load Forecasting System.
Run the full pipeline or individual modules.

Usage:
    python main.py                  # full pipeline
    python main.py --task fault     # fault detection only
    python main.py --task load      # load forecasting only
    python main.py --task generate  # generate data only
"""

import argparse
import os
import sys


def banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║   ⚡ Industrial Power Fault Detection & Load Forecasting  ║
║      AI-Powered Intelligent Agent System                  ║
║      BYOP Project — Machine Learning / AI Course          ║
╚══════════════════════════════════════════════════════════╝
    """)


def run_generate():
    print("\n📂 Step 1: Generating synthetic power sensor dataset...")
    import generate_data
    generate_data.main()


def run_fault():
    print("\n🔴 Step 2: Running Fault Detection (Classification)...")
    import fault_detection
    X, y, feature_cols = fault_detection.load_data("power_sensor_data.csv")
    model, scaler = fault_detection.train_and_evaluate(X, y, feature_cols)
    fault_detection.run_agent_demo(model, scaler)


def run_load():
    print("\n📈 Step 3: Running Load Forecasting (Regression)...")
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    import load_forecasting as lf

    df = pd.read_csv("power_sensor_data.csv")
    X, feature_cols = lf.engineer_features(df)
    y = df[lf.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"\n  Dataset shape : {df.shape}")
    print(f"  Target stats  : mean={y.mean():.2f} kW, std={y.std():.2f} kW")

    lf.demonstrate_bias_variance(X_train, y_train, X_test, y_test)
    lf.train_regressors(X_train, X_test, y_train, y_test, feature_cols)


def main():
    banner()
    parser = argparse.ArgumentParser(
        description="Industrial Power AI System"
    )
    parser.add_argument("--task", choices=["generate", "fault", "load", "all"],
                        default="all",
                        help="Which task to run (default: all)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if args.task in ("generate", "all"):
        run_generate()
    if args.task in ("fault", "all"):
        if not os.path.exists("power_sensor_data.csv"):
            run_generate()
        run_fault()
    if args.task in ("load", "all"):
        if not os.path.exists("power_sensor_data.csv"):
            run_generate()
        run_load()

    print("\n✅ All tasks complete.")


if __name__ == "__main__":
    main()
