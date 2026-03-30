"""
load_forecasting.py
────────────────────
Load Forecasting for Industrial Power Systems using Regression.

Syllabus concepts applied:
  - Supervised Learning (Regression)
  - Linear Regression and extensions
  - Overfitting & Underfitting demonstration
  - Bias-Variance Tradeoff
  - Hyperparameter tuning (Grid Search — Informed Search)
  - Statistical Decision Theory (MSE, MAE as loss functions)
  - Feature Engineering and Data Representation
  - Cross-Validation and Validation Sets
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


FEATURES = ["voltage_v", "current_a", "power_factor",
            "frequency_hz", "temperature_c", "thd_pct"]
TARGET = "load_kw"


# ── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    # Time-based features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]       = df["timestamp"].dt.hour
    df["day_of_week"]= df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["shift"]      = pd.cut(df["hour"],
                               bins=[-1, 7, 15, 23],
                               labels=[0, 1, 2]).astype(int)  # night/day/evening

    # Power system derived features
    df["apparent_power_kva"]  = (df["voltage_v"] * df["current_a"]) / 1000
    df["voltage_deviation"]   = np.abs(df["voltage_v"] - 415)
    df["current_x_pf"]        = df["current_a"] * df["power_factor"]
    df["temp_x_current"]      = df["temperature_c"] * df["current_a"]

    feature_cols = FEATURES + [
        "hour", "day_of_week", "is_weekend", "shift",
        "apparent_power_kva", "voltage_deviation",
        "current_x_pf", "temp_x_current"
    ]
    return df[feature_cols], feature_cols


# ── Bias-Variance Demonstration ──────────────────────────────────────────────
def demonstrate_bias_variance(X_train, y_train, X_test, y_test):
    print("\n" + "─" * 60)
    print("  📐 Bias-Variance Tradeoff Demonstration")
    print("─" * 60)
    print(f"  {'Model':<30} {'Train RMSE':>12} {'Test RMSE':>12} {'Status'}")
    print("  " + "─" * 56)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Underfitting: degree-1 polynomial (too simple)
    for deg, label in [(1, "Linear (Underfitting)"),
                       (2, "Polynomial-2 (Good fit)"),
                       (3, "Polynomial-3 (Slight overfit)")]:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("reg",  LinearRegression())
        ])
        pipe.fit(X_tr, y_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, pipe.predict(X_tr)))
        test_rmse  = np.sqrt(mean_squared_error(y_test,  pipe.predict(X_te)))
        gap = test_rmse - train_rmse
        if gap > 3:
            status = "⚠️  Overfitting"
        elif train_rmse > 8:
            status = "⚠️  Underfitting"
        else:
            status = "✅ Good"
        print(f"  {label:<30} {train_rmse:>12.4f} {test_rmse:>12.4f}  {status}")


# ── Model Training ───────────────────────────────────────────────────────────
def train_regressors(X_train, X_test, y_train, y_test, feature_cols):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    print("\n" + "─" * 60)
    print("  📊 Comparing Regression Models")
    print("─" * 60)
    print(f"  {'Model':<28} {'CV RMSE':>10} {'Test RMSE':>10} {'R²':>8}")
    print("  " + "─" * 56)

    models = {
        "Linear Regression":    LinearRegression(),
        "Ridge (L2)":           Ridge(alpha=1.0),
        "Lasso (L1)":           Lasso(alpha=0.1),
        "ElasticNet":           ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        cv_rmse = np.sqrt(-cross_val_score(
            model, X_tr, y_train, cv=5,
            scoring="neg_mean_squared_error").mean())
        model.fit(X_tr, y_train)
        preds = model.predict(X_te)
        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {"model": model, "test_rmse": test_rmse, "r2": r2}
        print(f"  {name:<28} {cv_rmse:>10.4f} {test_rmse:>10.4f} {r2:>8.4f}")

    # ── Grid Search on Random Forest ──────────────────────────────
    print("\n  🔍 Grid Search — Random Forest Regressor")
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [10, 20, None],
        "max_features": ["sqrt", "log2"],
    }
    gs = GridSearchCV(RandomForestRegressor(random_state=42),
                      param_grid, cv=5,
                      scoring="neg_mean_squared_error",
                      n_jobs=-1)
    gs.fit(X_tr, y_train)
    best = gs.best_estimator_
    best_preds = best.predict(X_te)
    best_rmse = np.sqrt(mean_squared_error(y_test, best_preds))
    best_r2   = r2_score(y_test, best_preds)
    best_mae  = mean_absolute_error(y_test, best_preds)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Test RMSE   : {best_rmse:.4f} kW")
    print(f"  Test MAE    : {best_mae:.4f} kW")
    print(f"  R² Score    : {best_r2:.4f}")

    # Feature importances
    print("\n  🔑 Feature Importances:")
    importances = pd.Series(best.feature_importances_,
                             index=feature_cols).sort_values(ascending=False).head(8)
    for feat, imp in importances.items():
        bar = "█" * int(imp * 60)
        print(f"    {feat:<28} {imp:.4f}  {bar}")

    # Sample predictions
    print("\n  📋 Sample Predictions vs Actuals (first 10):")
    print(f"  {'Actual (kW)':>12} {'Predicted (kW)':>15} {'Error (kW)':>12}")
    for actual, pred in list(zip(y_test, best_preds))[:10]:
        err = pred - actual
        print(f"  {actual:>12.2f} {pred:>15.2f} {err:>+12.2f}")

    # Save
    joblib.dump(best,   "load_model.pkl")
    joblib.dump(scaler, "load_scaler.pkl")
    print("\n💾 Model saved: load_model.pkl, load_scaler.pkl")

    return best, scaler


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "power_sensor_data.csv"

    df = pd.read_csv(data_path)

    print("=" * 60)
    print("  ⚡ INDUSTRIAL LOAD FORECASTING — Regression")
    print("=" * 60)
    print(f"  Dataset shape : {df.shape}")
    print(f"  Target stats  : mean={df[TARGET].mean():.2f} kW, "
          f"std={df[TARGET].std():.2f} kW, "
          f"range=[{df[TARGET].min():.1f}, {df[TARGET].max():.1f}]")

    X, feature_cols = engineer_features(df)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    demonstrate_bias_variance(X_train, y_train, X_test, y_test)
    train_regressors(X_train, X_test, y_train, y_test, feature_cols)
