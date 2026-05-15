"""
model_trainer.py
----------------
Trains three regression models to predict Lead Time:
  1. Linear Regression (baseline)
  2. Random Forest Regressor
  3. Gradient Boosting Regressor

Evaluates with RMSE, MAE, R².
Saves the best model + encoders to disk and returns them.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "best_model.pkl")
ENCODER_PATH  = os.path.join(os.path.dirname(__file__), "encoders.pkl")


# ────────────────────────────────────────────────
#  EVALUATION HELPER
# ────────────────────────────────────────────────

def _evaluate(name: str, model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"  [{name}]  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
    return {"name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2, "preds": preds}


# ────────────────────────────────────────────────
#  MAIN TRAINER
# ────────────────────────────────────────────────

def train_models(X: pd.DataFrame, y: pd.Series, encoders: dict,
                 test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    Train all three models, select the best (lowest RMSE), persist to disk.

    Returns
    -------
    results : dict  {
        "best_model":   fitted sklearn estimator,
        "best_name":    str,
        "metrics":      list[dict],
        "X_test":       pd.DataFrame,
        "y_test":       pd.Series,
        "feature_importances": dict (if tree-based),
        "feature_names": list
    }
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "Linear Regression":     LinearRegression(),
        "Random Forest":         RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1),
        "Gradient Boosting":     GradientBoostingRegressor(n_estimators=150, random_state=random_state),
    }

    print("\n🔧 Training models …")
    metrics = []
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        metrics.append(_evaluate(name, mdl, X_test, y_test))

    # Select best by RMSE
    best = min(metrics, key=lambda d: d["rmse"])
    print(f"\n✅ Best model: {best['name']}  (RMSE={best['rmse']:.2f})")

    # Persist
    joblib.dump(best["model"], MODEL_PATH)
    joblib.dump(encoders,      ENCODER_PATH)
    print(f"   Saved → {MODEL_PATH}")

    # Feature importances (tree models only)
    fi = {}
    if hasattr(best["model"], "feature_importances_"):
        fi = dict(zip(X.columns, best["model"].feature_importances_))

    return {
        "best_model":           best["model"],
        "best_name":            best["name"],
        "metrics":              metrics,
        "X_test":               X_test,
        "y_test":               y_test,
        "feature_importances":  fi,
        "feature_names":        list(X.columns),
    }


def load_model_and_encoders():
    """Load persisted model + encoders from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run `python run_analysis.py` first to train models."
        )
    model    = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders


def predict_lead_time(model, encoders, region: str, ship_mode: str,
                      factory: str, division: str,
                      distance_km: float, units: int) -> float:
    """
    Predict lead time for a single observation.
    All categorical inputs are raw strings; encoding is applied internally.
    """
    import pandas as pd

    row = pd.DataFrame([{
        "Region":      region,
        "Ship Mode":   ship_mode,
        "Factory":     factory,
        "Division":    division,
        "Distance_km": distance_km,
        "Units":       units,
    }])

    cat_cols = ["Region", "Ship Mode", "Factory", "Division"]
    for col in cat_cols:
        le = encoders[col]
        val = row[col].astype(str).iloc[0]
        if val in le.classes_:
            row[col] = le.transform([val])[0]
        else:
            row[col] = -1   # unseen label → handled gracefully

    scaler = encoders["scaler"]
    row[["Distance_km", "Units"]] = scaler.transform(row[["Distance_km", "Units"]])

    return float(model.predict(row)[0])
