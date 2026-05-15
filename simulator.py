"""
simulator.py
------------
Scenario simulation engine.

For every (product, region, ship_mode) combination, the engine:
  1. Looks up the CURRENT factory assignment
  2. Simulates assignment to ALL other factories
  3. Predicts lead time for each scenario using the trained model
  4. Computes lead-time reduction (%), profit sensitivity, confidence score
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

from data_loader import FACTORY_COORDS, REGION_CENTROIDS, ALL_FACTORIES


# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _encode_predict(model, encoders,
                    region, ship_mode, factory, division, distance_km, units):
    """Encode one row and predict lead time, guarding unseen labels."""
    row = pd.DataFrame([{
        "Region": region, "Ship Mode": ship_mode, "Factory": factory,
        "Division": division, "Distance_km": distance_km, "Units": units,
    }])
    cat_cols = ["Region", "Ship Mode", "Factory", "Division"]
    for col in cat_cols:
        le = encoders[col]
        v  = str(row[col].iloc[0])
        row[col] = le.transform([v])[0] if v in le.classes_ else -1
    row[["Distance_km", "Units"]] = encoders["scaler"].transform(
        row[["Distance_km", "Units"]]
    )
    return float(model.predict(row)[0])


# ────────────────────────────────────────────────
#  MAIN SIMULATION
# ────────────────────────────────────────────────

def simulate_product(df: pd.DataFrame, model, encoders,
                     product_name: str,
                     region: str = None,
                     ship_mode: str = None,
                     units: int = 3) -> pd.DataFrame:
    """
    Simulate the given product assigned to EACH factory.

    Parameters
    ----------
    product_name  : one of the 15 products in the dataset
    region        : if None, uses the most common region for that product
    ship_mode     : if None, uses the most common ship mode for that product
    units         : number of units to use for prediction (default 3)

    Returns
    -------
    DataFrame with one row per factory, columns:
      Factory, Is_Current, Distance_km, Predicted_Lead_Time,
      Current_Lead_Time, Lead_Time_Delta, Lead_Time_Reduction_Pct,
      Avg_Profit, Profit_Margin, Confidence_Score
    """
    subset = df[df["Product Name"] == product_name].copy()
    if subset.empty:
        return pd.DataFrame()

    # Defaults for optional params
    if region is None:
        region = subset["Region"].mode()[0]
    if ship_mode is None:
        ship_mode = subset["Ship Mode"].mode()[0]

    division   = subset["Division"].mode()[0]
    current_factory = subset["Factory"].mode()[0]

    # Region centroid
    r_lat, r_lon = REGION_CENTROIDS.get(region, (39.0, -95.0))

    # Current baseline
    current_lead_time = _encode_predict(
        model, encoders, region, ship_mode, current_factory, division,
        _haversine_km(
            FACTORY_COORDS[current_factory]["lat"],
            FACTORY_COORDS[current_factory]["lon"],
            r_lat, r_lon
        ), units
    )

    rows = []
    for factory in ALL_FACTORIES:
        dist = _haversine_km(
            FACTORY_COORDS[factory]["lat"],
            FACTORY_COORDS[factory]["lon"],
            r_lat, r_lon
        )
        pred = _encode_predict(
            model, encoders, region, ship_mode, factory, division, dist, units
        )
        pred = max(0, pred)

        delta = current_lead_time - pred
        reduction_pct = (delta / current_lead_time * 100) if current_lead_time > 0 else 0.0

        # Avg profit from historical data for this factory (or global avg)
        fac_subset  = subset[subset["Factory"] == factory]
        hist_subset = subset if fac_subset.empty else fac_subset
        avg_profit  = hist_subset["Gross Profit"].mean()
        avg_margin  = hist_subset["Profit_Margin"].mean()

        # Confidence score: based on sample size (0–1)
        sample = len(subset[subset["Region"] == region])
        confidence = min(1.0, sample / 50)

        rows.append({
            "Factory":                 factory,
            "Is_Current":             factory == current_factory,
            "Distance_km":            round(dist, 1),
            "Predicted_Lead_Time":    round(pred, 2),
            "Current_Lead_Time":      round(current_lead_time, 2),
            "Lead_Time_Delta":        round(delta, 2),
            "Lead_Time_Reduction_Pct": round(reduction_pct, 1),
            "Avg_Profit":             round(avg_profit, 2),
            "Profit_Margin_Pct":      round(avg_margin * 100, 1),
            "Confidence_Score":       round(confidence, 2),
        })

    return pd.DataFrame(rows).sort_values("Predicted_Lead_Time")


def simulate_all_products(df: pd.DataFrame, model, encoders) -> pd.DataFrame:
    """
    Run simulation for every unique (product, region, ship_mode) combo.
    Returns a comprehensive scenario DataFrame used by the Recommendation engine.
    """
    records = []
    combos = (
        df.groupby(["Product Name", "Region", "Ship Mode"])
        .agg(
            Units    = ("Units", "median"),
            Avg_Profit = ("Gross Profit", "mean"),
        )
        .reset_index()
    )

    for _, row in combos.iterrows():
        sim = simulate_product(
            df, model, encoders,
            product_name=row["Product Name"],
            region=row["Region"],
            ship_mode=row["Ship Mode"],
            units=int(row["Units"]),
        )
        if sim.empty:
            continue
        sim["Product Name"] = row["Product Name"]
        sim["Region"]       = row["Region"]
        sim["Ship Mode"]    = row["Ship Mode"]
        records.append(sim)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)
