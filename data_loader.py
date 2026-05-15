"""
data_loader.py
--------------
Loads and pre-processes the Nassau Candy Distributor dataset.
- Parses mixed date formats
- Engineers Lead Time (days)
- Maps products → factories → coordinates
- Computes Haversine distance (factory → customer approximation via region centroid)
- Encodes categoricals, normalises numerics, removes outliers
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ────────────────────────────────────────────────
#  STATIC REFERENCE DATA (from tech-doc)
# ────────────────────────────────────────────────

FACTORY_COORDS = {
    "Lot's O' Nuts":    {"lat": 32.881893, "lon": -111.768036},
    "Wicked Choccy's":  {"lat": 32.076176, "lon": -81.088371},
    "Sugar Shack":      {"lat": 48.11914,  "lon": -96.18115},
    "Secret Factory":   {"lat": 41.446333, "lon": -90.565487},
    "The Other Factory":{"lat": 35.1175,   "lon": -89.971107},
}

PRODUCT_FACTORY_MAP = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

# Approximate region centroids (lat, lon) for distance calculation
REGION_CENTROIDS = {
    "Atlantic":  (39.5, -75.5),
    "Gulf":      (30.0, -85.0),
    "Interior":  (39.0, -95.0),
    "Pacific":   (37.0, -120.0),
}

ALL_FACTORIES = list(FACTORY_COORDS.keys())


# ────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _parse_date(series: pd.Series) -> pd.Series:
    """Robustly parse mixed date formats like dd/mm/yyyy, dd-mm-yyyy, m/d/yyyy."""
    parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    mask = parsed.isna()
    if mask.any():
        parsed[mask] = pd.to_datetime(series[mask], format="%m/%d/%Y", errors="coerce")
    return parsed


def _remove_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 3.0) -> pd.DataFrame:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - factor * IQR) & (df[col] <= Q3 + factor * IQR)]


# ────────────────────────────────────────────────
#  MAIN LOADER
# ────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV and return cleaned, feature-rich DataFrame."""
    df = pd.read_csv(filepath)

    # ── Date parsing ──────────────────────────────
    df["Order Date"] = _parse_date(df["Order Date"])
    df["Ship Date"]  = _parse_date(df["Ship Date"])

    # Drop rows where dates are still unparseable
    df = df.dropna(subset=["Order Date", "Ship Date"])

    # ── Lead Time ─────────────────────────────────
    df["Lead_Time_Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    # Keep only non-negative lead times; upper bound removed (dataset has future Ship Dates)
    df = df[df["Lead_Time_Days"] >= 0]

    # ── Factory mapping ───────────────────────────
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY_MAP)
    df = df.dropna(subset=["Factory"])

    df["Factory_Lat"] = df["Factory"].map(lambda f: FACTORY_COORDS[f]["lat"])
    df["Factory_Lon"] = df["Factory"].map(lambda f: FACTORY_COORDS[f]["lon"])

    # ── Region centroid distance ───────────────────
    df["Region_Lat"] = df["Region"].map(lambda r: REGION_CENTROIDS.get(r, (39.0, -95.0))[0])
    df["Region_Lon"] = df["Region"].map(lambda r: REGION_CENTROIDS.get(r, (39.0, -95.0))[1])

    df["Distance_km"] = df.apply(
        lambda row: haversine_km(
            row["Factory_Lat"], row["Factory_Lon"],
            row["Region_Lat"],  row["Region_Lon"]
        ), axis=1
    )

    # ── Financial metrics ─────────────────────────
    df["Profit_Margin"] = (df["Gross Profit"] / df["Sales"].replace(0, np.nan)).fillna(0)

    # ── Remove extreme outliers ───────────────────
    for col in ["Lead_Time_Days", "Sales", "Gross Profit", "Distance_km"]:
        df = _remove_outliers_iqr(df, col)

    df = df.reset_index(drop=True)
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Build encoded feature matrix X and target y for ML.
    Returns: X (DataFrame), y (Series), feature_names (list), encoders (dict)
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    feature_cols = ["Region", "Ship Mode", "Factory", "Division", "Distance_km", "Units"]
    target_col   = "Lead_Time_Days"

    subset = df[feature_cols + [target_col]].dropna().copy()

    encoders = {}
    cat_cols = ["Region", "Ship Mode", "Factory", "Division"]
    for col in cat_cols:
        le = LabelEncoder()
        subset[col] = le.fit_transform(subset[col].astype(str))
        encoders[col] = le

    # Normalise numerics
    scaler = StandardScaler()
    subset[["Distance_km", "Units"]] = scaler.fit_transform(subset[["Distance_km", "Units"]])
    encoders["scaler"] = scaler

    X = subset[feature_cols]
    y = subset[target_col]
    return X, y, feature_cols, encoders


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return a dict of high-level KPIs for the EDA dashboard."""
    return {
        "total_orders":        len(df),
        "unique_products":     df["Product Name"].nunique(),
        "unique_customers":    df["Customer ID"].nunique(),
        "avg_lead_time":       round(df["Lead_Time_Days"].mean(), 1),
        "median_lead_time":    round(df["Lead_Time_Days"].median(), 1),
        "total_sales":         round(df["Sales"].sum(), 2),
        "total_profit":        round(df["Gross Profit"].sum(), 2),
        "avg_profit_margin":   round(df["Profit_Margin"].mean() * 100, 1),
        "regions":             sorted(df["Region"].unique().tolist()),
        "factories":           sorted(df["Factory"].unique().tolist()),
        "ship_modes":          sorted(df["Ship Mode"].unique().tolist()),
        "products":            sorted(df["Product Name"].unique().tolist()),
    }
