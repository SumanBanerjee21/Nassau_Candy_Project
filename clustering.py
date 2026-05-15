"""
clustering.py
-------------
KMeans-based route clustering to identify:
  - Consistently slow routes
  - Congested region-product combinations
  - Efficient routes worth replicating
"""

import numpy as np
import pandas as pd
from sklearn.cluster        import KMeans
from sklearn.preprocessing  import StandardScaler


# ────────────────────────────────────────────────
#  CLUSTER LABELS
# ────────────────────────────────────────────────

CLUSTER_LABELS = {
    0: "🟢 Efficient",
    1: "🟡 Moderate",
    2: "🔴 Congested",
}


def build_route_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by (Factory, Region, Product Name, Ship Mode) to build
    a performance profile per route.
    """
    agg = (
        df.groupby(["Factory", "Region", "Product Name", "Ship Mode"])
        .agg(
            Avg_Lead_Time   = ("Lead_Time_Days",  "mean"),
            Std_Lead_Time   = ("Lead_Time_Days",  "std"),
            Avg_Profit      = ("Gross Profit",    "mean"),
            Avg_Sales       = ("Sales",           "mean"),
            Avg_Distance_km = ("Distance_km",     "mean"),
            Order_Count     = ("Lead_Time_Days",  "count"),
            Profit_Margin   = ("Profit_Margin",   "mean"),
        )
        .reset_index()
    )
    agg["Std_Lead_Time"] = agg["Std_Lead_Time"].fillna(0)
    return agg


def cluster_routes(route_df: pd.DataFrame, n_clusters: int = 3,
                   random_state: int = 42) -> pd.DataFrame:
    """
    Apply KMeans clustering to route profiles.
    Returns route_df with added columns: Cluster (int), Cluster_Label (str).
    """
    feature_cols = ["Avg_Lead_Time", "Std_Lead_Time", "Avg_Distance_km",
                    "Avg_Profit", "Order_Count"]

    X = route_df[feature_cols].copy().fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    route_df = route_df.copy()
    route_df["Cluster"] = labels

    # Assign human-readable labels based on avg lead time ordering
    cluster_means = (
        route_df.groupby("Cluster")["Avg_Lead_Time"].mean()
        .sort_values()
        .reset_index()
    )
    label_map = {int(row["Cluster"]): CLUSTER_LABELS[i]
                 for i, row in cluster_means.iterrows()}
    route_df["Cluster_Label"] = route_df["Cluster"].map(label_map)

    return route_df


def get_slow_routes(cluster_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top-N slowest (congested) routes sorted by average lead time."""
    slow = cluster_df[cluster_df["Cluster_Label"].str.contains("Congested")].copy()
    return slow.sort_values("Avg_Lead_Time", ascending=False).head(top_n)


def get_product_region_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a pivot table (Product × Region) of average lead time
    suitable for a heatmap visualisation.
    """
    pivot = df.pivot_table(
        index="Product Name",
        columns="Region",
        values="Lead_Time_Days",
        aggfunc="mean"
    ).round(1)
    return pivot


def cluster_summary(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """High-level summary of each cluster."""
    return (
        cluster_df.groupby("Cluster_Label")
        .agg(
            Routes          = ("Order_Count", "count"),
            Avg_Lead_Time   = ("Avg_Lead_Time", "mean"),
            Avg_Profit      = ("Avg_Profit", "mean"),
            Avg_Distance_km = ("Avg_Distance_km", "mean"),
        )
        .round(2)
        .reset_index()
    )
