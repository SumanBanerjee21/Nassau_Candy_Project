"""
recommender.py
--------------
Optimization & Recommendation Engine.

Inputs  : comprehensive scenario DataFrame from simulator.simulate_all_products()
Outputs : ranked factory reassignment recommendations with KPI scores
"""

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────
#  SCORING WEIGHTS (adjustable via priority slider)
# ────────────────────────────────────────────────

def _score(reduction_pct: float, profit_margin: float,
           confidence: float, priority: float) -> float:
    """
    Composite score.
    priority = 0.0 → maximise profit
    priority = 1.0 → maximise speed (lead-time reduction)
    """
    speed_score  = max(0, reduction_pct) / 100
    profit_score = max(0, profit_margin) / 100
    conf_score   = confidence
    return (priority * speed_score
            + (1 - priority) * profit_score
            + 0.1 * conf_score)


# ────────────────────────────────────────────────
#  TOP-N RECOMMENDATIONS
# ────────────────────────────────────────────────

def generate_recommendations(scenario_df: pd.DataFrame,
                              top_n: int = 10,
                              priority: float = 0.7,
                              min_reduction_pct: float = 5.0) -> pd.DataFrame:
    """
    Filter and rank the best factory reassignment recommendations.

    Parameters
    ----------
    scenario_df       : output of simulator.simulate_all_products()
    top_n             : number of recommendations to return
    priority          : 0 = profit, 1 = speed
    min_reduction_pct : only include rows where lead-time reduces by at least this %

    Returns
    -------
    DataFrame of top-N recommendations with composite scores
    """
    if scenario_df.empty:
        return pd.DataFrame()

    # Keep only rows that are NOT the current factory and improve lead time
    rec = scenario_df[
        (~scenario_df["Is_Current"]) &
        (scenario_df["Lead_Time_Reduction_Pct"] >= min_reduction_pct)
    ].copy()

    if rec.empty:
        # Relax filter if nothing passes
        rec = scenario_df[~scenario_df["Is_Current"]].copy()

    rec["Composite_Score"] = rec.apply(
        lambda r: _score(
            r["Lead_Time_Reduction_Pct"],
            r["Profit_Margin_Pct"],
            r["Confidence_Score"],
            priority,
        ),
        axis=1,
    )

    rec = rec.sort_values("Composite_Score", ascending=False).drop_duplicates(
        subset=["Product Name", "Region", "Ship Mode"]
    )

    return rec.head(top_n).reset_index(drop=True)


def get_kpis(original_df: pd.DataFrame, recommendations: pd.DataFrame) -> dict:
    """
    Compute project-level KPIs comparing current state vs. recommended.
    """
    if recommendations.empty:
        return {}

    avg_lead_reduction  = recommendations["Lead_Time_Reduction_Pct"].mean()
    avg_confidence      = recommendations["Confidence_Score"].mean()
    coverage            = recommendations["Product Name"].nunique()
    total_products      = original_df["Product Name"].nunique()
    coverage_pct        = round(coverage / total_products * 100, 1)

    # Profit impact stability (low std = stable)
    profit_std          = recommendations["Avg_Profit"].std()
    profit_stability    = round(max(0, 1 - profit_std / (recommendations["Avg_Profit"].mean() + 1e-6)) * 100, 1)

    return {
        "Lead Time Reduction (%)":     round(avg_lead_reduction, 1),
        "Profit Impact Stability (%)": profit_stability,
        "Scenario Confidence Score":   round(avg_confidence, 2),
        "Recommendation Coverage (%)": coverage_pct,
    }


def flag_risky_recommendations(recommendations: pd.DataFrame,
                                min_confidence: float = 0.3,
                                min_profit_margin: float = 10.0) -> pd.DataFrame:
    """
    Add a Risk_Flag column to recommendations.
    Flags rows with low confidence or low profit margin as HIGH RISK.
    """
    if recommendations.empty:
        return recommendations

    df = recommendations.copy()
    df["Risk_Flag"] = "✅ Low Risk"
    df.loc[
        (df["Confidence_Score"] < min_confidence) |
        (df["Profit_Margin_Pct"] < min_profit_margin),
        "Risk_Flag"
    ] = "⚠️ High Risk"
    return df


def profit_impact_analysis(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-factory profit impact summary from historical data.
    """
    return (
        original_df.groupby("Factory")
        .agg(
            Total_Profit    = ("Gross Profit", "sum"),
            Avg_Profit      = ("Gross Profit", "mean"),
            Avg_Margin_Pct  = ("Profit_Margin", lambda x: round(x.mean() * 100, 1)),
            Order_Count     = ("Gross Profit", "count"),
            Avg_Lead_Time   = ("Lead_Time_Days", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("Avg_Lead_Time")
    )
