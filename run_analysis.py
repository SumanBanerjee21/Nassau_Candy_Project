"""
run_analysis.py
---------------
Standalone Python script for PURE ANALYSIS (no Streamlit required).

Run this FIRST to:
  1. Load & clean the dataset
  2. Train all 3 models and select the best
  3. Perform route clustering
  4. Run factory simulation for all products
  5. Generate and print recommendations
  6. Save results as CSV files in the output/ folder

Usage:
  python run_analysis.py

Output files in ./output/:
  - clean_data.csv             : cleaned dataset with engineered features
  - model_metrics.csv          : RMSE / MAE / R2 for all 3 models
  - route_clusters.csv         : KMeans cluster assignments per route
  - slow_routes.csv            : top congested routes
  - product_region_heatmap.csv : pivot of avg lead time
  - simulation_results.csv     : all factory simulation scenarios
  - recommendations.csv        : top-N ranked factory reassignments
  - profit_impact.csv          : per-factory profit analysis
  - kpis.txt                   : project-level KPI summary
"""

import os
import sys
import pandas as pd

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure local imports work
sys.path.insert(0, os.path.dirname(__file__))

from data_loader  import load_data, get_feature_matrix, get_summary_stats
from model_trainer import train_models
from clustering   import (build_route_profiles, cluster_routes,
                          get_slow_routes, get_product_region_heatmap, cluster_summary)
from simulator    import simulate_all_products, simulate_product
from recommender  import (generate_recommendations, get_kpis,
                          flag_risky_recommendations, profit_impact_analysis)

# ── CONFIG ────────────────────────────────────────
DATA_PATH  = r"C:\Users\suman\OneDrive\Desktop\Nassau Candy Distributor.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def banner(msg):
    sep = "-" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")


def save(df, name):
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False)
    print(f"  [SAVED] output/{name}")


# ── MAIN ─────────────────────────────────────────
def main():

    # ── STEP 1: Load Data ─────────────────────────
    banner("STEP 1 - Loading & Cleaning Data")
    df = load_data(DATA_PATH)
    print(f"  Rows loaded     : {len(df):,}")
    print(f"  Columns         : {list(df.columns)}")

    stats = get_summary_stats(df)
    print(f"\n  Products        : {stats['unique_products']}")
    print(f"  Unique Customers: {stats['unique_customers']:,}")
    print(f"  Avg Lead Time   : {stats['avg_lead_time']} days")
    print(f"  Total Sales     : ${stats['total_sales']:,.2f}")
    print(f"  Total Profit    : ${stats['total_profit']:,.2f}")
    print(f"  Avg Margin      : {stats['avg_profit_margin']}%")

    save(df, "clean_data.csv")

    # ── STEP 2: Train Models ──────────────────────
    banner("STEP 2 - Feature Engineering & Model Training")
    X, y, feature_names, encoders = get_feature_matrix(df)
    print(f"  Features : {feature_names}")
    print(f"  Samples  : {len(X):,}")

    results    = train_models(X, y, encoders)
    best_model = results["best_model"]

    metrics_rows = [
        {"Model": m["name"], "RMSE": round(m["rmse"], 4),
         "MAE": round(m["mae"], 4), "R2": round(m["r2"], 4)}
        for m in results["metrics"]
    ]
    save(pd.DataFrame(metrics_rows), "model_metrics.csv")

    if results["feature_importances"]:
        print("\n  Feature Importances:")
        for feat, imp in sorted(results["feature_importances"].items(), key=lambda x: -x[1]):
            print(f"    {feat:<22}: {imp:.4f}")

    # ── STEP 3: Route Clustering ──────────────────
    banner("STEP 3 - Route Clustering")
    route_profiles = build_route_profiles(df)
    cluster_df     = cluster_routes(route_profiles)
    summary_df     = cluster_summary(cluster_df)

    print("\n  Cluster Summary:")
    print(summary_df.to_string(index=False))

    save(cluster_df,              "route_clusters.csv")
    save(get_slow_routes(cluster_df, top_n=10), "slow_routes.csv")

    heatmap = get_product_region_heatmap(df)
    heatmap.to_csv(os.path.join(OUTPUT_DIR, "product_region_heatmap.csv"))
    print("  [SAVED] output/product_region_heatmap.csv")

    # ── STEP 4: Simulation ────────────────────────
    banner("STEP 4 - Factory Scenario Simulation (all products)")
    print("  This may take 30-60 seconds ...")
    scenario_df = simulate_all_products(df, best_model, encoders)
    save(scenario_df, "simulation_results.csv")
    print(f"  Total scenarios generated: {len(scenario_df):,}")

    # ── STEP 5: Recommendations ───────────────────
    banner("STEP 5 - Generating Recommendations")
    recommendations = generate_recommendations(scenario_df, top_n=15, priority=0.7)
    recommendations = flag_risky_recommendations(recommendations)
    save(recommendations, "recommendations.csv")

    print("\n  Top Reassignment Recommendations:")
    display_cols = [
        "Product Name", "Region", "Ship Mode", "Factory",
        "Lead_Time_Reduction_Pct", "Profit_Margin_Pct",
        "Confidence_Score", "Composite_Score", "Risk_Flag"
    ]
    available = [c for c in display_cols if c in recommendations.columns]
    print(recommendations[available].to_string(index=False))

    # ── STEP 6: Profit Impact ─────────────────────
    banner("STEP 6 - Factory Profit Impact Analysis")
    profit_df = profit_impact_analysis(df)
    save(profit_df, "profit_impact.csv")
    print(profit_df.to_string(index=False))

    # ── STEP 7: KPIs ─────────────────────────────
    banner("STEP 7 - Project KPIs")
    kpis = get_kpis(df, recommendations)
    with open(os.path.join(OUTPUT_DIR, "kpis.txt"), "w", encoding="utf-8") as f:
        for k, v in kpis.items():
            line = f"  {k:<40}: {v}"
            print(line)
            f.write(line + "\n")
    print("  [SAVED] output/kpis.txt")

    # ── BONUS: Single-Product Deep-Dive ──────────
    banner("BONUS - Product Deep-Dive: Wonka Bar - Milk Chocolate")
    deep = simulate_product(
        df, best_model, encoders,
        product_name="Wonka Bar - Milk Chocolate",
        region="Atlantic",
        ship_mode="Standard Class",
    )
    if not deep.empty:
        print(deep[[
            "Factory", "Is_Current", "Distance_km",
            "Predicted_Lead_Time", "Lead_Time_Reduction_Pct",
            "Profit_Margin_Pct", "Confidence_Score"
        ]].to_string(index=False))

    banner("ANALYSIS COMPLETE - All files saved to ./output/")


if __name__ == "__main__":
    main()
