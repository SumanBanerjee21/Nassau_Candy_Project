"""
app.py
------
Nassau Candy Distributor — Factory Reallocation & Shipping Optimization
FINAL FIXED DEPLOYMENT VERSION
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import (
    load_data,
    get_summary_stats,
    FACTORY_COORDS
)

from model_trainer import load_model_and_encoders

from clustering import (
    build_route_profiles,
    cluster_routes,
    get_slow_routes,
    get_product_region_heatmap
)

from simulator import (
    simulate_product,
    simulate_all_products
)

from recommender import (
    generate_recommendations,
    get_kpis,
    flag_risky_recommendations,
    profit_impact_analysis
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Nassau Candy Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.main {
    background: #0f1117;
}

.stMetric {
    background: #1e2130;
    border-radius: 12px;
    padding: 12px;
}

.big-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0c060;
    border-bottom: 2px solid #f0c060;
    padding-bottom: 8px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: #a0b4d0;
    margin: 20px 0 10px 0;
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# CSV SOURCE
# =============================================================================

CSV_URL = "https://raw.githubusercontent.com/SumanBanerjee21/Nassau_Candy_Project/main/data.csv"

# =============================================================================
# LOAD DATA
# =============================================================================

@st.cache_data(show_spinner="📥 Loading dataset...")
def get_data():

    temp_path = "temp_data.csv"

    df = pd.read_csv(CSV_URL)

    df.to_csv(temp_path, index=False)

    return load_data(temp_path)

# =============================================================================
# LOAD MODEL
# =============================================================================

@st.cache_resource(show_spinner="🤖 Loading ML model...")
def get_model():

    model, encoders = load_model_and_encoders()

    return model, encoders

# =============================================================================
# CLUSTER CACHE
# =============================================================================

@st.cache_data(show_spinner="🔍 Clustering routes...")
def get_clusters(_df):

    rp = build_route_profiles(_df)

    return cluster_routes(rp)

# =============================================================================
# SCENARIO CACHE
# =============================================================================

@st.cache_data(show_spinner="⚙️ Running simulations...")
def get_all_scenarios(_df, _model, _encoders):

    return simulate_all_products(_df, _model, _encoders)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/emoji/96/candy-emoji.png",
        width=80
    )

    st.markdown("## 🍬 Nassau Candy")
    st.markdown("### Optimizer Dashboard")

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📊 EDA Overview",
            "🏭 Factory Optimizer",
            "🏆 Recommendations",
            "⚠️ Risk & Impact"
        ]
    )

# =============================================================================
# MAIN LOAD
# =============================================================================

df = get_data()

model, encoders = get_model()

stats = get_summary_stats(df)

# =============================================================================
# GLOBAL FILTERS
# =============================================================================

with st.sidebar:

    st.divider()

    regions = ["All"] + stats["regions"]

    ship_modes = ["All"] + stats["ship_modes"]

    sel_region = st.selectbox(
        "Region",
        regions
    )

    sel_ship_mode = st.selectbox(
        "Ship Mode",
        ship_modes
    )

    sel_priority = st.slider(
        "Optimization Priority",
        0.0,
        1.0,
        0.7,
        0.05
    )

# =============================================================================
# FILTER DATA
# =============================================================================

dff = df.copy()

if sel_region != "All":
    dff = dff[dff["Region"] == sel_region]

if sel_ship_mode != "All":
    dff = dff[dff["Ship Mode"] == sel_ship_mode]

# =============================================================================
# PAGE 1 — EDA
# =============================================================================

if page == "📊 EDA Overview":

    st.markdown(
        '<div class="big-header">📊 Exploratory Data Analysis</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Orders", len(dff))

    c2.metric("Products", stats["unique_products"])

    c3.metric(
        "Avg Lead Time",
        f"{dff['Lead_Time_Days'].mean():.1f} d"
    )

    c4.metric(
        "Sales",
        f"${dff['Sales'].sum():,.0f}"
    )

    c5.metric(
        "Margin",
        f"{(dff['Profit_Margin'].mean()*100):.1f}%"
    )

    # =========================================================================
    # LEAD TIME DISTRIBUTION
    # =========================================================================

    st.markdown(
        '<div class="section-title">Lead Time Distribution</div>',
        unsafe_allow_html=True
    )

    fig = px.histogram(
        dff,
        x="Lead_Time_Days",
        nbins=40,
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =========================================================================
    # SHIP MODE ANALYSIS
    # =========================================================================

    st.markdown(
        '<div class="section-title">Average Lead Time by Ship Mode</div>',
        unsafe_allow_html=True
    )

    ship_df = (
        dff.groupby("Ship Mode")["Lead_Time_Days"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        ship_df,
        x="Ship Mode",
        y="Lead_Time_Days",
        color="Lead_Time_Days",
        template="plotly_dark"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =========================================================================
    # REGION ANALYSIS
    # =========================================================================

    st.markdown(
        '<div class="section-title">Region Analysis</div>',
        unsafe_allow_html=True
    )

    region_df = (
        dff.groupby("Region")["Lead_Time_Days"]
        .mean()
        .reset_index()
    )

    fig3 = px.bar(
        region_df,
        x="Region",
        y="Lead_Time_Days",
        color="Lead_Time_Days",
        template="plotly_dark"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # =========================================================================
    # HEATMAP
    # =========================================================================

    st.markdown(
        '<div class="section-title">Product × Region Heatmap</div>',
        unsafe_allow_html=True
    )

    heatmap = get_product_region_heatmap(dff)

    fig4 = px.imshow(
        heatmap,
        template="plotly_dark",
        aspect="auto"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # =========================================================================
    # FACTORY MAP
    # =========================================================================

    st.markdown(
        '<div class="section-title">Factory Locations</div>',
        unsafe_allow_html=True
    )

    fmap = pd.DataFrame([
        {
            "Factory": k,
            "Lat": v["lat"],
            "Lon": v["lon"]
        }
        for k, v in FACTORY_COORDS.items()
    ])

    fig5 = px.scatter_geo(
        fmap,
        lat="Lat",
        lon="Lon",
        text="Factory",
        scope="usa",
        template="plotly_dark"
    )

    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("🔍 Raw Data Preview"):

        st.dataframe(
            dff.head(200),
            use_container_width=True
        )

# =============================================================================
# PAGE 2 — FACTORY OPTIMIZER
# =============================================================================

elif page == "🏭 Factory Optimizer":

    st.markdown(
        '<div class="big-header">🏭 Factory Optimizer</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        product = st.selectbox(
            "Product",
            sorted(df["Product Name"].unique())
        )

    with col2:

        region_opt = st.selectbox(
            "Region",
            stats["regions"]
        )

    with col3:

        mode_opt = st.selectbox(
            "Ship Mode",
            stats["ship_modes"]
        )

    with col4:

        units_opt = st.slider(
            "Units",
            1,
            20,
            3
        )

    if st.button("🔮 Simulate"):

        sim_df = simulate_product(
            df,
            model,
            encoders,
            product_name=product,
            region=region_opt,
            ship_mode=mode_opt,
            units=units_opt
        )

        st.dataframe(
            sim_df,
            use_container_width=True
        )

        fig = px.bar(
            sim_df,
            x="Factory",
            y="Predicted_Lead_Time",
            color="Predicted_Lead_Time",
            template="plotly_dark"
        )

        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 3 — RECOMMENDATIONS
# =============================================================================

elif page == "🏆 Recommendations":

    st.markdown(
        '<div class="big-header">🏆 Recommendations</div>',
        unsafe_allow_html=True
    )

    with st.spinner("Generating recommendations..."):

        all_scenarios = get_all_scenarios(
            df,
            model,
            encoders
        )

        rec_df = generate_recommendations(
            all_scenarios,
            top_n=15,
            priority=sel_priority
        )

        rec_df = flag_risky_recommendations(rec_df)

    kpis = get_kpis(df, rec_df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Lead Time Reduction",
        f"{kpis.get('Lead Time Reduction (%)', 0)}%"
    )

    c2.metric(
        "Profit Stability",
        f"{kpis.get('Profit Impact Stability (%)', 0)}%"
    )

    c3.metric(
        "Confidence",
        kpis.get("Scenario Confidence Score", 0)
    )

    c4.metric(
        "Coverage",
        f"{kpis.get('Recommendation Coverage (%)', 0)}%"
    )

    st.dataframe(
        rec_df,
        use_container_width=True
    )

# =============================================================================
# PAGE 4 — RISK PANEL
# =============================================================================

elif page == "⚠️ Risk & Impact":

    st.markdown(
        '<div class="big-header">⚠️ Risk & Impact</div>',
        unsafe_allow_html=True
    )

    profit_df = profit_impact_analysis(dff)

    st.dataframe(
        profit_df,
        use_container_width=True
    )

    fig = px.bar(
        profit_df,
        x="Factory",
        y="Total_Profit",
        color="Avg_Lead_Time",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    cluster_df = get_clusters(df)

    slow_routes = get_slow_routes(
        cluster_df,
        top_n=10
    )

    st.markdown(
        '<div class="section-title">Top Congested Routes</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        slow_routes,
        use_container_width=True
    )

# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption("🚀 Developed by Suman Banerjee")