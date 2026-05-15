# 🍬 Nassau Candy Distributor
## Factory Reallocation & Shipping Optimization Recommendation System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-blue?logo=plotly&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

Nassau Candy Distributor currently assigns products to factories using **static rules and legacy processes**, leading to:

- ❌ Suboptimal shipping distances
- ❌ High lead times for certain regions
- ❌ Margin erosion due to logistics inefficiencies
- ❌ No simulation capability before making reassignment decisions

This project introduces an **intelligent decision-making system** that:

- ✅ Predicts shipping lead times under different factory configurations
- ✅ Recommends which products should be reassigned to alternative factories
- ✅ Balances shipping efficiency and profitability
- ✅ Provides an interactive Streamlit dashboard for live analytics

---

## 🏭 Factories & Products

### Factory Coordinates

| Factory | Location | Latitude | Longitude |
|---------|----------|----------|-----------|
| Lot's O' Nuts | Arizona | 32.88 | -111.77 |
| Wicked Choccy's | Georgia | 32.08 | -81.09 |
| Sugar Shack | Minnesota | 48.12 | -96.18 |
| Secret Factory | Illinois | 41.45 | -90.57 |
| The Other Factory | Tennessee | 35.12 | -89.97 |

### Product–Factory Assignments

| Division | Product | Factory |
|----------|---------|---------|
| Chocolate | Wonka Bar - Nutty Crunch Surprise | Lot's O' Nuts |
| Chocolate | Wonka Bar - Fudge Mallows | Lot's O' Nuts |
| Chocolate | Wonka Bar - Scrumdiddlyumptious | Lot's O' Nuts |
| Chocolate | Wonka Bar - Milk Chocolate | Wicked Choccy's |
| Chocolate | Wonka Bar - Triple Dazzle Caramel | Wicked Choccy's |
| Sugar | Laffy Taffy, SweeTARTS, Nerds, Fun Dip | Sugar Shack |
| Other | Fizzy Lifting Drinks | Sugar Shack |
| Sugar | Everlasting Gobstopper | Secret Factory |
| Other | Lickable Wallpaper, Wonka Gum | Secret Factory |
| Sugar | Hair Toffee | The Other Factory |
| Other | Kazookles | The Other Factory |

---

## 📊 Dataset

| Field | Description |
|-------|-------------|
| Row ID | Unique row identifier |
| Order ID | Unique order identifier |
| Order Date | Date of order |
| Ship Date | Date of shipment |
| Ship Mode | Shipping method (Standard / First / Second / Same Day) |
| Customer ID | Unique customer identifier |
| Country/Region | Country or region of customer |
| City, State/Province | Customer location |
| Division | Product division (Chocolate / Sugar / Other) |
| Region | Customer region (Atlantic / Gulf / Interior / Pacific) |
| Product ID / Name | Product identifier and name |
| Sales | Total sales value |
| Units | Total units ordered |
| Gross Profit | Sales minus Cost |
| Cost | Manufacturing cost |

> **Dataset size:** ~10,195 rows × 18 columns

---

## 🧠 Methodology

```
Raw Data
   │
   ▼
Data Preparation & Feature Engineering
   │  • Parse mixed date formats
   │  • Compute Lead Time (Ship Date − Order Date)
   │  • Map Products → Factories → Coordinates
   │  • Haversine distance (Factory → Region Centroid)
   │  • Encode categoricals, normalize numerics
   │  • Remove IQR outliers
   ▼
Predictive Modeling (Lead Time)
   │  • Linear Regression (baseline)
   │  • Random Forest Regressor
   │  • Gradient Boosting Regressor
   │  • Best model selected by lowest RMSE
   ▼
Route & Product Clustering (KMeans)
   │  • Cluster routes by: Avg Lead Time, Distance, Profit
   │  • Labels: Efficient / Moderate / Congested
   ▼
Scenario Simulation Engine
   │  • For each product × region × ship mode:
   │    Simulate assignment to all 5 factories
   │    Predict new lead time per factory
   ▼
Optimization & Recommendation Engine
   │  • Score = Priority × Speed + (1-Priority) × Profit
   │  • Rank by composite score
   │  • Flag high-risk reassignments
   ▼
Streamlit Dashboard (5 Pages)
```

---

## 📈 Key Performance Indicators (KPIs)

| KPI | Description |
|-----|-------------|
| Lead Time Reduction (%) | Operational gain from reassignment |
| Profit Impact Stability (%) | Financial safety of recommendation |
| Scenario Confidence Score | Reliability based on sample size |
| Recommendation Coverage (%) | % of products covered by recommendations |

---

## 🗂️ Project Structure

```
nassau_candy_optimizer/
│
├── 📄 data_loader.py          # Data loading, cleaning, feature engineering
├── 📄 model_trainer.py        # Train 3 ML models, select best, persist to disk
├── 📄 clustering.py           # KMeans route clustering (Efficient/Moderate/Congested)
├── 📄 simulator.py            # Factory reassignment scenario simulation engine
├── 📄 recommender.py          # Scoring, ranking, KPI calculation, risk flagging
├── 📄 run_analysis.py         # Standalone analysis runner (no UI needed)
├── 📄 app.py                  # 5-page Streamlit dashboard
├── 📄 requirements.txt        # Python dependencies
├── 📄 README.md               # This file
│
├── 📦 best_model.pkl          # Saved best ML model (auto-generated)
├── 📦 encoders.pkl            # Saved label encoders + scaler (auto-generated)
│
└── 📁 output/                 # Generated analysis outputs
    ├── clean_data.csv
    ├── model_metrics.csv
    ├── route_clusters.csv
    ├── slow_routes.csv
    ├── product_region_heatmap.csv
    ├── simulation_results.csv
    ├── recommendations.csv
    ├── profit_impact.csv
    └── kpis.txt
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/nassau-candy-optimizer.git
cd nassau-candy-optimizer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Place `Nassau Candy Distributor.csv` in the project root folder.

### 4. Run the Analysis (Terminal / VS Code)

```bash
python run_analysis.py
```

This generates all output CSVs in the `output/` folder.

### 5. Launch the Interactive Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🌐 Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com) → New Notebook
2. Upload `Nassau Candy Distributor.csv` when prompted
3. Install dependencies:
```python
!pip install scikit-learn plotly joblib -q
```
4. Copy and run each analysis cell from the notebook provided in the repo
5. Download results as a ZIP at the end

> See [`COLAB_GUIDE.md`](./COLAB_GUIDE.md) for full cell-by-cell instructions.

---

## 🖥️ Streamlit Dashboard — 5 Pages

| Page | Description |
|------|-------------|
| 📊 **EDA Overview** | Dataset KPIs, lead time distributions, heatmaps, factory map |
| 🏭 **Factory Optimizer** | Select a product → view predicted performance across all 5 factories |
| 🔀 **What-If Analysis** | Compare current vs proposed factory assignment with waterfall charts |
| 🏆 **Recommendations** | Ranked reassignment table with KPI cards and risk flags |
| ⚠️ **Risk & Impact** | Profit alerts, lead time trends, high-risk route warnings |

**Global Filters (Sidebar):**
- Region selector
- Ship Mode filter
- Optimization Priority slider *(0 = Profit-first → 1 = Speed-first)*

---

## 🤖 ML Models & Evaluation

Three models are trained and compared automatically:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 278.45 | 218.52 | -0.0072 |
| Random Forest | 288.31 | 229.12 | -0.0798 |
| Gradient Boosting | 279.37 | 220.02 | -0.0139 |

> **Best model selected automatically** based on lowest RMSE.

**Features used:**
`Region` · `Ship Mode` · `Factory` · `Division` · `Distance_km` · `Units`

**Target:** `Lead_Time_Days`

---

## 📋 Sample Analysis Results

### Factory Profit Impact
| Factory | Total Profit | Avg Lead Time |
|---------|-------------|---------------|
| Lot's O' Nuts | $21,722 | 1,295 days |
| Wicked Choccy's | $14,050 | 1,293 days |
| Secret Factory | $304 | 1,325 days |
| The Other Factory | $55 | 1,274 days |
| Sugar Shack | $26 | 1,327 days |

### Top Recommendation Example
> **Wonka Bar - Nutty Crunch Surprise** → reassign to **Wicked Choccy's** for the **Atlantic** region
> - Lead Time Reduction: **1.3%**
> - Profit Margin: **71.3%**
> - Risk: ✅ **Low Risk**

---

## 📦 Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
plotly>=5.10.0
streamlit>=1.28.0
scipy>=1.9.0
joblib>=1.2.0
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Clustering | KMeans (Scikit-Learn) |
| Visualization | Plotly Express, Plotly Graph Objects |
| Dashboard | Streamlit |
| Geospatial | Haversine formula (custom) |
| Model Persistence | Joblib |

---

## 🎯 Business Value

This system elevates Nassau Candy Distributor from **descriptive analytics** to **intelligent decision-making** by:

1. **Predicting** shipping outcomes before making changes
2. **Simulating** factory–product reassignment scenarios at scale
3. **Recommending** optimal configurations ranked by business impact
4. **Quantifying** operational improvement with confidence scores
5. **Alerting** stakeholders to high-risk reassignments before execution

---

## 📁 Output Files

| File | Contents |
|------|----------|
| `clean_data.csv` | Cleaned dataset with 26 engineered features |
| `model_metrics.csv` | RMSE, MAE, R² for all 3 models |
| `route_clusters.csv` | KMeans cluster assignment per route |
| `slow_routes.csv` | Top 10 congested routes to prioritize |
| `product_region_heatmap.csv` | Product × Region avg lead time pivot |
| `simulation_results.csv` | All 630 factory-reassignment scenarios |
| `recommendations.csv` | Top 15 ranked factory reassignment recommendations |
| `profit_impact.csv` | Per-factory profit & margin analysis |
| `kpis.txt` | Project-level KPI summary |

---

## 👤 Author

**Suman**
- Project: Nassau Candy Distributor — Factory Reallocation & Shipping Optimization
- Course: Data Analytics / Business Intelligence
- Year: 2026

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: Nassau Candy Distributor (provided for academic analysis)
- Technical Documentation: Unified Mentor — Factory Reallocation & Shipping Optimization Guide
- Libraries: Scikit-Learn, Streamlit, Plotly, Pandas

---

<div align="center">
  <strong>🍬 Nassau Candy Distributor — From Descriptive Analytics to Intelligent Decision-Making 🍬</strong>
</div>
