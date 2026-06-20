# 🚚 AI-Powered Delivery ETA Optimization
### Graph-Based Network Intelligence for Logistics

> An industry-grade ML + Graph Analytics system for predicting delivery ETAs, detecting bottleneck hubs, and optimizing logistics routes — built with Python, Scikit-learn, NetworkX, and Streamlit.

---

## 📋 Table of Contents
1. [Project Overview](#overview)
2. [System Architecture](#architecture)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [Setup & Installation](#setup)
6. [Usage](#usage)
7. [Dashboard Pages](#dashboard)
8. [Dataset](#dataset)
9. [Model Performance](#models)
10. [Future Improvements](#future)

---

## 🎯 Project Overview <a name="overview"></a>

This project builds an intelligent logistics analytics system that:
- **Predicts Delivery ETA** using ML regression (Linear, Random Forest, Gradient Boosting)
- **Detects Bottleneck Hubs** via graph centrality analysis (Betweenness, Closeness, Degree)
- **Models the Network** as a weighted directed graph using NetworkX
- **Optimizes Routes** with Dijkstra's shortest-path algorithm (fastest / safest / shortest)
- **Generates Business Insights** automatically from ML + Graph outputs
- **Visualizes Everything** via a 9-page Streamlit dashboard

---

## 🏗️ System Architecture <a name="architecture"></a>

```
Raw Logistics Data (5,000 rows)
         │
         ▼
┌─────────────────────────────┐
│   PREPROCESSING PIPELINE    │
│  Dedup → Impute → Engineer  │
│  Encode → Normalize         │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌────────┐   ┌──────────────┐
│  ML    │   │    GRAPH     │
│Pipeline│   │   Pipeline   │
│        │   │              │
│ Lin Reg│   │ Build Graph  │
│ Rand F │   │ Centrality   │
│ Grad B │   │ Dijkstra     │
└───┬────┘   └──────┬───────┘
    └────────┬───────┘
             ▼
    ┌─────────────────┐
    │  STREAMLIT      │
    │  DASHBOARD      │
    │  + Insights     │
    └─────────────────┘
```

---

## ✨ Features <a name="features"></a>

| Feature | Description |
|---|---|
| 📦 Data Pipeline | Dedup, imputation, feature engineering, encoding, normalization |
| 🤖 ETA Prediction | 3 ML models with R²=0.99 on best model |
| 🕸️ Graph Analytics | 20-node, 380-edge directed delivery network |
| 🔍 Bottleneck Detection | Centrality metrics + composite bottleneck score |
| 🗺️ Route Optimization | Fastest / Safest / Shortest path comparison |
| 💡 Business Insights | 8+ auto-generated actionable recommendations |
| 📊 Dashboard | 9-page interactive Streamlit application |
| 📈 Model Comparison | MAE / RMSE / R² across all 3 models |

---

## 🛠️ Tech Stack <a name="tech-stack"></a>

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| ML | `scikit-learn` (LinearReg, RandomForest, GradientBoosting) |
| Graph | `networkx` (DiGraph, Dijkstra, Centrality) |
| Visualization | `matplotlib`, `seaborn` |
| Dashboard | `streamlit` |
| Utilities | `scipy`, `pickle`, `json` |

---

## ⚙️ Setup & Installation <a name="setup"></a>

```bash
# 1. Clone / download the project
cd delivery_eta

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run full pipeline (preprocesses data + trains models)
python main.py

# 4. Launch Streamlit dashboard
streamlit run src/dashboard/app.py
```

---

## 🚀 Usage <a name="usage"></a>

### Run Pipeline Only
```bash
python main.py
```

### Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

### Predict ETA Programmatically
```python
from src.models.predict import load_best_model, build_feature_row

model, name, _ = load_best_model()
X = build_feature_row(
    distance=500, traffic=0.6, hub_load=0.5, stops=2,
    weather="Rain", priority="Express", route_type="Highway"
)
eta_hrs = model.predict(X)[0]
print(f"Predicted ETA: {eta_hrs:.2f} hrs")
```

---

## 📱 Dashboard Pages <a name="dashboard"></a>

| Page | Description |
|---|---|
| 🏠 Home | KPI cards, system architecture, tech stack |
| 📊 Dataset Overview | Sample data, statistics, distribution charts |
| 🔮 ETA Prediction | Interactive form with real-time predictions |
| 🕸️ Graph Network | Network visualization with path highlighting |
| 🔍 Bottleneck Analysis | Centrality table, subgraph, risky routes |
| 🗺️ Route Optimizer | Compare fastest / safest / shortest routes |
| 💡 Business Insights | Auto-generated operational recommendations |
| 📈 Model Performance | Metrics, actual vs predicted, feature importance |
| 🏁 Conclusion | Achievements, future work, deployment notes |

---

## 📊 Dataset <a name="dataset"></a>

Real-world **Delhivery logistics dataset** mapped across 20 major Indian logistics hubs.

| Column | Description |
|---|---|
| `source_hub` | Origin city / warehouse (mapped from raw source name) |
| `destination_hub` | Delivery destination (mapped from raw destination name) |
| `route_distance` | Distance in km |
| `traffic_level` | Traffic congestion factor based on OSRM travel times |
| `weather_condition` | Weather condition assigned based on temporal monsoons |
| `num_stops` | Intermediate stops |
| `hub_load` | Hub capacity utilization load factor |
| `shipment_priority` | Economy / Standard / Express / Same-Day |
| `delay_minutes` | Actual delay vs OSRM travel time |
| `delivery_time_hrs` | **Target variable** (actual trip travel time in hours) |

**Engineered features:** `congestion_score`, `weather_risk`, `avg_delay_per_route`, `est_travel_hrs`, `is_peak_hour`, `is_weekend`, etc.

---

## 🤖 Model Performance <a name="models"></a>

| Model | MAE (hrs) | RMSE (hrs) | R² |
|---|---|---|---|
| Linear Regression | ~8.13 | ~11.62 | ~0.904 |
| Random Forest | ~3.45 | ~5.62 | ~0.978 |
| **Gradient Boosting** | **~2.23** | **~4.09** | **~0.988** |

---

## 🔭 Future Improvements <a name="future"></a>

- 🌐 Live logistics API integration (Delhivery, FedEx)
- 🗺️ Geo-mapped network with real coordinates (Folium / Kepler.gl)
- ⏱️ Time-series delay forecasting (Prophet / LSTM)
- 🔄 Real-time dynamic rerouting engine
- 🧠 Graph Neural Network embeddings
- ☁️ Cloud deployment (AWS/GCP with Docker)
- 🔔 SLA breach alert system (Email / SMS)

---

## 📄 License
MIT License — Free for educational and commercial use.

---
*Built as an industry-grade capstone project in ML + Graph Analytics.*
