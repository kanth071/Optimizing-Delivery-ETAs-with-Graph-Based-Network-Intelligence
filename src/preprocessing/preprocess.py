"""
src/preprocessing/preprocess.py
================================
Production-grade preprocessing pipeline for the logistics delivery dataset.
Supports both synthetic data and real-world Delhivery logistics dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

HUBS = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Surat", "Nagpur", "Indore", "Bhopal", "Patna",
    "Chandigarh", "Coimbatore", "Vizag", "Kochi", "Vadodara"
]

HUB_CAPACITY = {
    "Mumbai": 5000, "Delhi": 4800, "Bangalore": 4200, "Chennai": 3500,
    "Hyderabad": 3800, "Kolkata": 3200, "Pune": 2800, "Ahmedabad": 2600,
    "Jaipur": 2200, "Lucknow": 2000, "Surat": 1800, "Nagpur": 1600,
    "Indore": 1400, "Bhopal": 1200, "Patna": 1100,
    "Chandigarh": 1000, "Coimbatore": 900, "Vizag": 850, "Kochi": 800,
    "Vadodara": 750,
}

CORRIDORS = {
    ("Mumbai", "Delhi"): "Western Corridor",
    ("Delhi", "Kolkata"): "Northern-Eastern Corridor",
    ("Mumbai", "Bangalore"): "Western-Southern Corridor",
    ("Chennai", "Bangalore"): "Southern Corridor",
    ("Delhi", "Jaipur"): "NCR-Rajasthan Corridor",
    ("Hyderabad", "Chennai"): "Deccan-Southern Corridor",
    ("Mumbai", "Pune"): "Mumbai-Pune Expressway",
    ("Ahmedabad", "Mumbai"): "Gujarat-Maharashtra Corridor",
    ("Kolkata", "Patna"): "Eastern Corridor",
    ("Delhi", "Lucknow"): "UP Corridor",
}

WEATHER_RISK = {
    "Clear": 1, "Cloudy": 2, "Rain": 3,
    "Heavy Rain": 4, "Fog": 3, "Storm": 5
}

PRIORITY_RANK = {
    "Economy": 1, "Standard": 2, "Express": 3, "Same-Day": 4
}

ROUTE_SPEED = {
    "FTL": 65, "Carting": 45, "Highway": 80, "Expressway": 95, "City Road": 35, "Rural": 45, "Mixed": 60
}

VEHICLE_SPEED_FACTOR = {
    "FTL Truck": 1.0, "Carting Vehicle": 0.75, "Express Van": 1.15, "Mini Truck": 0.9, "Container": 0.85,
}

SLA_LIMITS = {
    "Same-Day": 24, "Express": 48, "Standard": 72, "Economy": 120,
}


def _get_corridor(src, dst):
    if (src, dst) in CORRIDORS:
        return CORRIDORS[(src, dst)]
    if (dst, src) in CORRIDORS:
        return CORRIDORS[(dst, src)]
    return "General Network"


def _categorize_time_of_day(hour):
    if 6 <= hour < 10:
        return "Morning Rush"
    elif 10 <= hour < 14:
        return "Midday"
    elif 14 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening Rush"
    else:
        return "Night"


def map_hub_name(name):
    if not isinstance(name, str):
        return "Mumbai"
    nl = name.lower()
    for c in HUBS:
        if c.lower() in nl:
            return c
    # Fallbacks based on common keywords
    if 'bhiwandi' in nl or 'lowerparel' in nl or 'maharashtra' in nl:
        return 'Mumbai'
    if 'gurgaon' in nl or 'noida' in nl or 'bilaspur' in nl or 'haryana' in nl:
        return 'Delhi'
    if 'bengaluru' in nl or 'karnataka' in nl:
        return 'Bangalore'
    if 'kanpur' in nl or 'up' in nl or 'uttar' in nl:
        return 'Lucknow'
    if 'gujarat' in nl or 'anand' in nl or 'khambhat' in nl:
        return 'Ahmedabad'
    if 'kerala' in nl or 'aluva' in nl:
        return 'Kochi'
    if 'tamil' in nl:
        return 'Chennai'
    if 'bengal' in nl:
        return 'Kolkata'
    if 'rajasthan' in nl:
        return 'Jaipur'
    if 'mp' in nl or 'madhya' in nl:
        return 'Indore'
    if 'bihar' in nl:
        return 'Patna'
    if 'punjab' in nl:
        return 'Chandigarh'
    # Deterministic hash fallback
    h = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16)
    return HUBS[h % len(HUBS)]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all object columns (excluding IDs & datetime)."""
    cat_cols = ["source_hub", "destination_hub", "weather_condition",
                "route_type", "shipment_priority", "route_id"]

    # Add new categorical columns if present
    for extra in ["vehicle_type", "corridor", "time_of_day", "day_of_week"]:
        if extra in df.columns:
            cat_cols.append(extra)

    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    df.attrs["encoders"] = encoders
    log.info(f"[ENC]   Encoded {len(cat_cols)} categorical columns")
    return df


def normalize_numerics(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """MinMax-normalize selected numeric columns."""
    if cols is None:
        cols = ["route_distance", "traffic_level", "hub_load",
                "num_stops", "congestion_score", "avg_delay_per_route",
                "est_travel_hrs", "weather_risk", "priority_rank"]

    cols = [c for c in cols if c in df.columns]
    scaler = MinMaxScaler()
    df[[f"{c}_norm" for c in cols]] = scaler.fit_transform(df[cols])
    df.attrs["scaler"] = scaler
    df.attrs["scaled_cols"] = cols
    log.info(f"[NORM]  Normalized {len(cols)} columns")
    return df


def run_pipeline(raw_path: str  = "data/raw/delivery_data.csv",
                 save_path: str = "data/processed/logistics_processed.csv"
                 ) -> pd.DataFrame:
    """Execute Delhivery logistics dataset preprocessing pipeline."""
    log.info(f"[PIPELINE] Starting preprocessing for Delhivery dataset at: {raw_path}")
    
    # 1. Load raw data
    df = pd.read_csv(raw_path)
    log.info(f"[LOAD] Raw shape: {df.shape}")
    
    # 2. Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"[DEDUP] Removed {before - len(df)} duplicates -> {len(df)} rows")
    
    # 3. Drop rows with null trip_uuid, source_name, or destination_name
    df = df.dropna(subset=["trip_uuid", "source_name", "destination_name"])
    log.info(f"[CLEAN] After dropping critical NaNs: {df.shape}")
    
    # 4. Parse timestamps
    df["trip_creation_time"] = pd.to_datetime(df["trip_creation_time"])
    
    # 5. Group by trip_uuid to aggregate at trip level
    log.info("[GROUP] Aggregating tracking updates at trip level...")
    
    # Determine the number of updates per trip (num_stops equivalent)
    num_updates = df.groupby("trip_uuid").size()
    
    # Aggregate fields by taking the last state of each trip (since updates are cumulative)
    df_last = df.groupby("trip_uuid").last().reset_index()
    df_last["num_stops"] = df_last["trip_uuid"].map(num_updates)
    
    log.info(f"[GROUP] Grouped into {len(df_last)} unique trips")
    
    # 6. Map hubs to the 20 major Indian cities
    df_last["source_hub"] = df_last["source_name"].apply(map_hub_name)
    df_last["destination_hub"] = df_last["destination_name"].apply(map_hub_name)
    df_last["intermediate_hub"] = None
    df_last["corridor"] = df_last.apply(lambda r: _get_corridor(r["source_hub"], r["destination_hub"]), axis=1)
    
    # 7. Map vehicle_type and shipment_priority
    def assign_priority(row):
        h = int(hashlib.md5(row['trip_uuid'].encode('utf-8')).hexdigest(), 16)
        if row['route_type'] == 'FTL':
            return 'Standard' if (h % 10 < 7) else 'Economy'
        else:
            return 'Express' if (h % 10 < 8) else 'Same-Day'
            
    df_last["shipment_priority"] = df_last.apply(assign_priority, axis=1)
    df_last["vehicle_type"] = df_last["route_type"].map({"FTL": "FTL Truck", "Carting": "Express Van"}).fillna("Express Van")
    df_last["vehicle_speed_factor"] = df_last["vehicle_type"].map(VEHICLE_SPEED_FACTOR).fillna(1.0)
    
    # 8. Set up distances and times
    # Convert actual_time (minutes) to delivery_time_hrs (hours)
    df_last["delivery_time_hrs"] = (df_last["actual_time"] / 60.0).round(3)
    df_last["actual_delivery_time"] = df_last["delivery_time_hrs"]
    df_last["route_distance"] = df_last["actual_distance_to_destination"].round(1)
    
    # Delay in minutes
    df_last["delay_minutes"] = (df_last["actual_time"] - df_last["osrm_time"]).clip(lower=0).round(1)
    
    # SLA limit and SLA breach calculation
    df_last["promised_delivery_time"] = df_last["shipment_priority"].map(SLA_LIMITS)
    df_last["sla_breach"] = (df_last["delivery_time_hrs"] > df_last["promised_delivery_time"]).astype(int)
    
    # 9. Extract temporal features
    df_last["shipment_datetime"] = df_last["trip_creation_time"]
    df_last["hour_of_day"] = df_last["shipment_datetime"].dt.hour
    df_last["day_of_week_num"] = df_last["shipment_datetime"].dt.dayofweek
    df_last["day_of_week"] = df_last["shipment_datetime"].dt.strftime("%A")
    df_last["month"] = df_last["shipment_datetime"].dt.month
    df_last["is_weekend"] = (df_last["day_of_week_num"] >= 5).astype(int)
    df_last["is_peak_hour"] = df_last["hour_of_day"].apply(
        lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 20) else 0
    )
    df_last["quarter"] = df_last["shipment_datetime"].dt.quarter
    df_last["time_of_day"] = df_last["hour_of_day"].apply(_categorize_time_of_day)
    
    # 10. Map weather condition deterministically based on month
    def assign_weather(row):
        dt = row['shipment_datetime']
        h = int(hashlib.md5(row['trip_uuid'].encode('utf-8')).hexdigest(), 16)
        month = dt.month
        if month in [6, 7, 8, 9]:
            return ['Rain', 'Heavy Rain', 'Storm'][h % 3]
        elif month in [12, 1, 2]:
            return ['Fog', 'Cloudy', 'Clear'][h % 3]
        else:
            return ['Clear', 'Cloudy'][h % 2]
            
    df_last["weather_condition"] = df_last.apply(assign_weather, axis=1)
    df_last["weather_risk"] = df_last["weather_condition"].map(WEATHER_RISK).fillna(2)
    df_last["priority_rank"] = df_last["shipment_priority"].map(PRIORITY_RANK).fillna(2)
    
    # 11. Calculate hub load and hub capacity
    hub_counts = df_last["source_hub"].value_counts()
    min_load = hub_counts.min()
    max_load = hub_counts.max()
    # Scale density to load factor [0.2, 1.0]
    scaled_loads = 0.2 + 0.8 * (hub_counts - min_load) / (max_load - min_load + 1e-9)
    df_last["hub_load"] = df_last["source_hub"].map(scaled_loads).round(3)
    df_last["hub_capacity"] = df_last["source_hub"].map(HUB_CAPACITY).fillna(1000)
    
    # 12. Calculate traffic level and congestion score
    # factor = actual_time / osrm_time
    # scale actual_time/osrm_time factor to traffic level [0.1, 1.0]
    df_last["traffic_level"] = (0.1 + 0.9 * (df_last["actual_time"] / (df_last["osrm_time"] + 1.0) - 1.0).clip(0, 3) / 3.0).round(3)
    df_last["congestion_score"] = (0.6 * df_last["traffic_level"] + 0.4 * df_last["hub_load"]).round(3)
    
    # 13. Route-specific averages
    df_last["route_id"] = df_last["source_hub"] + "_" + df_last["destination_hub"]
    route_avg_delay = df_last.groupby("route_id")["delay_minutes"].transform("mean")
    df_last["avg_delay_per_route"] = route_avg_delay.round(2)
    
    route_volume = df_last.groupby("route_id")["trip_uuid"].transform("count")
    df_last["route_volume"] = route_volume
    
    df_last["base_speed"] = df_last["route_type"].map(ROUTE_SPEED).fillna(50)
    df_last["est_travel_hrs"] = (df_last["route_distance"] / df_last["base_speed"]).round(3)
    
    # 14. Interactions
    df_last["delay_ratio"] = (df_last["delay_minutes"] / (df_last["est_travel_hrs"] * 60 + 1)).round(3)
    df_last["traffic_weather_interaction"] = (df_last["traffic_level"] * df_last["weather_risk"]).round(3)
    df_last["distance_congestion"] = (df_last["route_distance"] * df_last["congestion_score"]).round(2)
    df_last["stops_delay_interaction"] = (df_last["num_stops"] * df_last["delay_minutes"]).round(2)
    df_last["capacity_utilization"] = (df_last["hub_load"] * df_last["hub_capacity"]).round(0)
    df_last["sla_margin_hrs"] = (df_last["promised_delivery_time"] - df_last["delivery_time_hrs"]).round(3)
    
    # 15. Rename trip_uuid to shipment_id
    df_last["shipment_id"] = df_last["trip_uuid"]
    
    # 16. Label encoding
    df_processed = encode_categoricals(df_last)
    
    # 17. Min-max normalization
    df_processed = normalize_numerics(df_processed)
    
    # Save processed dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_processed.to_csv(save_path, index=False)
    log.info(f"[SAVE] Processed data -> {save_path} shape: {df_processed.shape}")
    
    return df_processed


if __name__ == "__main__":
    df = run_pipeline()
    print("\nSample columns:", df.columns.tolist()[:20])
