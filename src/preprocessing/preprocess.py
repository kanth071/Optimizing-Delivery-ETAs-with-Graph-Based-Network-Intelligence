"""
src/preprocessing/preprocess.py
================================
Production-grade preprocessing pipeline for the logistics delivery dataset.

Steps:
  1. Load raw CSV
  2. Remove duplicates
  3. Handle missing values (median/mode imputation)
  4. Outlier detection & capping (IQR method)
  5. Feature engineering (time, route, risk, interaction)
  6. Encode categoricals (Label Encoding)
  7. Normalize numerics (MinMax)
  8. Save processed output
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os, logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

WEATHER_RISK = {
    "Clear": 1, "Cloudy": 2, "Rain": 3,
    "Heavy Rain": 4, "Fog": 3, "Storm": 5
}

PRIORITY_RANK = {
    "Economy": 1, "Standard": 2, "Express": 3, "Same-Day": 4
}

ROUTE_SPEED = {
    "Highway": 80, "Expressway": 95, "City Road": 35,
    "Rural": 45, "Mixed": 60
}

VEHICLE_SPEED_FACTOR = {
    "FTL Truck": 1.0, "Carting Vehicle": 0.75, "Express Van": 1.15,
    "Mini Truck": 0.9, "Container": 0.85,
}


# ─── Pipeline Steps ──────────────────────────────────────────────────────────

def load_raw(path: str = "data/raw/logistics_dataset.csv") -> pd.DataFrame:
    """Load the raw logistics CSV."""
    df = pd.read_csv(path, parse_dates=["shipment_datetime"])
    log.info(f"[LOAD]  Raw shape: {df.shape}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"[DEDUP] Removed {before - len(df)} duplicates -> {len(df)} rows")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: Numerics -> median, Categoricals -> mode."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    log.info(f"[MISS]  Missing values after fill: {df.isna().sum().sum()}")
    return df


def handle_outliers(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Cap outliers using IQR method (1.5x rule) on numeric columns."""
    if cols is None:
        cols = ["route_distance", "delay_minutes", "delivery_time_hrs",
                "traffic_level", "hub_load"]
    cols = [c for c in cols if c in df.columns]

    capped = 0
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        capped += before

    log.info(f"[OUTLR] Capped {capped} outlier values across {len(cols)} columns")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-specific engineered features."""

    # ── Time-based features ──────────────────────────────────────────────────
    df["hour_of_day"]   = df["shipment_datetime"].dt.hour
    df["day_of_week_num"] = df["shipment_datetime"].dt.dayofweek   # 0=Mon
    df["month"]         = df["shipment_datetime"].dt.month
    df["is_weekend"]    = (df["day_of_week_num"] >= 5).astype(int)
    df["is_peak_hour"]  = df["hour_of_day"].apply(
        lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 20) else 0
    )
    df["quarter"]       = df["shipment_datetime"].dt.quarter

    # ── Route-based features ─────────────────────────────────────────────────
    if "route_id" not in df.columns:
        df["route_id"] = df["source_hub"] + "_" + df["destination_hub"]

    # Average delay per route (computed from the dataset itself)
    route_avg_delay = df.groupby("route_id")["delay_minutes"].transform("mean")
    df["avg_delay_per_route"] = route_avg_delay.round(2)

    # Route shipment volume
    route_volume = df.groupby("route_id")["shipment_id"].transform("count")
    df["route_volume"] = route_volume

    # Estimated travel time without delay (hours)
    df["base_speed"] = df["route_type"].map(ROUTE_SPEED)
    df["est_travel_hrs"] = (df["route_distance"] / df["base_speed"]).round(3)

    # ── Risk / load features ─────────────────────────────────────────────────
    df["weather_risk"]    = df["weather_condition"].map(WEATHER_RISK)
    df["priority_rank"]   = df["shipment_priority"].map(PRIORITY_RANK)

    # Combined congestion score (traffic + hub load)
    if "congestion_score" not in df.columns:
        df["congestion_score"] = (
            0.6 * df["traffic_level"] + 0.4 * df["hub_load"]
        ).round(3)

    # Delay ratio (actual / estimated travel)
    df["delay_ratio"] = (
        df["delay_minutes"] / (df["est_travel_hrs"] * 60 + 1)
    ).round(3)

    # ── Interaction features ─────────────────────────────────────────────────
    df["traffic_weather_interaction"] = (
        df["traffic_level"] * df["weather_risk"]
    ).round(3)

    df["distance_congestion"] = (
        df["route_distance"] * df["congestion_score"]
    ).round(2)

    df["stops_delay_interaction"] = (
        df["num_stops"] * df["delay_minutes"]
    ).round(2)

    # ── Vehicle features ─────────────────────────────────────────────────────
    if "vehicle_type" in df.columns:
        df["vehicle_speed_factor"] = df["vehicle_type"].map(VEHICLE_SPEED_FACTOR).fillna(1.0)
    else:
        df["vehicle_speed_factor"] = 1.0

    # ── Hub capacity utilization ─────────────────────────────────────────────
    if "hub_capacity" in df.columns:
        df["capacity_utilization"] = (df["hub_load"] * df["hub_capacity"]).round(0)

    # ── SLA margin ───────────────────────────────────────────────────────────
    if "promised_delivery_time" in df.columns:
        df["sla_margin_hrs"] = (
            df["promised_delivery_time"] - df["delivery_time_hrs"]
        ).round(3)

    log.info(f"[FEAT]  Engineered features added -> shape: {df.shape}")
    return df


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


def run_pipeline(raw_path: str  = "data/raw/logistics_dataset.csv",
                 save_path: str = "data/processed/logistics_processed.csv"
                 ) -> pd.DataFrame:
    """Execute full preprocessing pipeline end-to-end."""

    df = load_raw(raw_path)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = feature_engineering(df)
    df = encode_categoricals(df)
    df = normalize_numerics(df)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    log.info(f"[SAVE]  Processed data -> {save_path}  shape: {df.shape}")
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print("\nSample columns:", df.columns.tolist()[:20])
