"""
data/generate_dataset.py
=========================
Production-grade Synthetic Logistics Dataset Generator.
Generates realistic 5000+ row delivery dataset with comprehensive fields
for ETA optimization, graph analytics, and SLA intelligence.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ─── Hub and Route Definitions ────────────────────────────────────────────────

HUBS = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Surat", "Nagpur", "Indore", "Bhopal", "Patna",
    "Chandigarh", "Coimbatore", "Vizag", "Kochi", "Vadodara"
]

# Hub capacities (max daily throughput)
HUB_CAPACITY = {
    "Mumbai": 5000, "Delhi": 4800, "Bangalore": 4200, "Chennai": 3500,
    "Hyderabad": 3800, "Kolkata": 3200, "Pune": 2800, "Ahmedabad": 2600,
    "Jaipur": 2200, "Lucknow": 2000, "Surat": 1800, "Nagpur": 1600,
    "Indore": 1400, "Bhopal": 1200, "Patna": 1100,
    "Chandigarh": 1000, "Coimbatore": 900, "Vizag": 850, "Kochi": 800,
    "Vadodara": 750,
}

# Corridors (major logistics corridors in India)
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

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Heavy Rain", "Fog", "Storm"]
ROUTE_TYPES       = ["Highway", "City Road", "Rural", "Expressway", "Mixed"]
SHIPMENT_PRIORITY = ["Standard", "Express", "Same-Day", "Economy"]
VEHICLE_TYPES     = ["FTL Truck", "Carting Vehicle", "Express Van", "Mini Truck", "Container"]

# Weather delay multipliers
WEATHER_DELAY = {
    "Clear": 1.0, "Cloudy": 1.1, "Rain": 1.4,
    "Heavy Rain": 1.8, "Fog": 1.5, "Storm": 2.2
}

# Route speed multipliers (km/h base speed)
ROUTE_SPEED = {
    "Highway": 80, "Expressway": 95, "City Road": 35,
    "Rural": 45, "Mixed": 60
}

# Vehicle speed multipliers
VEHICLE_SPEED_FACTOR = {
    "FTL Truck": 1.0, "Carting Vehicle": 0.75, "Express Van": 1.15,
    "Mini Truck": 0.9, "Container": 0.85,
}

# SLA thresholds (hours)
SLA_LIMITS = {
    "Same-Day": 24, "Express": 48, "Standard": 72, "Economy": 120,
}


def _get_corridor(src, dst):
    """Determine corridor for a route pair."""
    if (src, dst) in CORRIDORS:
        return CORRIDORS[(src, dst)]
    if (dst, src) in CORRIDORS:
        return CORRIDORS[(dst, src)]
    return "General Network"


def _get_intermediate_hub(src, dst):
    """Assign an intermediate hub for multi-hop routes."""
    if random.random() < 0.4:  # 40% of routes have an intermediate stop
        candidates = [h for h in HUBS if h != src and h != dst]
        return random.choice(candidates)
    return None


def _categorize_time_of_day(hour):
    """Categorize hour into time-of-day bucket."""
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


def generate_dataset(n_rows: int = 5000) -> pd.DataFrame:
    """Generate n_rows of synthetic logistics delivery data with comprehensive fields."""

    records = []

    for i in range(n_rows):
        source = random.choice(HUBS)
        dest   = random.choice([h for h in HUBS if h != source])

        route_type        = random.choice(ROUTE_TYPES)
        weather           = random.choice(WEATHER_CONDITIONS)
        priority          = random.choice(SHIPMENT_PRIORITY)
        vehicle_type      = random.choice(VEHICLE_TYPES)
        traffic_level     = round(random.uniform(0.1, 1.0), 2)
        hub_load          = round(random.uniform(0.2, 1.0), 2)
        num_stops         = random.randint(0, 6)

        # Distance: rough realistic values between Indian cities
        base_dist = random.randint(100, 2500)
        route_distance = base_dist + random.randint(-20, 20)

        # Base travel time (hours)
        base_speed   = ROUTE_SPEED[route_type]
        vehicle_factor = VEHICLE_SPEED_FACTOR[vehicle_type]
        travel_hours = route_distance / (base_speed * vehicle_factor)

        # Delay contributions
        weather_factor   = WEATHER_DELAY[weather]
        traffic_factor   = 1 + traffic_level * 0.8
        stop_delay_hrs   = num_stops * random.uniform(0.2, 0.5)
        hub_delay_hrs    = hub_load * random.uniform(0.5, 1.5)
        priority_factor  = {"Same-Day": 0.85, "Express": 0.92,
                            "Standard": 1.0, "Economy": 1.12}[priority]

        # Total actual delivery time in hours
        actual_delivery_time = (
            travel_hours * weather_factor * traffic_factor * priority_factor
            + stop_delay_hrs
            + hub_delay_hrs
            + random.gauss(0, 0.3)
        )
        actual_delivery_time = max(0.5, actual_delivery_time)

        # Promised delivery time (based on SLA)
        sla_limit = SLA_LIMITS[priority]
        promised_delivery_time = min(sla_limit, travel_hours * 1.3 + random.uniform(1, 5))

        # Delay in minutes vs ideal (no traffic, clear weather)
        ideal_time    = route_distance / (base_speed * vehicle_factor)
        delay_minutes = max(0, (actual_delivery_time - ideal_time) * 60)

        # SLA breach flag
        sla_breach = 1 if actual_delivery_time > sla_limit else 0

        # Congestion score (composite)
        congestion_score = round(0.6 * traffic_level + 0.4 * hub_load, 3)

        # Simulated shipment datetime
        base_date      = datetime(2024, 1, 1)
        shipment_date  = base_date + timedelta(
            days=random.randint(0, 364),
            hours=random.randint(6, 22)
        )

        hour = shipment_date.hour
        time_of_day = _categorize_time_of_day(hour)
        day_of_week = shipment_date.strftime("%A")

        # Corridor and intermediate hub
        corridor = _get_corridor(source, dest)
        intermediate_hub = _get_intermediate_hub(source, dest)

        # Hub capacity
        hub_capacity = HUB_CAPACITY.get(source, 1000)

        records.append({
            "shipment_id":           f"SHP{i+1:05d}",
            "source_hub":            source,
            "destination_hub":       dest,
            "intermediate_hub":      intermediate_hub,
            "corridor":              corridor,
            "route_id":              f"{source}_{dest}",
            "route_type":            route_type,
            "vehicle_type":          vehicle_type,
            "route_distance":        round(route_distance, 1),
            "traffic_level":         traffic_level,
            "weather_condition":     weather,
            "congestion_score":      congestion_score,
            "num_stops":             num_stops,
            "hub_load":              hub_load,
            "hub_capacity":          hub_capacity,
            "shipment_priority":     priority,
            "time_of_day":           time_of_day,
            "day_of_week":           day_of_week,
            "actual_delivery_time":  round(actual_delivery_time, 3),
            "promised_delivery_time":round(promised_delivery_time, 3),
            "delivery_time_hrs":     round(actual_delivery_time, 3),
            "delay_minutes":         round(delay_minutes, 1),
            "sla_breach":            sla_breach,
            "shipment_datetime":     shipment_date,
        })

    df = pd.DataFrame(records)

    # ── Inject 2% missing values in selected columns ─────────────────────────
    for col in ["traffic_level", "hub_load", "weather_condition"]:
        idx = np.random.choice(df.index, size=int(0.02 * n_rows), replace=False)
        df.loc[idx, col] = np.nan

    # ── Inject ~0.5% duplicate rows ──────────────────────────────────────────
    dup_idx = np.random.choice(df.index, size=int(0.005 * n_rows), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df


if __name__ == "__main__":
    df = generate_dataset(5000)
    out_path = "data/raw/logistics_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Dataset generated: {df.shape[0]} rows x {df.shape[1]} cols")
    print(df.head(3).to_string())
