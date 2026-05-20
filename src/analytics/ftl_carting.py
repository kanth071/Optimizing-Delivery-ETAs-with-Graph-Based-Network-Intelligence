"""
src/analytics/ftl_carting.py
=============================
FTL (Full Truck Load) vs Carting Decision Intelligence Module.

Compares:
  - FTL vs Carting routes on cost, speed, SLA risk
  - Delay probability per vehicle type
  - Optimal vehicle recommendation engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List


# ─── Cost model (simplified) ─────────────────────────────────────────────────
VEHICLE_COST_PER_KM = {
    "FTL Truck":       12.0,   # INR per km
    "Carting Vehicle":  8.0,
    "Express Van":     18.0,
    "Mini Truck":       9.5,
    "Container":       15.0,
}

VEHICLE_CAPACITY_KG = {
    "FTL Truck":       10000,
    "Carting Vehicle":  2000,
    "Express Van":      800,
    "Mini Truck":       4000,
    "Container":       20000,
}


def ftl_vs_carting_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare FTL and Carting performance metrics."""
    if "vehicle_type" not in df.columns:
        return pd.DataFrame()

    vtype_stats = df.groupby("vehicle_type").agg(
        shipment_count    = ("shipment_id", "count"),
        avg_delivery_hrs  = ("delivery_time_hrs", "mean"),
        avg_delay_min     = ("delay_minutes", "mean"),
        avg_distance_km   = ("route_distance", "mean"),
        avg_traffic       = ("traffic_level", "mean"),
        sla_breach_count  = ("sla_breach", "sum") if "sla_breach" in df.columns else ("delay_minutes", lambda x: (x > 300).sum()),
    ).reset_index()

    vtype_stats["sla_breach_pct"] = (
        100 * vtype_stats["sla_breach_count"] / vtype_stats["shipment_count"]
    ).round(2)

    # Cost estimation
    vtype_stats["cost_per_km"] = vtype_stats["vehicle_type"].map(VEHICLE_COST_PER_KM)
    vtype_stats["avg_total_cost"] = (
        vtype_stats["cost_per_km"] * vtype_stats["avg_distance_km"]
    ).round(0)

    # Capacity
    vtype_stats["capacity_kg"] = vtype_stats["vehicle_type"].map(VEHICLE_CAPACITY_KG)

    # Cost efficiency (cost per hour of delivery)
    vtype_stats["cost_per_hr"] = (
        vtype_stats["avg_total_cost"] / vtype_stats["avg_delivery_hrs"]
    ).round(2)

    return vtype_stats.sort_values("avg_delivery_hrs")


def vehicle_recommendation(distance: float, priority: str,
                            traffic: float, weather: str) -> Dict:
    """Recommend optimal vehicle type based on shipment parameters."""

    # Score each vehicle
    scores = {}
    for vtype in VEHICLE_COST_PER_KM:
        cost     = VEHICLE_COST_PER_KM[vtype] * distance
        capacity = VEHICLE_CAPACITY_KG[vtype]

        # Speed factor
        speed_map = {"FTL Truck": 1.0, "Carting Vehicle": 0.75,
                     "Express Van": 1.15, "Mini Truck": 0.9, "Container": 0.85}
        speed = speed_map.get(vtype, 1.0)

        # Priority bonus (Express/Same-Day prefer faster vehicles)
        priority_bonus = {"Same-Day": 1.5, "Express": 1.2,
                          "Standard": 1.0, "Economy": 0.8}.get(priority, 1.0)

        # Traffic penalty (carting vehicles suffer more in traffic)
        traffic_penalty = 1.0 + traffic * (0.3 if vtype in ["Carting Vehicle", "Mini Truck"] else 0.15)

        # Overall score (higher = better)
        score = (speed * priority_bonus) / (cost / 10000 * traffic_penalty)
        scores[vtype] = round(score, 3)

    best = max(scores, key=scores.get)
    return {
        "recommended_vehicle": best,
        "scores": scores,
        "estimated_cost": round(VEHICLE_COST_PER_KM[best] * distance, 0),
        "capacity_kg": VEHICLE_CAPACITY_KG[best],
    }


def ftl_carting_insights(df: pd.DataFrame) -> List[str]:
    """Generate FTL vs Carting business insights."""
    if "vehicle_type" not in df.columns:
        return ["Vehicle type data not available."]

    stats = ftl_vs_carting_analysis(df)
    insights = []

    # Fastest vehicle
    fastest = stats.loc[stats["avg_delivery_hrs"].idxmin()]
    insights.append(
        f"Fastest Vehicle: '{fastest['vehicle_type']}' with avg delivery "
        f"{fastest['avg_delivery_hrs']:.1f} hrs — best for time-critical shipments."
    )

    # Most cost effective
    cheapest = stats.loc[stats["avg_total_cost"].idxmin()]
    insights.append(
        f"Most Cost-Effective: '{cheapest['vehicle_type']}' at avg "
        f"INR {cheapest['avg_total_cost']:,.0f} per shipment — ideal for Economy tier."
    )

    # Highest SLA compliance
    best_sla = stats.loc[stats["sla_breach_pct"].idxmin()]
    insights.append(
        f"Best SLA Compliance: '{best_sla['vehicle_type']}' with only "
        f"{best_sla['sla_breach_pct']:.1f}% breach rate."
    )

    # FTL vs Carting comparison
    ftl = stats[stats["vehicle_type"] == "FTL Truck"]
    cart = stats[stats["vehicle_type"] == "Carting Vehicle"]
    if not ftl.empty and not cart.empty:
        ftl_r = ftl.iloc[0]
        cart_r = cart.iloc[0]
        time_diff = cart_r["avg_delivery_hrs"] - ftl_r["avg_delivery_hrs"]
        cost_diff = ftl_r["avg_total_cost"] - cart_r["avg_total_cost"]
        insights.append(
            f"FTL vs Carting: FTL is {abs(time_diff):.1f} hrs faster but costs "
            f"INR {abs(cost_diff):,.0f} more per shipment. "
            f"Use FTL for Express/Same-Day, Carting for Economy."
        )

    return insights
