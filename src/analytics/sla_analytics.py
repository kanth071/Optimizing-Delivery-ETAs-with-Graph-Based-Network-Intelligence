"""
src/analytics/sla_analytics.py
================================
Dedicated SLA Breach Intelligence Module.

Analyzes:
  - Why breaches happen (root cause)
  - Which hubs contribute most to breaches
  - Peak breach hours & days
  - Delay propagation patterns
  - SLA risk scoring per route
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# ─── SLA Thresholds (hours) ───────────────────────────────────────────────────
SLA_LIMITS = {
    "Same-Day": 24,
    "Express":  48,
    "Standard": 72,
    "Economy":  120,
}


def compute_sla_breaches(df: pd.DataFrame) -> pd.DataFrame:
    """Add SLA breach columns and compute breach details."""
    df = df.copy()
    df["sla_limit_hrs"] = df["shipment_priority"].map(SLA_LIMITS)
    if "sla_breach" not in df.columns:
        df["sla_breach"] = (df["delivery_time_hrs"] > df["sla_limit_hrs"]).astype(int)
    df["sla_margin_hrs"] = (df["sla_limit_hrs"] - df["delivery_time_hrs"]).round(3)
    df["breach_severity"] = np.where(
        df["sla_breach"] == 1,
        np.where(df["sla_margin_hrs"] < -24, "Critical",
                 np.where(df["sla_margin_hrs"] < -6, "High", "Moderate")),
        "None"
    )
    return df


def breach_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    """SLA breach rate per shipment priority."""
    rows = []
    for priority, limit in SLA_LIMITS.items():
        sub   = df[df["shipment_priority"] == priority]
        total = len(sub)
        if total == 0:
            continue
        breached = (sub["delivery_time_hrs"] > limit).sum()
        rows.append({
            "priority":         priority,
            "sla_limit_hrs":    limit,
            "total_shipments":  total,
            "breached":         int(breached),
            "breach_rate_pct":  round(100 * breached / total, 2),
            "avg_delivery_hrs": round(sub["delivery_time_hrs"].mean(), 2),
            "avg_delay_min":    round(sub["delay_minutes"].mean(), 1),
        })
    return pd.DataFrame(rows).sort_values("breach_rate_pct", ascending=False)


def breach_by_hub(df: pd.DataFrame) -> pd.DataFrame:
    """Identify which source hubs contribute most to SLA breaches."""
    df_b = compute_sla_breaches(df)
    hub_stats = df_b.groupby("source_hub").agg(
        total_shipments  = ("shipment_id", "count"),
        total_breaches   = ("sla_breach", "sum"),
        avg_delay_min    = ("delay_minutes", "mean"),
        avg_hub_load     = ("hub_load", "mean"),
    ).reset_index()
    hub_stats["breach_rate_pct"] = (
        100 * hub_stats["total_breaches"] / hub_stats["total_shipments"]
    ).round(2)
    hub_stats["delay_contribution_pct"] = (
        100 * hub_stats["total_breaches"] / hub_stats["total_breaches"].sum()
    ).round(2)
    return hub_stats.sort_values("breach_rate_pct", ascending=False)


def breach_by_time(df: pd.DataFrame) -> Dict:
    """Analyze peak breach hours and days."""
    df_b = compute_sla_breaches(df)
    breached = df_b[df_b["sla_breach"] == 1]

    # Peak breach hours
    hour_col = "hour_of_day" if "hour_of_day" in breached.columns else None
    hour_dist = None
    if hour_col:
        hour_dist = breached[hour_col].value_counts().sort_index().to_dict()

    # Peak breach days
    day_col = "day_of_week" if "day_of_week" in breached.columns else "day_of_week_num"
    day_dist = None
    if day_col in breached.columns:
        day_dist = breached[day_col].value_counts().to_dict()

    # Monthly trend
    month_dist = None
    if "month" in breached.columns:
        month_dist = breached["month"].value_counts().sort_index().to_dict()

    return {
        "total_breaches": int(breached.shape[0]),
        "breach_rate_pct": round(100 * len(breached) / len(df_b), 2),
        "peak_hours": hour_dist,
        "peak_days": day_dist,
        "monthly_trend": month_dist,
    }


def breach_by_corridor(df: pd.DataFrame) -> pd.DataFrame:
    """SLA breach analysis by logistics corridor."""
    if "corridor" not in df.columns:
        return pd.DataFrame()

    df_b = compute_sla_breaches(df)
    corr_stats = df_b.groupby("corridor").agg(
        total_shipments  = ("shipment_id", "count"),
        total_breaches   = ("sla_breach", "sum"),
        avg_delay_min    = ("delay_minutes", "mean"),
        avg_traffic      = ("traffic_level", "mean"),
    ).reset_index()
    corr_stats["breach_rate_pct"] = (
        100 * corr_stats["total_breaches"] / corr_stats["total_shipments"]
    ).round(2)
    return corr_stats.sort_values("breach_rate_pct", ascending=False)


def breach_root_cause_analysis(df: pd.DataFrame) -> List[str]:
    """Generate root cause insights for SLA breaches."""
    df_b = compute_sla_breaches(df)
    breached = df_b[df_b["sla_breach"] == 1]
    non_breached = df_b[df_b["sla_breach"] == 0]

    insights = []

    # Traffic correlation
    avg_traffic_breach = breached["traffic_level"].mean()
    avg_traffic_ok     = non_breached["traffic_level"].mean()
    insights.append(
        f"Traffic Impact: Breached shipments have {avg_traffic_breach:.2f} avg traffic "
        f"vs {avg_traffic_ok:.2f} for on-time deliveries "
        f"({((avg_traffic_breach/avg_traffic_ok - 1)*100):.0f}% higher)."
    )

    # Weather impact
    if "weather_condition" in breached.columns:
        worst_weather = breached["weather_condition"].value_counts().idxmax()
        pct = (breached["weather_condition"] == worst_weather).mean() * 100
        insights.append(
            f"Weather Factor: '{worst_weather}' accounts for {pct:.0f}% of SLA breaches."
        )

    # Hub load
    avg_load_breach = breached["hub_load"].mean()
    insights.append(
        f"Hub Overload: Breached shipments pass through hubs with "
        f"{avg_load_breach:.2f} avg load ({avg_load_breach*100:.0f}% capacity)."
    )

    # Stops impact
    avg_stops_breach = breached["num_stops"].mean()
    avg_stops_ok     = non_breached["num_stops"].mean()
    insights.append(
        f"Stop Overhead: Breached shipments average {avg_stops_breach:.1f} stops "
        f"vs {avg_stops_ok:.1f} for on-time ({(avg_stops_breach-avg_stops_ok):.1f} extra)."
    )

    # Distance
    avg_dist_breach = breached["route_distance"].mean()
    insights.append(
        f"Distance: Average distance for breached shipments is {avg_dist_breach:.0f} km "
        f"vs {non_breached['route_distance'].mean():.0f} km for on-time."
    )

    return insights


def delay_propagation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze how delays propagate through the hub network."""
    hub_delays = df.groupby("source_hub").agg(
        outbound_delay    = ("delay_minutes", "mean"),
        outbound_volume   = ("shipment_id", "count"),
        avg_traffic       = ("traffic_level", "mean"),
        avg_hub_load      = ("hub_load", "mean"),
    ).reset_index()

    dest_delays = df.groupby("destination_hub").agg(
        inbound_delay     = ("delay_minutes", "mean"),
        inbound_volume    = ("shipment_id", "count"),
    ).reset_index().rename(columns={"destination_hub": "source_hub"})

    merged = hub_delays.merge(dest_delays, on="source_hub", how="outer").fillna(0)
    merged["propagation_score"] = (
        0.5 * (merged["outbound_delay"] / merged["outbound_delay"].max()) +
        0.3 * (merged["inbound_delay"] / merged["inbound_delay"].max()) +
        0.2 * (merged["avg_hub_load"])
    ).round(4)
    merged = merged.rename(columns={"source_hub": "hub"})
    return merged.sort_values("propagation_score", ascending=False)
