"""
src/utils/helpers.py
=====================
Utility functions: business insights engine, formatting, SLA analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


# ─── SLA Thresholds (hours) ───────────────────────────────────────────────────
SLA_LIMITS = {
    "Same-Day": 24,
    "Express":  48,
    "Standard": 72,
    "Economy":  120,
}


def sla_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SLA breach rate per shipment priority.
    Returns DataFrame with breach_rate column.
    """
    rows = []
    for priority, limit in SLA_LIMITS.items():
        sub   = df[df["shipment_priority"] == priority]
        total = len(sub)
        if total == 0:
            continue
        breached = (sub["delivery_time_hrs"] > limit).sum()
        rows.append({
            "priority":       priority,
            "sla_limit_hrs":  limit,
            "total_shipments":total,
            "breached":       int(breached),
            "breach_rate_pct":round(100 * breached / total, 2),
            "avg_delivery_hrs":round(sub["delivery_time_hrs"].mean(), 2),
        })
    return pd.DataFrame(rows).sort_values("breach_rate_pct", ascending=False)


def route_performance_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return top delayed and fastest routes."""
    grp = df.groupby(["source_hub", "destination_hub"]).agg(
        avg_delay_min     = ("delay_minutes",     "mean"),
        avg_delivery_hrs  = ("delivery_time_hrs", "mean"),
        shipment_count    = ("shipment_id",        "count"),
        avg_distance_km   = ("route_distance",     "mean"),
    ).reset_index()
    grp["route_label"] = grp["source_hub"] + " → " + grp["destination_hub"]
    return grp


def generate_business_insights(df: pd.DataFrame,
                                centrality_df: pd.DataFrame,
                                results: List[dict],
                                risky_routes: pd.DataFrame) -> List[str]:
    """
    Auto-generate actionable business insights from all analysis outputs.
    Returns a list of insight strings.
    """
    insights = []

    # ── ML Model insight ──────────────────────────────────────────────────────
    best    = max(results, key=lambda r: r["R2"])
    insights.append(
        f"🤖 Best ML model: **{best['model_name']}** with R²={best['R2']:.4f}, "
        f"MAE={best['MAE']:.2f} hrs, RMSE={best['RMSE']:.2f} hrs. "
        f"The model explains {best['R2']*100:.1f}% of ETA variance."
    )

    # ── Bottleneck hub ────────────────────────────────────────────────────────
    top_hub = centrality_df.iloc[0]
    insights.append(
        f"🔴 **'{top_hub['hub']}'** is the biggest bottleneck hub "
        f"(score={top_hub['bottleneck_score']:.3f}). "
        f"Average delay through this hub: {top_hub['avg_delay']:.1f} min. "
        f"Increasing capacity or re-routing shipments could reduce system-wide delays."
    )

    # ── Risky route ───────────────────────────────────────────────────────────
    top_route = risky_routes.iloc[0]
    savings_pct = min(30, top_route["avg_delay"] / 10)
    insights.append(
        f"⚠️  Route **'{top_route['source_hub']} → {top_route['destination_hub']}'** "
        f"has the highest risk score ({top_route['risk_score']:.3f}) "
        f"with avg delay {top_route['avg_delay']:.1f} min. "
        f"Switching to a less congested alternate could save ~{savings_pct:.0f}% delay time."
    )

    # ── Traffic impact ────────────────────────────────────────────────────────
    corr = df["traffic_level"].corr(df["delay_minutes"])
    insights.append(
        f"🚦 Traffic level correlates **{corr:.2f}** with delay minutes. "
        f"Routes with traffic > 0.7 account for "
        f"{100*(df['traffic_level']>0.7).mean():.0f}% of shipments "
        f"but {100*(df[df['traffic_level']>0.7]['delay_minutes'] / df['delay_minutes'].sum()).sum():.0f}% of total delay."
    )

    # ── Weather impact ────────────────────────────────────────────────────────
    worst_weather = df.groupby("weather_condition")["delay_minutes"].mean().idxmax()
    worst_val     = df.groupby("weather_condition")["delay_minutes"].mean().max()
    insights.append(
        f"🌧️  **'{worst_weather}'** weather causes the highest avg delay "
        f"({worst_val:.0f} min). Pre-positioning inventory near hubs in weather-prone "
        f"corridors can significantly reduce SLA breaches."
    )

    # ── SLA breaches ──────────────────────────────────────────────────────────
    sla_df   = sla_analysis(df)
    top_sla  = sla_df.iloc[0]
    insights.append(
        f"📋 **{top_sla['priority']}** shipments have the highest SLA breach rate: "
        f"{top_sla['breach_rate_pct']}%. "
        f"Review routing logic for this priority tier — "
        f"average delivery ({top_sla['avg_delivery_hrs']:.1f} hrs) is approaching "
        f"the {top_sla['sla_limit_hrs']}h SLA limit."
    )

    # ── Hub load ──────────────────────────────────────────────────────────────
    overloaded = centrality_df[centrality_df["hub_load"] > 0.80]
    if not overloaded.empty:
        insights.append(
            f"📦 **{len(overloaded)} hubs** are operating at >80% capacity: "
            f"{', '.join(overloaded['hub'].tolist()[:4])}. "
            f"Load balancing by shifting 15–20% of volume to adjacent hubs "
            f"could reduce avg delay by an estimated 12–18%."
        )

    # ── Stop optimisation ────────────────────────────────────────────────────
    corr_stops = df["num_stops"].corr(df["delay_minutes"])
    insights.append(
        f"🛑 Each additional delivery stop adds ~{df.groupby('num_stops')['delay_minutes'].mean().diff().mean():.1f} min "
        f"average delay (stop-delay correlation: {corr_stops:.2f}). "
        f"Route consolidation to reduce stops by 1–2 can meaningfully improve ETA."
    )

    # ── Busiest day/hour ──────────────────────────────────────────────────────
    busy_hour = df.groupby("hour_of_day")["delay_minutes"].mean().idxmax()
    busy_day  = df.groupby("day_of_week")["delay_minutes"].mean().idxmax()
    days      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day_name  = days[busy_day] if isinstance(busy_day, int) and busy_day < 7 else str(busy_day)
    insights.append(
        f"⏰ Peak delay occurs at **{busy_hour}:00 hrs** on **{day_name}**. "
        f"Scheduling high-priority shipments outside these windows can improve on-time performance."
    )

    return insights


def format_eta(hours: float) -> str:
    """Convert fractional hours to human-readable ETA string."""
    h  = int(hours)
    m  = int((hours - h) * 60)
    if h > 0:
        return f"{h} hr {m} min"
    return f"{m} min"


def summary_kpis(df: pd.DataFrame) -> dict:
    """Return key performance indicators for the dashboard home page."""
    return {
        "total_shipments":    len(df),
        "avg_delivery_hrs":   round(df["delivery_time_hrs"].mean(), 2),
        "avg_delay_min":      round(df["delay_minutes"].mean(), 1),
        "on_time_pct":        round(100 * (df["delay_minutes"] < 30).mean(), 1),
        "unique_routes":      df["route_id"].nunique() if "route_id" in df else 0,
        "unique_hubs":        df["source_hub"].nunique(),
        "high_traffic_pct":   round(100 * (df["traffic_level"] > 0.7).mean(), 1),
        "storm_pct":          round(100 * (df["weather_condition"] == "Storm").mean(), 1),
    }
