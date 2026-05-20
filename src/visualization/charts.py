"""
src/visualization/charts.py
============================
Matplotlib/Seaborn chart generators.
All functions return Figure objects so they can be embedded in Streamlit.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional

# ─── Aesthetic theme ──────────────────────────────────────────────────────────
PALETTE   = ["#2563EB", "#16A34A", "#DC2626", "#D97706", "#7C3AED", "#0891B2"]
BG_COLOR  = "#0F172A"
TEXT_COLOR = "#F1F5F9"
GRID_COLOR = "#1E293B"

def _apply_dark_style(fig, ax_list):
    fig.patch.set_facecolor(BG_COLOR)
    for ax in (ax_list if isinstance(ax_list, (list, np.ndarray)) else [ax_list]):
        ax.set_facecolor(GRID_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.title.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.tick_params(axis='x', colors=TEXT_COLOR)
        ax.tick_params(axis='y', colors=TEXT_COLOR)


# ─── 1. Delivery Time Distribution ───────────────────────────────────────────

def plot_delivery_time_dist(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _apply_dark_style(fig, axes)

    axes[0].hist(df["delivery_time_hrs"], bins=40, color=PALETTE[0],
                 edgecolor="#0F172A", alpha=0.85)
    axes[0].set_title("Delivery Time Distribution (hrs)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Delivery Time (hrs)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(df["delay_minutes"], bins=40, color=PALETTE[2],
                 edgecolor="#0F172A", alpha=0.85)
    axes[1].set_title("Delay Minutes Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Delay (minutes)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    return fig


# ─── 2. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(feat_importance: dict,
                             model_name: str = "Gradient Boosting") -> plt.Figure:
    if model_name not in feat_importance:
        model_name = next(iter(feat_importance))

    fi    = feat_importance[model_name]
    items = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:12]
    names = [i[0] for i in items]
    vals  = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(9, 5))
    _apply_dark_style(fig, ax)

    colors = [PALETTE[0] if v >= vals[0]*0.5 else PALETTE[1] for v in vals]
    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="#0F172A")

    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color=TEXT_COLOR, fontsize=9)

    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold",
                 color=TEXT_COLOR)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig


# ─── 3. Model Comparison ─────────────────────────────────────────────────────

def plot_model_comparison(results: list) -> plt.Figure:
    names = [r["model_name"] for r in results]
    maes  = [r["MAE"]  for r in results]
    rmses = [r["RMSE"] for r in results]
    r2s   = [r["R2"]   for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    _apply_dark_style(fig, axes)

    for ax, metric, vals, color in zip(
        axes, ["MAE (lower=better)", "RMSE (lower=better)", "R² (higher=better)"],
        [maes, rmses, r2s], PALETTE[:3]
    ):
        bars = ax.bar(names, vals, color=color, edgecolor="#0F172A", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(vals),
                    f"{val:.3f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)

    plt.tight_layout()
    return fig


# ─── 4. Actual vs Predicted ───────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, predictions, model_name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    _apply_dark_style(fig, ax)

    sample = min(500, len(y_test))
    idx    = np.random.choice(len(y_test), sample, replace=False)
    y_s    = np.array(y_test)[idx]
    p_s    = np.array(predictions)[idx]

    ax.scatter(y_s, p_s, alpha=0.45, s=18, color=PALETTE[0], label="Predictions")
    mn, mx = min(y_s.min(), p_s.min()), max(y_s.max(), p_s.max())
    ax.plot([mn, mx], [mn, mx], "--", color=PALETTE[2], linewidth=1.5, label="Perfect fit")

    ax.set_xlabel("Actual (hrs)")
    ax.set_ylabel("Predicted (hrs)")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(facecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    return fig


# ─── 5. Traffic Distribution ─────────────────────────────────────────────────

def plot_traffic_analysis(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _apply_dark_style(fig, axes)

    # Traffic level histogram
    axes[0].hist(df["traffic_level"], bins=30, color=PALETTE[3],
                 edgecolor="#0F172A", alpha=0.85)
    axes[0].set_title("Traffic Level Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Traffic Level (0=free, 1=jam)")

    # Delay by weather
    if "weather_condition" in df.columns:
        weather_order = ["Clear", "Cloudy", "Rain", "Fog", "Heavy Rain", "Storm"]
        weather_order = [w for w in weather_order if w in df["weather_condition"].unique()]
        grp = df.groupby("weather_condition")["delay_minutes"].mean().reindex(weather_order).dropna()
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(grp))]
        bars = axes[1].bar(grp.index, grp.values, color=colors, edgecolor="#0F172A")
        for bar, val in zip(bars, grp.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=8)
        axes[1].set_title("Avg Delay by Weather Condition", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Weather")
        axes[1].set_ylabel("Avg Delay (min)")
        axes[1].tick_params(axis='x', rotation=20)

    plt.tight_layout()
    return fig


# ─── 6. Hub Performance ───────────────────────────────────────────────────────

def plot_hub_performance(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _apply_dark_style(fig, axes)

    # Top delayed hubs (as source)
    top_delay = (df.groupby("source_hub")["delay_minutes"]
                   .mean().sort_values(ascending=False).head(top_n))
    colors = [PALETTE[2] if v > top_delay.mean() else PALETTE[0]
              for v in top_delay.values]
    axes[0].barh(top_delay.index[::-1], top_delay.values[::-1],
                 color=colors[::-1], edgecolor="#0F172A")
    axes[0].set_title(f"Top {top_n} Hubs by Avg Delay (as Source)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Avg Delay (min)")

    # Busiest hubs by volume
    hub_vol = (pd.concat([df["source_hub"], df["destination_hub"]])
                 .value_counts().head(top_n))
    axes[1].barh(hub_vol.index[::-1], hub_vol.values[::-1],
                 color=PALETTE[1], edgecolor="#0F172A")
    axes[1].set_title(f"Top {top_n} Busiest Hubs (Volume)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Shipment Count")

    plt.tight_layout()
    return fig


# ─── 7. Delay Heatmap by Day & Hour ──────────────────────────────────────────

def plot_delay_heatmap(df: pd.DataFrame) -> plt.Figure:
    pivot = df.pivot_table(
        values="delay_minutes", index="hour_of_day",
        columns="day_of_week", aggfunc="mean"
    )
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.columns = [day_labels[c] for c in pivot.columns if c < 7]

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_dark_style(fig, ax)

    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.3,
                linecolor="#0F172A", annot=False, fmt=".0f",
                cbar_kws={"shrink": 0.8})
    ax.set_title("Avg Delay Heatmap — Hour of Day vs Day of Week",
                 fontsize=12, fontweight="bold", color=TEXT_COLOR)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    ax.tick_params(colors=TEXT_COLOR)
    plt.tight_layout()
    return fig


# ─── 8. Route Type Analysis ───────────────────────────────────────────────────

def plot_route_type_analysis(df: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _apply_dark_style(fig, axes)

    rt_delay = df.groupby("route_type")["delay_minutes"].mean().sort_values(ascending=False)
    axes[0].bar(rt_delay.index, rt_delay.values,
                color=PALETTE[:len(rt_delay)], edgecolor="#0F172A")
    axes[0].set_title("Avg Delay by Route Type", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Route Type")
    axes[0].set_ylabel("Avg Delay (min)")

    rt_time = df.groupby("route_type")["delivery_time_hrs"].mean().sort_values(ascending=False)
    axes[1].bar(rt_time.index, rt_time.values,
                color=PALETTE[2:2+len(rt_time)], edgecolor="#0F172A")
    axes[1].set_title("Avg Delivery Time by Route Type", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Route Type")
    axes[1].set_ylabel("Avg Delivery Time (hrs)")

    plt.tight_layout()
    return fig
