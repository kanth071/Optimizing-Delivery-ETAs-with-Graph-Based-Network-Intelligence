"""
src/visualization/network_visuals.py
======================================
NetworkX graph visualizations using Matplotlib.
"""

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from typing import Optional

BG_COLOR   = "#0F172A"
GRID_COLOR = "#1E293B"
TEXT_COLOR = "#F1F5F9"


def draw_logistics_network(G: nx.DiGraph,
                            centrality_df: pd.DataFrame,
                            highlight_path: Optional[list] = None,
                            title: str = "Logistics Delivery Network") -> plt.Figure:
    """
    Draw the full logistics network.
      - Node size  ∝ betweenness centrality
      - Node color ∝ bottleneck_score (red=high, green=low)
      - Edge width ∝ shipment_count
      - Path       highlighted in yellow if provided
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # ── Node styling ──────────────────────────────────────────────────────────
    score_map = dict(zip(centrality_df["hub"], centrality_df["bottleneck_score"]))
    scores    = np.array([score_map.get(n, 0) for n in G.nodes])
    mn, mx    = scores.min(), scores.max()
    norm_s    = (scores - mn) / (mx - mn + 1e-9)

    # Color: green (safe) → red (critical)
    node_colors = [plt.cm.RdYlGn_r(v) for v in norm_s]

    bw_map  = dict(zip(centrality_df["hub"], centrality_df["betweenness_centrality"]))
    bw_vals = np.array([bw_map.get(n, 0) for n in G.nodes])
    node_sizes = 300 + 2500 * (bw_vals / (bw_vals.max() + 1e-9))

    # ── Edge styling ──────────────────────────────────────────────────────────
    edge_counts = [G[u][v].get("shipment_count", 1) for u, v in G.edges()]
    max_count   = max(edge_counts) if edge_counts else 1
    edge_widths = [0.3 + 2.5 * (c / max_count) for c in edge_counts]
    edge_colors = ["#334155" for _ in G.edges()]

    nx.draw_networkx_edges(G, pos, ax=ax,
                           width=edge_widths,
                           edge_color=edge_colors,
                           alpha=0.6,
                           arrows=True,
                           arrowsize=10,
                           connectionstyle="arc3,rad=0.08")

    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes,
                           node_color=node_colors,
                           alpha=0.92)

    nx.draw_networkx_labels(G, pos, ax=ax,
                            font_size=7, font_color=TEXT_COLOR,
                            font_weight="bold")

    # ── Highlight path ────────────────────────────────────────────────────────
    if highlight_path and len(highlight_path) >= 2:
        path_edges = [(highlight_path[i], highlight_path[i+1])
                      for i in range(len(highlight_path)-1)
                      if G.has_edge(highlight_path[i], highlight_path[i+1])]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax,
                               width=4, edge_color="#FACC15",
                               arrows=True, arrowsize=18,
                               connectionstyle="arc3,rad=0.08")
        path_nodes = {n for e in path_edges for n in e}
        nx.draw_networkx_nodes(G, pos, nodelist=list(path_nodes), ax=ax,
                               node_size=600, node_color="#FACC15", alpha=1.0)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_els = [
        plt.scatter([], [], s=200, c="#DC2626", label="Critical Hub"),
        plt.scatter([], [], s=200, c="#16A34A", label="Safe Hub"),
        plt.Line2D([0],[0], color="#FACC15", linewidth=3, label="Highlighted Path"),
    ]
    ax.legend(handles=legend_els, loc="lower left",
              facecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT_COLOR, pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_bottleneck_subgraph(G: nx.DiGraph,
                              centrality_df: pd.DataFrame,
                              top_n: int = 8) -> plt.Figure:
    """Draw only the top-N bottleneck hubs and their connections."""
    top_hubs  = centrality_df.head(top_n)["hub"].tolist()
    sub       = G.subgraph(top_hubs).copy()

    # Add one-hop neighbours to show connectivity
    neighbours = set(top_hubs)
    for h in top_hubs:
        neighbours.update(list(G.predecessors(h))[:2])
        neighbours.update(list(G.successors(h))[:2])
    sub = G.subgraph(neighbours).copy()

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    pos = nx.kamada_kawai_layout(sub)

    score_map   = dict(zip(centrality_df["hub"], centrality_df["bottleneck_score"]))
    node_colors = ["#DC2626" if n in top_hubs else "#2563EB" for n in sub.nodes]
    node_sizes  = [700 if n in top_hubs else 280 for n in sub.nodes]

    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color="#475569",
                           width=1.5, alpha=0.7, arrows=True, arrowsize=12)
    nx.draw_networkx_nodes(sub, pos, ax=ax,
                           node_color=node_colors, node_size=node_sizes, alpha=0.92)
    nx.draw_networkx_labels(sub, pos, ax=ax,
                            font_size=8, font_color=TEXT_COLOR, font_weight="bold")

    ax.set_title(f"Top {top_n} Bottleneck Hubs — Subgraph View",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=10)

    legend_els = [
        plt.scatter([], [], s=180, c="#DC2626", label="Bottleneck Hub"),
        plt.scatter([], [], s=120, c="#2563EB", label="Connected Hub"),
    ]
    ax.legend(handles=legend_els, loc="lower right",
              facecolor=GRID_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    ax.axis("off")
    plt.tight_layout()
    return fig


def draw_centrality_bars(centrality_df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of top hubs by each centrality metric."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axes:
        ax.set_facecolor(GRID_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    metrics = [
        ("betweenness_centrality", "#DC2626", "Betweenness Centrality"),
        ("closeness_centrality",   "#2563EB", "Closeness Centrality"),
        ("degree_centrality",      "#16A34A", "Degree Centrality"),
    ]

    for ax, (col, color, title) in zip(axes, metrics):
        top = centrality_df.nlargest(10, col)[["hub", col]].iloc[::-1]
        ax.barh(top["hub"], top[col], color=color, edgecolor="#0F172A", alpha=0.85)
        ax.set_title(title, fontsize=10, fontweight="bold", color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color("#334155")

    plt.tight_layout()
    return fig
