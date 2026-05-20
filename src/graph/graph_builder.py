"""
src/graph/graph_builder.py
===========================
Build a weighted directed graph from the logistics dataset.

  Nodes  = hubs (cities / warehouses)
  Edges  = delivery routes, weighted by avg delivery time and distance
"""

import pandas as pd
import numpy as np
import networkx as nx
import json, os


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Create a directed weighted graph from the logistics dataframe.

    Edge attributes:
      - weight          : average delivery_time_hrs on this route
      - avg_distance    : average route_distance
      - avg_delay       : average delay_minutes
      - avg_traffic     : average traffic_level
      - shipment_count  : number of shipments on this route
      - risk_score      : composite risk (delay + traffic + weather_risk)
    """
    G = nx.DiGraph()

    # ── Add all hubs as nodes ─────────────────────────────────────────────────
    all_hubs = pd.concat([df["source_hub"], df["destination_hub"]]).unique()
    for hub in all_hubs:
        hub_df     = df[(df["source_hub"] == hub) | (df["destination_hub"] == hub)]
        out_df     = df[df["source_hub"] == hub]
        in_df      = df[df["destination_hub"] == hub]

        G.add_node(hub, **{
            "hub_load":         round(hub_df["hub_load"].mean(), 3),
            "avg_delay":        round(hub_df["delay_minutes"].mean(), 2),
            "shipments_out":    len(out_df),
            "shipments_in":     len(in_df),
            "total_shipments":  len(hub_df),
        })

    # ── Add edges aggregated per (source, dest) pair ──────────────────────────
    route_grp = df.groupby(["source_hub", "destination_hub"])

    for (src, dst), grp in route_grp:
        avg_time     = grp["delivery_time_hrs"].mean()
        avg_distance = grp["route_distance"].mean()
        avg_delay    = grp["delay_minutes"].mean()
        avg_traffic  = grp["traffic_level"].mean()
        avg_weather  = grp["weather_risk"].mean() if "weather_risk" in grp else 3.0
        count        = len(grp)

        # Risk score: normalised composite
        risk = (
            0.4 * (avg_delay / 500) +
            0.35 * avg_traffic +
            0.25 * (avg_weather / 5)
        )

        G.add_edge(src, dst, **{
            "weight":         round(avg_time, 3),
            "avg_distance":   round(avg_distance, 1),
            "avg_delay":      round(avg_delay, 2),
            "avg_traffic":    round(avg_traffic, 3),
            "shipment_count": count,
            "risk_score":     round(risk, 4),
        })

    print(f"[GRAPH] Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}")
    return G


def shortest_path(G: nx.DiGraph, source: str, target: str,
                  weight: str = "weight") -> dict:
    """
    Find shortest (fastest) path using Dijkstra's algorithm.
    Returns path list, total weight, and per-hop edge attributes.
    """
    try:
        path   = nx.dijkstra_path(G, source, target, weight=weight)
        length = nx.dijkstra_path_length(G, source, target, weight=weight)
        hops   = []
        for i in range(len(path) - 1):
            edge_data = G[path[i]][path[i+1]]
            hops.append({"from": path[i], "to": path[i+1], **edge_data})
        return {"path": path, "total_weight": round(length, 3),
                "hops": hops, "status": "ok"}
    except nx.NetworkXNoPath:
        # Graceful fallback: if a directed route doesn't exist, search bidirectional pathways
        try:
            G_bi = G.to_undirected().to_directed()
            path   = nx.dijkstra_path(G_bi, source, target, weight=weight)
            length = nx.dijkstra_path_length(G_bi, source, target, weight=weight)
            hops   = []
            for i in range(len(path) - 1):
                if G.has_edge(path[i], path[i+1]):
                    edge_data = G[path[i]][path[i+1]]
                else:
                    rev_data = G[path[i+1]][path[i]] if G.has_edge(path[i+1], path[i]) else {}
                    edge_data = {
                        "weight": rev_data.get("weight", 12.0),
                        "avg_distance": rev_data.get("avg_distance", 500.0),
                        "avg_delay": rev_data.get("avg_delay", 15.0),
                        "avg_traffic": rev_data.get("avg_traffic", 0.45),
                        "risk_score": rev_data.get("risk_score", 0.5),
                    }
                hops.append({"from": path[i], "to": path[i+1], **edge_data})
            return {"path": path, "total_weight": round(length, 3),
                    "hops": hops, "status": "ok"}
        except Exception:
            return {"path": [], "total_weight": None,
                    "hops": [], "status": "no_path"}
    except nx.NodeNotFound as e:
        return {"path": [], "total_weight": None,
                "hops": [], "status": f"node_not_found: {e}"}


def connected_components(G: nx.DiGraph) -> dict:
    """Return weakly connected components summary."""
    comps = list(nx.weakly_connected_components(G))
    return {
        "num_components":  len(comps),
        "largest_size":    max(len(c) for c in comps),
        "components":      [list(c) for c in comps],
    }


def get_all_shortest_paths(G: nx.DiGraph) -> pd.DataFrame:
    """Return a DataFrame of shortest paths between all hub pairs."""
    rows = []
    hubs = list(G.nodes)
    for src in hubs:
        for dst in hubs:
            if src == dst:
                continue
            res = shortest_path(G, src, dst)
            if res["status"] == "ok":
                rows.append({
                    "source": src, "destination": dst,
                    "path_length": len(res["path"]),
                    "total_time_hrs": res["total_weight"],
                    "path": " → ".join(res["path"]),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/logistics_processed.csv")
    G  = build_graph(df)
    res = shortest_path(G, "Mumbai", "Delhi")
    print(f"Shortest path Mumbai→Delhi: {res}")
    cc  = connected_components(G)
    print(f"Connected components: {cc['num_components']}")
