"""
src/graph/route_optimizer.py
=============================
Optimise delivery routes using graph intelligence.

Features:
  - Find fastest route  (min delivery time)
  - Find safest route   (min risk score)
  - Find shortest route (min distance)
  - Multi-hub routing   (travelling through mandatory stops)
  - Route comparison tool
"""

import pandas as pd
import networkx as nx
from src.graph.graph_builder import build_graph, shortest_path


def build_risk_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Clone G but use risk_score as edge weight for 'safest path' search."""
    R = G.copy()
    for u, v, data in R.edges(data=True):
        R[u][v]["weight"] = data.get("risk_score", 0.5)
    return R


def build_distance_graph(G: nx.DiGraph) -> nx.DiGraph:
    """Clone G but use avg_distance as weight for 'shortest distance' search."""
    D = G.copy()
    for u, v, data in D.edges(data=True):
        D[u][v]["weight"] = data.get("avg_distance", 500)
    return D


def compare_routes(G: nx.DiGraph, source: str, destination: str) -> dict:
    """
    Compare fastest / safest / shortest routes between source and destination.
    Returns a dict with three route options and their key metrics.
    """
    R = build_risk_graph(G)
    D = build_distance_graph(G)

    fastest  = shortest_path(G, source, destination, weight="weight")
    safest   = shortest_path(R, source, destination, weight="weight")
    shortest = shortest_path(D, source, destination, weight="weight")

    def enrich(res, G_orig):
        if res["status"] != "ok":
            return res
        path = res["path"]
        total_dist  = sum(G_orig[path[i]][path[i+1]].get("avg_distance", 0)
                          for i in range(len(path)-1))
        total_risk  = sum(G_orig[path[i]][path[i+1]].get("risk_score", 0)
                          for i in range(len(path)-1))
        total_delay = sum(G_orig[path[i]][path[i+1]].get("avg_delay", 0)
                          for i in range(len(path)-1))
        total_time  = sum(G_orig[path[i]][path[i+1]].get("weight", 0)
                          for i in range(len(path)-1))
        res["total_distance_km"] = round(total_dist, 1)
        res["total_risk"]        = round(total_risk, 4)
        res["total_delay_min"]   = round(total_delay, 1)
        res["total_time_hrs"]    = round(total_time, 2)
        return res

    return {
        "source":      source,
        "destination": destination,
        "fastest":     enrich(fastest, G),
        "safest":      enrich(safest,  G),
        "shortest":    enrich(shortest, G),
    }


def multi_stop_route(G: nx.DiGraph, stops: list) -> dict:
    """
    Find the best sequential path through a list of mandatory stops.
    E.g. stops = ["Mumbai", "Pune", "Hyderabad", "Chennai"]
    """
    segments = []
    total_time = 0.0
    full_path  = [stops[0]]

    for i in range(len(stops) - 1):
        seg = shortest_path(G, stops[i], stops[i+1])
        if seg["status"] != "ok":
            return {"status": "infeasible", "segments": segments,
                    "full_path": full_path, "total_time_hrs": None}
        total_time += seg["total_weight"]
        segments.append(seg)
        full_path.extend(seg["path"][1:])   # avoid duplicating junction node

    return {
        "status":        "ok",
        "stops":         stops,
        "full_path":     full_path,
        "segments":      segments,
        "total_time_hrs": round(total_time, 3),
        "num_hops":      len(full_path) - 1,
    }


def top_routes_by_volume(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the busiest routes by shipment count."""
    grp = df.groupby(["source_hub", "destination_hub"]).agg(
        shipment_count   = ("shipment_id",       "count"),
        avg_time_hrs     = ("delivery_time_hrs",  "mean"),
        avg_delay_min    = ("delay_minutes",       "mean"),
        avg_distance_km  = ("route_distance",      "mean"),
    ).reset_index()
    grp["route_label"] = grp["source_hub"] + " → " + grp["destination_hub"]
    return grp.sort_values("shipment_count", ascending=False).head(top_n)


def optimization_recommendations(compare: dict) -> list:
    """Auto-generate routing recommendations from compare_routes output."""
    recs = []
    f, s, sh = compare["fastest"], compare["safest"], compare["shortest"]

    if f["status"] == "ok":
        recs.append(
            f"⚡ Fastest route ({' → '.join(f['path'])}) "
            f"takes {f['total_weight']:.2f} hrs with "
            f"{f.get('total_delay_min',0):.0f} min avg delay."
        )
    if s["status"] == "ok" and s["path"] != f["path"]:
        recs.append(
            f"🛡️  Safest route ({' → '.join(s['path'])}) "
            f"has risk score {s.get('total_risk',0):.3f} — "
            f"recommended for high-value shipments."
        )
    if sh["status"] == "ok":
        recs.append(
            f"📏 Shortest route ({' → '.join(sh['path'])}) "
            f"covers {sh.get('total_distance_km',0):.0f} km — "
            f"best for cost-sensitive Economy shipments."
        )
    return recs


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    df = pd.read_csv("data/processed/logistics_processed.csv")
    G  = build_graph(df)
    res = compare_routes(G, "Mumbai", "Chennai")
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, "→ path:", v.get("path"), "time:", v.get("total_weight"))
