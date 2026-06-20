"""
main.py
========
End-to-end runner: generates data -> preprocesses -> trains models -> validates graph.
Run this ONCE before launching the Streamlit dashboard.

Usage:
    python main.py
    streamlit run src/dashboard/app.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print('='*60)

def main():
    t0 = time.time()

    banner("STEP 1/5 -- Checking Raw Dataset")
    print("  Using local dataset at data/raw/delivery_data.csv")

    banner("STEP 2/5 -- Preprocessing Pipeline")
    from src.preprocessing.preprocess import run_pipeline
    df = run_pipeline()
    print(f"  Processed: {df.shape[0]} rows x {df.shape[1]} cols")

    banner("STEP 3/5 -- Training ML Models")
    from src.models.train_model import train_all
    out = train_all()
    best = out["best_model_name"]
    best_r2 = next(r["R2"] for r in out["results"] if r["model_name"]==best)
    n_models = out["n_models_trained"]
    print(f"  {n_models} models trained")
    print(f"  Best model: {best}  |  R2={best_r2:.4f}")

    banner("STEP 4/5 -- Graph Construction & Validation")
    import pandas as pd
    from src.graph.graph_builder import build_graph, shortest_path
    from src.graph.bottleneck_analysis import compute_centrality, detect_risky_routes
    df_proc = pd.read_csv("data/processed/logistics_processed.csv")
    G   = build_graph(df_proc)
    cdf = compute_centrality(G)
    rr  = detect_risky_routes(df_proc, top_n=5)
    res = shortest_path(G, "Mumbai", "Delhi")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Top bottleneck hub: {cdf.iloc[0]['hub']} (score={cdf.iloc[0]['bottleneck_score']:.3f})")
    if res["status"] == "ok":
        print(f"  Mumbai->Delhi path: {' -> '.join(res['path'])} ({res['total_weight']:.2f} hrs)")

    banner("STEP 5/5 -- Analytics Validation")
    from src.analytics.sla_analytics import breach_by_priority, breach_root_cause_analysis
    from src.analytics.ftl_carting import ftl_vs_carting_analysis
    sla_df = breach_by_priority(df_proc)
    print(f"  SLA Analysis: {len(sla_df)} priority tiers analyzed")
    if "vehicle_type" in df_proc.columns:
        ftl_df = ftl_vs_carting_analysis(df_proc)
        print(f"  FTL/Carting: {len(ftl_df)} vehicle types compared")
    root_causes = breach_root_cause_analysis(df_proc)
    print(f"  Root Causes: {len(root_causes)} insights generated")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE  ({elapsed:.1f}s)")
    print(f"  Launch dashboard:  streamlit run src/dashboard/app.py")
    print('='*60)

if __name__ == "__main__":
    main()
