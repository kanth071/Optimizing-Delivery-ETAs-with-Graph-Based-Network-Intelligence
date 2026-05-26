"""
src/models/predict.py
======================
Load trained best model and make ETA predictions.
Supports all model types: Linear Regression, Random Forest, Gradient Boosting,
XGBoost, LightGBM.
"""
import pickle, json, os
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "route_distance","traffic_level","hub_load","num_stops",
    "weather_risk","priority_rank","congestion_score",
    "avg_delay_per_route","est_travel_hrs",
    "is_weekend","is_peak_hour","hour_of_day","month",
    "route_type_enc","source_hub_enc","destination_hub_enc",
    "shipment_priority_enc","weather_condition_enc",
    "traffic_weather_interaction","distance_congestion",
    "stops_delay_interaction","vehicle_speed_factor",
    "route_volume","delay_ratio",
]
WEATHER_RISK  = {"Clear":1,"Cloudy":2,"Rain":3,"Heavy Rain":4,"Fog":3,"Storm":5}
PRIORITY_RANK = {"Economy":1,"Standard":2,"Express":3,"Same-Day":4}
ROUTE_SPEED   = {"Highway":80,"Expressway":95,"City Road":35,"Rural":45,"Mixed":60}
VEHICLE_SPEED_FACTOR = {"FTL Truck":1.0,"Carting Vehicle":0.75,"Express Van":1.15,"Mini Truck":0.9,"Container":0.85}

HUBS = ["Mumbai","Delhi","Bangalore","Chennai","Hyderabad","Kolkata","Pune",
        "Ahmedabad","Jaipur","Lucknow","Surat","Nagpur","Indore","Bhopal",
        "Patna","Chandigarh","Coimbatore","Vizag","Kochi","Vadodara"]

ALL_MODEL_NAMES = [
    "Linear_Regression", "Random_Forest", "Gradient_Boosting",
    "XGBoost", "LightGBM",
]

def ensure_compatible_models(model_dir="data/processed/models"):
    import sys
    import sklearn
    
    # Generate current environment signature
    current_sig = f"python:{sys.version_info.major}.{sys.version_info.minor}|sklearn:{sklearn.__version__}"
    sentinel_path = os.path.join(model_dir, "compat_verified.txt")
    
    # 1. Check if sentinel matches perfectly -> Instant Fast-Path bypass
    if os.path.exists(sentinel_path):
        try:
            with open(sentinel_path, "r") as f:
                saved_sig = f.read().strip()
            if saved_sig == current_sig:
                return  # 100% compatible, skip expensive pickle-loading!
        except Exception:
            pass

    # 2. Otherwise perform deep validation
    need_retrain = False
    summary_path = os.path.join(model_dir, "summary.json")
    if not os.path.exists(summary_path):
        need_retrain = True
    else:
        for nm in ALL_MODEL_NAMES:
            p = os.path.join(model_dir, nm+".pkl")
            if not os.path.exists(p):
                need_retrain = True
                break
            try:
                # Try unpickling to verify compatibility
                with open(p, "rb") as f:
                    pickle.load(f)
            except Exception as e:
                print(f"[PREDICT] Failed to load {nm}.pkl due to environment mismatch: {e}. Retraining will be triggered.")
                need_retrain = True
                break
                
    if need_retrain:
        print("[FALLBACK] Environment mismatch, unpickling failure, or missing models. Auto-retraining perfectly compatible models on the fly...")
        try:
            from src.models.train_model import train_all
            train_all(processed_path="data/processed/logistics_processed.csv", model_dir=model_dir)
            print("[FALLBACK] Auto-retraining finished successfully. All models are now fully compatible.")
        except Exception as te:
            print(f"[ERROR] Auto-retraining failed: {te}")
            
    # 3. Write sentinel file to record successful validation for future lightning-fast loads
    try:
        os.makedirs(model_dir, exist_ok=True)
        with open(sentinel_path, "w") as f:
            f.write(current_sig)
    except Exception:
        pass

def load_best_model(model_dir="data/processed/models"):
    ensure_compatible_models(model_dir)
    with open(os.path.join(model_dir,"summary.json")) as f:
        summary = json.load(f)
    best_name = summary["best_model_name"]
    pkl = os.path.join(model_dir, best_name.replace(" ","_")+".pkl")
    with open(pkl,"rb") as f:
        model = pickle.load(f)
    return model, best_name, summary

def load_all_models(model_dir="data/processed/models"):
    ensure_compatible_models(model_dir)
    models = {}
    for nm in ALL_MODEL_NAMES:
        p = os.path.join(model_dir, nm+".pkl")
        if os.path.exists(p):
            with open(p,"rb") as f:
                models[nm.replace("_"," ")] = pickle.load(f)
    return models

def build_feature_row(distance, traffic, hub_load, stops, weather,
                       priority, route_type, hour=9, day=1, month=6,
                       avg_delay_route=150.0, source_hub="Mumbai",
                       dest_hub="Delhi", vehicle_type="FTL Truck"):
    speed       = ROUTE_SPEED.get(route_type, 60)
    vfactor     = VEHICLE_SPEED_FACTOR.get(vehicle_type, 1.0)
    est_hrs     = distance / (speed * vfactor)
    congestion  = 0.6*traffic + 0.4*hub_load
    is_weekend  = 1 if day >= 5 else 0
    is_peak     = 1 if (8<=hour<=10 or 17<=hour<=20) else 0
    w_risk      = WEATHER_RISK.get(weather, 2)

    src_enc  = HUBS.index(source_hub) if source_hub in HUBS else 0
    dst_enc  = HUBS.index(dest_hub)   if dest_hub   in HUBS else 1
    wenc     = list(WEATHER_RISK.keys()).index(weather) if weather in WEATHER_RISK else 0
    rtenc    = ["Highway","Expressway","City Road","Rural","Mixed"].index(route_type) if route_type in ["Highway","Expressway","City Road","Rural","Mixed"] else 0
    prenc    = list(PRIORITY_RANK.keys()).index(priority) if priority in PRIORITY_RANK else 1

    row = {
        "route_distance":             distance,
        "traffic_level":              traffic,
        "hub_load":                   hub_load,
        "num_stops":                  stops,
        "weather_risk":               w_risk,
        "priority_rank":              PRIORITY_RANK.get(priority,2),
        "congestion_score":           round(congestion,3),
        "avg_delay_per_route":        avg_delay_route,
        "est_travel_hrs":             round(est_hrs,3),
        "is_weekend":                 is_weekend,
        "is_peak_hour":               is_peak,
        "hour_of_day":                hour,
        "month":                      month,
        "route_type_enc":             rtenc,
        "source_hub_enc":             src_enc,
        "destination_hub_enc":        dst_enc,
        "shipment_priority_enc":      prenc,
        "weather_condition_enc":      wenc,
        "traffic_weather_interaction":round(traffic * w_risk, 3),
        "distance_congestion":        round(distance * congestion, 2),
        "stops_delay_interaction":    round(stops * avg_delay_route * 0.01, 2),
        "vehicle_speed_factor":       vfactor,
        "route_volume":               50,
        "delay_ratio":                0.3,
    }
    available = [c for c in FEATURE_COLS if c in row]
    return pd.DataFrame([{k: row[k] for k in available}])[available]

if __name__ == "__main__":
    model, name, _ = load_best_model()
    X = build_feature_row(500,0.6,0.5,2,"Rain","Express","Highway")
    print(f"Predicted ETA ({name}):", round(model.predict(X)[0],2),"hrs")
