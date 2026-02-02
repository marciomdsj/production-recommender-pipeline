import pandas as pd
import numpy as np
import pickle
from pathlib import Path

MODELS_DIR = Path("models")
FEATURES_DIR = Path("data/features")

CACHE = {
    "model": None,
    "user_map": None,
    "item_map": None,
    "inv_user_map": None,
    "n_items": 0
}

def load_resources():
    """"
    Loading the model and the maps we built.
    """
    if CACHE["model"] is None:
        with open(MODELS_DIR / "lightfm.pkl", "rb") as f:
            CACHE["model"] = pickle.load(f)
        
        with open(MODELS_DIR / "lightfm_shape.pkl", "rb") as f:
            meta = pickle.load(f)
            CACHE["n_items"] = meta["n_items"]

        user_df = pd.read_csv(FEATURES_DIR / "user_map.csv")
        item_df = pd.read_csv(FEATURES_DIR / "item_map.csv")
        
        CACHE["user_map"] = dict(zip(user_df["user_idx"], user_df["user_id"]))
        CACHE["item_map"] = dict(zip(item_df["item_idx"], item_df["item_id"]))
        CACHE["inv_user_map"] = dict(zip(user_df["user_id"], user_df["user_idx"]))
        
    return CACHE

def recommend_lightfm(user_id, top_k: int = 10):
    res = load_resources()
    
    try:
        target_id = int(user_id)
    except:
        target_id = user_id

    if target_id not in res["inv_user_map"]:
        return pd.DataFrame(columns=["product", "score"])

    user_idx = res["inv_user_map"][target_id]
    
    all_item_indices = np.arange(res["n_items"])
    
    scores = res["model"].predict(
        user_ids=int(user_idx), 
        item_ids=all_item_indices,
        num_threads=1
    )
    
    top_indices = np.argsort(-scores)[:top_k]
    
    recommendations = []
    for idx in top_indices:
        recommendations.append({
            "product": res["item_map"][idx],
            "score": float(scores[idx])
        })
        
    return pd.DataFrame(recommendations)

if __name__ == "__main__":
    print("Testando LightFM...")
    df = recommend_lightfm(user_id="1", top_k=5)
    print(df)