import pandas as pd
import numpy as np
import pickle
from pathlib import Path

MODELS_DIR = Path("models")
FEATURES_DIR = Path("data/features")

CACHE = {
    "model": None,
    "matrix": None,
    "user_map": None,
    "item_map": None,
    "inv_user_map": None,
    "popular_items": None
}

def load_resources():
    if CACHE["model"] is None:
        with open(MODELS_DIR / "knn_model.pkl", "rb") as f:
            CACHE["model"] = pickle.load(f)
        
        with open(MODELS_DIR / "knn_matrix.pkl", "rb") as f:
            CACHE["matrix"] = pickle.load(f)

        user_df = pd.read_csv(FEATURES_DIR / "user_map.csv")
        item_df = pd.read_csv(FEATURES_DIR / "item_map.csv")
        
        CACHE["user_map"] = dict(zip(user_df["user_idx"], user_df["user_id"]))
        CACHE["item_map"] = dict(zip(item_df["item_idx"], item_df["item_id"]))
        CACHE["inv_user_map"] = dict(zip(user_df["user_id"], user_df["user_idx"]))

        activity = np.array(CACHE["matrix"].sum(axis=0)).flatten()
        CACHE["popular_items"] = np.argsort(activity)[::-1]
    
    return CACHE

def recommend_knn(user_id, top_k: int = 10, remove_seen: bool = True):
    """
    Recommend based on nearest neighbors (User-Based). 
    """
    res = load_resources()
    
    try:
        target_id = int(user_id)
    except:
        target_id = user_id

    if target_id not in res["inv_user_map"]:
        return _get_popular_fallback(res, top_k, strategy="cold_start")

    user_idx = res["inv_user_map"][target_id]
    user_vector = res["matrix"][user_idx]

    distances, indices = res["model"].kneighbors(user_vector, n_neighbors=20)
    
    neighbor_indices = indices.flatten()
    neighbor_distances = distances.flatten()

    mask = neighbor_indices != user_idx
    neighbor_indices = neighbor_indices[mask]
    neighbor_distances = neighbor_distances[mask]

    similarities = 1 - neighbor_distances
    neighbor_matrix = res["matrix"][neighbor_indices].toarray()
    
    item_scores = similarities.dot(neighbor_matrix)

    if remove_seen:
        user_seen = user_vector.toarray().flatten()
        item_scores[user_seen > 0] = 0

    top_indices = np.argsort(item_scores)[::-1]
    
    recommendations = []
    seen_ids = set()

    for idx in top_indices:
        if len(recommendations) >= top_k:
            break
        
        score = item_scores[idx]
        if score > 0.00001:
            recommendations.append({
                "product": res["item_map"].get(idx, f"Item {idx}"),
                "score": float(score),
                "strategy": "knn_collaborative"
            })
            seen_ids.add(idx)

    if len(recommendations) < top_k:
        needed = top_k - len(recommendations)
        pop_indices = res["popular_items"]
        
        count = 0
        for idx in pop_indices:
            if count >= needed: break

            if idx not in seen_ids:
                if remove_seen and user_vector[0, idx] > 0:
                    continue

                recommendations.append({
                    "product": res["item_map"].get(idx),
                    "score": 0.0,
                    "strategy": "popular_fallback"
                })
                count += 1
                
    return pd.DataFrame(recommendations)

def _get_popular_fallback(res, k, strategy):
    """Helper to return popular items"""
    pop_indices = res["popular_items"][:k]
    return pd.DataFrame([
        {
            "product": res["item_map"].get(i),
            "score": 0.0, 
            "strategy": strategy
        }
        for i in pop_indices
    ])

if __name__ == "__main__":
    print(recommend_knn(user_id="1", top_k=5))