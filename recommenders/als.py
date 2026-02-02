import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import csr_matrix

MODELS_DIR = Path("models")
FEATURES_DIR = Path("data/features")

CACHE = {
    "model": None,
    "matrix": None,
    "user_map": None,
    "item_map": None,
    "inv_user_map": None
}

def load_resources():
    if CACHE["model"] is None:
        with open(MODELS_DIR / "als_model.pkl", "rb") as f:
            model = pickle.load(f)
            CACHE["model"] = model
        
        user_df = pd.read_csv(FEATURES_DIR / "user_map.csv")
        item_df = pd.read_csv(FEATURES_DIR / "item_map.csv")
        df_i = pd.read_csv(FEATURES_DIR / "interactions_indexed.csv")
        
        CACHE["user_map"] = dict(zip(user_df["user_idx"], user_df["user_id"]))
        CACHE["item_map"] = dict(zip(item_df["item_idx"], item_df["item_id"]))
        CACHE["inv_user_map"] = dict(zip(user_df["user_id"], user_df["user_idx"]))

        num_users = model.user_factors.shape[0]
        num_items = model.item_factors.shape[0]

        CACHE["matrix"] = csr_matrix(
            (df_i["interaction"].astype(float), 
             (df_i["user_idx"], df_i["item_idx"])),
            shape=(num_users, num_items)
        )
    return CACHE

def recommend_als(user_id, top_k: int = 10):
    res = load_resources()
    
    try:
        target_id = int(user_id)
    except:
        target_id = user_id

    if target_id not in res["inv_user_map"]:
        return pd.DataFrame(columns=["product", "score"])

    user_idx = res["inv_user_map"][target_id]

    user_row = res["matrix"][user_idx]

    ids, scores = res["model"].recommend(
        userid=user_idx,
        user_items=user_row,
        N=top_k,
        filter_already_liked_items=True
    )

    recommendations = [
        {"product": res["item_map"][idx], "score": float(score)}
        for idx, score in zip(ids, scores)
    ]
    
    return pd.DataFrame(recommendations)

if __name__ == "__main__":
    try:
        print("Iniciando busca...")
        df = recommend_als(user_id="13", top_k=5)
        print(df)
    except Exception as e:
        import traceback
        traceback.print_exc() # Isso vai mostrar EXATAMENTE onde quebrou