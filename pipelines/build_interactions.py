# File responsible for building the interactions, which means if the ID have some thing in common with the item.
import pandas as pd
from scipy.sparse import coo_matrix
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")

FEATURES_DIR.mkdir(parents=True, exist_ok=True)

def build_interaction_matrix():
    df = pd.read_csv(PROCESSED_DIR / "interactions.csv")

    # Create categoric index
    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["item_idx"] = df["item_id"].astype("category").cat.codes

    # Map
    user_map = dict(
        enumerate(df["user_id"].astype("category").cat.categories)
    )
    item_map = dict(
        enumerate(df["item_id"].astype("category").cat.categories)
    )

    # Creating sparse matrix
    matrix = coo_matrix(
        (
            df["interaction"].values,
            (df["user_idx"].values, df["item_idx"].values)
        )
    )

    return df, matrix, user_map, item_map

if __name__ == "__main__":
    df_idx, matrix, user_map, item_map = build_interaction_matrix()

    print("Matriz de interações:")
    print("Shape:", matrix.shape)
    print("Sparsity:",
          1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]))

    # Persistance
    df_idx.to_csv(FEATURES_DIR / "interactions_indexed.csv", index=False)

    pd.DataFrame(
        list(user_map.items()),
        columns=["user_idx", "user_id"]
    ).to_csv(FEATURES_DIR / "user_map.csv", index=False)
    
    pd.DataFrame(
        list(item_map.items()),
        columns=["item_idx", "item_id"]
    ).to_csv(FEATURES_DIR / "item_map.csv", index=False)