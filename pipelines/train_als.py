import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
from implicit.als import AlternatingLeastSquares
from pathlib import Path
import pickle

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_interactions():
    df = pd.read_csv(FEATURES_DIR / "interactions_indexed.csv")
 
    matrix = coo_matrix(
        (
            df["interaction"].values,
            (df["user_idx"].values, df["item_idx"].values)
        )
    ).tocsr()

    return matrix

def train_als(matrix):
    model = AlternatingLeastSquares(
        factors=50,
        regularization=0.1,
        iterations=20,
        random_state=42
    )
    model.fit(matrix.T)

    return model

if __name__ == "__main__":
    matrix = load_interactions()

    sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    print(f"Sparsity: {sparsity:.4f}")
    print("Matrix shape:", matrix.shape)

    als_model = train_als(matrix)

    with open(MODELS_DIR / "als_model.pkl", "wb") as f:
        pickle.dump(als_model, f)

    print("ALS model trained and saved.")