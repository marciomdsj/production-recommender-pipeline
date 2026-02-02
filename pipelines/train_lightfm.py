# File responsible for training the LightFM model with the dataset.
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import coo_matrix
from lightfm import LightFM

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(FEATURES_DIR / "interactions_indexed.csv")
    
    n_users = df["user_idx"].max() + 1
    n_items = df["item_idx"].max() + 1
    
    matrix = coo_matrix(
        (np.ones(df.shape[0]), (df["user_idx"], df["item_idx"])), 
        shape=(n_users, n_items)
    )
    
    return matrix, n_users, n_items

def train_lightfm(matrix):
    model = LightFM(
        learning_rate=0.05,
        loss='warp',
        no_components=30,
        random_state=42
    )
    
    print("Iniciando treino com WARP loss...")
    model.fit(matrix, epochs=20, num_threads=2)
    
    return model

if __name__ == "__main__":
    matrix, n_users, n_items = load_data()
    print(f"Matriz carregada: {n_users} usuários x {n_items} itens.")

    model = train_lightfm(matrix)

    with open(MODELS_DIR / "lightfm.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(MODELS_DIR / "lightfm_shape.pkl", "wb") as f:
        pickle.dump({"n_users": n_users, "n_items": n_items}, f)

    print("Modelo LightFM treinado e salvo com sucesso.")