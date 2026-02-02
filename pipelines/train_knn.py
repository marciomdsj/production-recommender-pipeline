# File responsible for training the KNN model with the dataset.
import pandas as pd
import pickle
from pathlib import Path
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    df = pd.read_csv(FEATURES_DIR / "interactions_indexed.csv")
    n_users = df["user_idx"].max() + 1
    n_items = df["item_idx"].max() + 1
    matrix = coo_matrix(
        (df["interaction"].values, (df["user_idx"].values, df["item_idx"].values)),
        shape=(n_users, n_items)
    ).tocsr()
    
    return matrix

def train_knn(matrix):
    # Using Cosine as metric
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model.fit(matrix)
    return model

if __name__ == "__main__":
    print("Carregando dados...")
    matrix = load_data()
    
    print(f"Treinando KNN com matriz shape: {matrix.shape}...")
    knn_model = train_knn(matrix)

    with open(MODELS_DIR / "knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)
     
    with open(MODELS_DIR / "knn_matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)

    print("Modelo KNN e Matriz de referência salvos.")