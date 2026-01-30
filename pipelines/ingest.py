import kagglehub
import pandas as pd
from pathlib import Path

def download_dataset():
    path = Path(
        kagglehub.dataset_download(
            "wardabilal/customer-shopping-behaviour-analysis"
        )
    )
    return path

def load_data():
    path = download_dataset()

    print("\nArquivos encontrados no dataset:")
    for file in path.iterdir():
        print(" -", file.name)

    # Take the first csv file
    csv_files = list(path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado no dataset")

    df = pd.read_csv(csv_files[0])
    return df

if __name__ == "__main__":
    df = load_data()
    print("\nPreview dos dados:")
    print(df.head())
    print("\nColunas:")
    print(df.columns.tolist())
