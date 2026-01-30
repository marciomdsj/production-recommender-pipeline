# File responsable to clean the dataset, make the dataset ready to recommending and ranking
import pandas as pd
from ingest import load_data
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def clean_dataset():
    df = load_data()

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    # Rename key-columns
    df = df.rename(columns={
        "customer_id": "user_id",
        "item_purchased": "item_id",
        "category": "category",
        "location": "location"
    })

    # Create implicit feedback
    df["interaction"] = 1

    # Relevant columns
    df_rec = df[[
        "user_id",
        "item_id",
        "category",
        "location",
        "interaction"
    ]].copy()

    return df_rec

if __name__ == "__main__":
    df_rec = clean_dataset()
    print(df_rec.head())
    print("\nShape:", df_rec.shape)
    df_rec.to_csv(
        PROCESSED_DIR / "interactions.csv",
        index=False
    )