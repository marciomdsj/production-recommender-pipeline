from fastapi import FastAPI, HTTPException
import sys
from pathlib import Path
import uvicorn

# Adds the project root to the system path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import recommendation functions from each file
from recommenders.als import recommend_als, load_resources as load_als
from recommenders.knn import recommend_knn, load_resources as load_knn
from recommenders.lightfm_recommender import recommend_lightfm, load_resources as load_lightfm

app = FastAPI(
    title="Recommendation System - Prod Pipeline",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Loads all models into RAM when starting the API.
    This prevents latency on the user's first request.
    """
    print("🚀 Starting API...")
    
    print("1. Loading ALS...")
    load_als()
    
    print("2. Loading KNN...")
    load_knn()
    
    print("3. Loading LightFM...")
    load_lightfm()
    
    print("✅ All models loaded and ready!")

@app.get("/")
def health_check():
    return {"status": "online", "models": ["als", "knn", "lightfm"]}

# --- ALS Endpoint ---
@app.get("/recommend/als/{user_id}")
def get_als(user_id: str, k: int = 10):
    try:
        df = recommend_als(user_id, top_k=k)
        if df.empty:
            return {"message": "No recommendations found", "data": []}
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in ALS: {str(e)}")

# --- KNN Endpoint (User-Based) ---
@app.get("/recommend/knn/{user_id}")
def get_knn(user_id: str, k: int = 10):
    try:
        # KNN has the option remove_seen. In APIs, we usually keep it True.
        df = recommend_knn(user_id, top_k=k, remove_seen=True)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in KNN: {str(e)}")

# --- LightFM Endpoint (Hybrid/Latent) ---
@app.get("/recommend/lightfm/{user_id}")
def get_lightfm(user_id: str, k: int = 10):
    try:
        df = recommend_lightfm(user_id, top_k=k)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in LightFM: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8010, reload=True)