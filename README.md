# 🚀 Production-Ready Recommendation System Pipeline

A scalable, modular Machine Learning pipeline for generating product recommendations using Implicit Feedback. This project implements and compares three distinct collaborative filtering algorithms (ALS, KNN, LightFM) and exposes them via a unified high-performance FastAPI service.

## 📋 Overview

This system is designed to handle the end-to-end recommendation lifecycle: from data ingestion and sparse matrix construction to model training and real-time inference.

### Key Features

- **Multi-Model Architecture**:
  - **ALS (Alternating Least Squares)**: Matrix factorization optimized for implicit feedback (via `implicit`).
  - **LightFM (Hybrid Matrix Factorization)**: Uses WARP loss for optimized ranking, effective for sparse datasets.
  - **User-Based KNN**: Memory-based collaborative filtering with a robust Popularity Fallback strategy for cold-start users.
- **Production API**: Asynchronous FastAPI implementation with singleton resource loading for low-latency inference (<50ms).
- **Optimized Data Structures**: Uses Scipy CSR (Compressed Sparse Row) matrices for memory efficiency.
- **Cold Start Handling**: Graceful fallback mechanisms for new users or sparse interaction histories.

## 🏗 Project Structure
```
Recommender-System/
├── api/
│   └── main.py                 # FastAPI application & entry point
├── data/
│   ├── features/               # Processed sparse matrices & mappings (CSVs)
│   └── processed/              # Cleaned interaction data
├── models/                     # Serialized models (.pkl)
├── pipelines/                  # ETL and Training Scripts
│   ├── build_interactions.py   # Encodes IDs and builds sparse matrices
│   ├── train_als.py            # Trains Implicit ALS
│   ├── train_knn.py            # Trains Scikit-Learn KNN
│   ├── train_lightfm.py        # Trains LightFM with WARP loss
│   └── ingest.py               # Ingests dataset
├── recommenders/               # Inference Logic Layers
│   ├── als.py                  # ALS inference
│   ├── knn.py                  # KNN inference + Fallback logic
│   └── lightfm_recommender.py  # LightFM inference
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🛠️ Installation

**Clone the repository:**
```bash
git clone https://github.com/your-username/recommender-system.git
cd recommender-system
```

**Set up the virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Install dependencies:**
```bash
pip install pandas numpy scipy scikit-learn implicit lightfm fastapi uvicorn
```

## ⚙️ Data Pipeline & Training

To prepare the system for production, execute the pipeline scripts in the following order.

### 1. Data Processing

Transforms raw logs into mapped indices and sparse matrices.
```bash
python pipelines/build_interactions.py
```

### 2. Model Training

Train the individual models. Each script saves the model artifacts to the `models/` directory.

**Train ALS:**
```bash
python pipelines/train_als.py
```

**Train KNN (User-Based):**
```bash
python pipelines/train_knn.py
```

**Train LightFM:**
```bash
python pipelines/train_lightfm.py
```

## 🚀 Running the API

The API loads all trained models into memory upon startup to ensure fast response times.

**Start the Server:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Access the Documentation:**

FastAPI provides automatic interactive documentation.

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 📡 API Endpoints

| Method | Endpoint                         | Description                                                                                              |
|--------|----------------------------------|----------------------------------------------------------------------------------------------------------|
| GET    | `/`                              | Health check and loaded models status.                                                                   |
| GET    | `/recommend/als/{user_id}`       | Recommendations using Matrix Factorization (ALS).                                                        |
| GET    | `/recommend/knn/{user_id}`       | Recommendations using Nearest Neighbors. Includes Popularity Fallback if no neighbors are found.        |
| GET    | `/recommend/lightfm/{user_id}`   | Recommendations using Hybrid LightFM (WARP loss).                                                        |

### Example Request
```bash
curl http://127.0.0.1:8000/recommend/lightfm/1
```

### Example Response
```json
[
  {
    "product": "Blouse",
    "score": 2.3878
  },
  {
    "product": "Jeans",
    "score": 0.6967
  },
  {
    "product": "Sweater",
    "score": 0.5715
  }
]
```

## 🧠 Algorithmic Details

### 1. Implicit ALS
- **Library**: `implicit`
- **Strategy**: Matrix Factorization.
- **Use Case**: Best for general collaborative filtering when dataset size is significant.

### 2. KNN (K-Nearest Neighbors)
- **Library**: `scikit-learn`
- **Metric**: Cosine Similarity.
- **Strategy**: Finds users with similar interaction history.
- **Fallback**: If a user is isolated (sparsity issue), the system automatically fills the recommendation slots with the global Top-N Popular Items to ensure the API never returns an empty list.

### 3. LightFM
- **Library**: `lightfm`
- **Loss Function**: WARP (Weighted Approximate-Rank Pairwise).
- **Use Case**: Optimized for ranking. It performs exceptionally well on sparse datasets where standard Matrix Factorization might struggle.

## 📈 Future Improvements

- Implement A/B Testing logic in the API to route traffic dynamically.
- Add Redis caching for hot users.
- Integrate metadata (item categories) into the LightFM model for Hybrid Filtering.