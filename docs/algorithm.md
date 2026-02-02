# 🧠 Machine Learning Models Documentation

This document provides a technical overview of the algorithms implemented in the Recommendation Pipeline. It details how they work, their inputs, hyperparameters, and their specific behavior regarding our dataset.

## 1. Alternating Least Squares (ALS)

### 📖 Conceptual Overview

ALS is a Matrix Factorization algorithm designed specifically for **Implicit Feedback** datasets (e.g., clicks, views, purchases, rather than star ratings).

It works by decomposing the large, sparse User-Item interaction matrix into two smaller, dense matrices:

- **User Factors**: Represents users' preferences for hidden features.
- **Item Factors**: Represents items' characteristics regarding those features.

These "hidden features" are called **Latent Factors**. For example, even if the dataset doesn't explicitly label a product as "Luxury" or "Budget," the model mathematically infers these characteristics based on which users buy which items.

### ⚙️ Inputs & Hyperparameters

- **Input Data**: Interaction Matrix (Sparse CSR).
- **factors** (e.g., 50): The number of latent dimensions.
  - Too low: Underfitting (model is too simple).
  - Too high: Overfitting (model memorizes the data).
- **iterations** (e.g., 20): How many times the algorithm alternates between fixing user factors and solving for item factors (and vice versa) to minimize error.
- **regularization** (e.g., 0.1): A penalty term to prevent the model from becoming "too confident" about a user's preference based on very few interactions.

### 🚀 When to use

- **Best for**: Large datasets with implicit feedback where discovery of hidden patterns is needed.
- **Pros**: Highly scalable and parallelizable.
- **Cons**: Pure ALS cannot handle Cold Start (new users/items with no interactions have no vectors).

## 2. K-Nearest Neighbors (KNN) - User-Based

### 📖 Conceptual Overview

KNN is a memory-based algorithm that follows the logic: *"Users who bought similar items in the past will buy similar items in the future."*

It calculates the distance (typically **Cosine Similarity**) between the target user vector and all other user vectors to find the "Top K" most similar neighbors. The recommendation is a weighted average of what those neighbors bought.

### ⚙️ Inputs & Hyperparameters

- **Input Data**: Interaction Matrix (Sparse CSR).
- **n_neighbors** (e.g., 20): The number of similar users to consider.
- **metric** (Cosine): Measures the angle between vectors. Useful for high-dimensional sparse data.

### ⚠️ The "Sparsity Paradox" (Dataset Limitation)

**Why pure KNN failed in our initial tests:**

Our dataset has high sparsity and a "One-Hit Wonder" characteristic, where many users bought only one specific item (e.g., Item A) and nothing else.

**The Scenario**: User 1 bought Blouse. User 2 also bought Blouse.

1. **The Neighbor**: The model correctly identifies User 2 as a "perfect neighbor" (Distance = 0).
2. **The Calculation**: The model looks at what else User 2 bought to recommend it to User 1.
3. **The Failure**: User 2 only bought the Blouse. There are no other items to recommend.
4. **Result**: The collaborative score for all other items is 0.0.

### ✅ Our Solution: Hybrid Fallback

To mitigate this, our pipeline implements a **Popularity Fallback**.

- **Primary**: Try to find collaborative neighbors.
- **Secondary**: If the neighbors provide no new information (score 0 or empty), fill the recommendation list with the Global Top Selling Items.

## 3. LightFM (Hybrid Matrix Factorization)

### 📖 Conceptual Overview

LightFM is a hybrid model that represents users and items as linear combinations of their content features (latent representations). Unlike basic Matrix Factorization, it allows the model to "borrow" information across users.

Crucially, we utilize the **WARP (Weighted Approximate-Rank Pairwise)** loss function.

- **Standard Loss**: Tries to predict the exact value (0 or 1).
- **WARP Loss**: Tries to predict the correct ranking order. It penalizes the model heavily if a positive item (one the user bought) is ranked lower than a negative item.

### ⚙️ Inputs & Hyperparameters

- **Input Data**: Interaction Matrix (COO format).
- **loss='warp'**: Optimizes for the top of the recommendation list (Precision@K).
- **no_components** (e.g., 30): Dimensionality of the feature vectors.
- **learning_rate**: How fast the model updates weights during Gradient Descent.

### 🚀 When to use

- **Best for**: Sparse datasets (like ours) and scenarios where ranking accuracy is more important than rating prediction.
- **Why it worked better**: Even with limited data, WARP loss forces the model to push relevant items to the top of the list more aggressively than ALS or KNN, resulting in non-zero scores even for users with limited history.