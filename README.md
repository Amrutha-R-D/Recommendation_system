# Netflix Prize Movie Recommendation System
This repository contains the complete data engineering pipeline (pipeline.py) designed to ingest, parse, memory-optimize, and partition the massive Netflix Prize Dataset (over 100 million rows). 

The pipeline converts raw, unstructured text files into highly compressed, machine-learning-ready .parquet files using efficient data downcasting to minimize RAM footprints.

A decoupled, production-grade recommendation engine framework built to handle the Netflix Prize dataset. This project implements and evaluates three distinct collaborative filtering architectures—Truncated SVD, Temporal Matrix Factorization (Temporal-MF), and Deep Neural Collaborative Filtering (Neural-CF)—to predict user ratings and generate personalized Top-K recommendations.

Project Architecture & Pipelines
This repository is strictly modularized to separate data engineering, model definition, evaluation metrics, and live inference:

```text
Recommendation_system/
│
├── src/
│   ├── __init__.py             # Python package initializer token
│   ├── data_pipeline.py        # Sub-sampling, continuous indexing, & temporal engineering
│   ├── models.py               # Structural blueprints for SVD, Temporal-MF, & Neural-CF
│   ├── evaluation.py           # Evaluation suites (RMSE, MAE) and ranking metric loops
│   └── recommender.py          # Vector cosine similarities and cold-start fallbacks
│
├── eda.ipynb                   # Exploratory Data Analysis notebook
├── pipeline.py                 # Stage 1 Ingestion: Raw Netflix data ➔ Compressed Parquet
├── main.py                     # Stage 2 Orchestration: Master execution entry point
└── requirements.txt            # Explicit software package dependencies
```
---

## Getting Started & Replication

Follow these steps to set up the environment and execute the pipeline locally or in a cloud environment:

### 1. Environment Setup
Install all required package dependencies using pip:

pip install -r requirements.txt

### 2. Stage 1: Data Processing Ingestion
Before running the models, use the pre-processing script to transform the raw Netflix text files into optimized, high-performance .parquet data partitions:

python pipeline.py

Note: Ensure your output Parquet files (df_train.parquet, df_val.parquet, etc.) are located in your targeted data directory (e.g., ./data/).

### 3. Stage 2: Execution Orchestration
Run the master pipeline script to process the data splits, train all three models sequentially, calculate validation metrics, and print inference lookups:

python main.py

---

## Model Pipeline Specs & Hyperparameters

The orchestration pipeline executes under the following specific configuration baselines:

| Algorithm | Key Hyperparameters & Latent Factors | Optimization Target |
| :--- | :--- | :--- |
| **Truncated SVD** | Factors: 50 | Algebraic Matrix Reconstruction |
| **Temporal-MF** | Factors: 50 \| Learning Rate: 0.005 \| Reg: 0.02 \| Epochs: 10 \| Time Bins: 30 | Vectorized Mini-Batch Stochastic Gradient Descent |
| **Neural-CF** | Embedding Dim: 32 \| MLP Layers: [64, 32] \| Learning Rate: 0.001 \| Epochs: 5 | Deep Learning Backpropagation via Adam Optimizer |

---

## Evaluation Suite Metrics

The system comprehensively measures performance across two standard recommendation dimensions:
1. Rating Prediction Quality (Lower is better): Tracks Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) against sparse validation sets.
2. Ranking & Retrieval Quality (Higher is better): Samples users to compute Mean Average Precision (MAP@10), Precision@10, Recall@10, Normalized Discounted Cumulative Gain (NDCG@10), and Hit Rate. It also tracks Catalog Coverage to measure item-recommendation diversity.

Upon a successful run, the execution pipeline automatically outputs a performance dashboard visual saved as model_comparison.png along with cross-architecture comparison grids.

---

## Inference & Fallback Modules

* Item-Item Collaborative Filtering: Uses the learned latent factor representations to extract vector cosine distances, allowing the engine to return highly accurate "Because you watched Movie X, our system recommends..." sequences.
* Cold-Start Engine: Simulates a brand new user with absolute zero historical interactions. The module leverages global user review volume thresholds (>5,000 ratings) combined with historical item biases to recommend universally acclaimed fallback titles.ases to recommend universally acclaimed fallback titles.
