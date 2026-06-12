# main.py
"""
Netflix Prize Recommendation Engine Gateway
-------------------------------------------
Orchestrates your data processing modules, training iterations, validation evaluation
suites, and inference generation outputs.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Import our decoupled modular code packages safely
from src.data_pipeline import load_and_prepare_data
from src.models import SVDRecommender, TemporalMFRecommender, NeuralCFRecommender
from src.evaluation import evaluate_model
from src.recommender import get_item_similarities, run_cold_start_module

# -------------------------------------------------------------------------
# Hyperparameter Declarations & Context Configurations
# -------------------------------------------------------------------------
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DATA_DIR = '/content/drive/MyDrive/data/'
SAMPLE_USERS = 50000         
MIN_USER_RATINGS = 20        
MIN_MOVIE_RATINGS = 50       

SVD_FACTORS = 50             
TMF_FACTORS = 50             
TMF_LR = 0.005               
TMF_REG = 0.02               
TMF_EPOCHS = 10            
TMF_TIME_BINS = 30           
TMF_BATCH_SIZE = 8192        

NCX_EMBED_DIM = 32           # Re-mapped configuration token referencing NCF_EMBED_DIM
NCF_EMBED_DIM = 32           
NCF_HIDDEN = [64, 32]        
NCF_LR = 0.001               
NCF_EPOCHS = 5            
NCF_BATCH_SIZE = 4096      

TOP_K = 10                   
RELEVANCE_THRESHOLD = 3.5    
MAP_SAMPLE_USERS = 3000      
RANDOM_STATE = 42

print("Configuration loaded.")

def main():
    # 1. Load and Prepare Data
    data = load_and_prepare_data(DATA_DIR, SAMPLE_USERS, MIN_USER_RATINGS, TMF_TIME_BINS)
    
    # 2. Model Training Pipeline
    print("=" * 70)
    print("  MODEL TRAINING")
    print("=" * 70)
    train_data = data['train_data']
    n_users = data['n_users']
    n_movies = data['n_movies']
    global_mean = data['global_mean']

    svd_model = SVDRecommender(n_factors=SVD_FACTORS)
    svd_model.fit(train_data, n_users, n_movies, global_mean)

    tmf_model = TemporalMFRecommender(
        n_factors=TMF_FACTORS, lr=TMF_LR, reg=TMF_REG,
        n_epochs=TMF_EPOCHS, n_time_bins=TMF_TIME_BINS, batch_size=TMF_BATCH_SIZE, random_state=RANDOM_STATE
    )
    tmf_model.fit(train_data, n_users, n_movies, global_mean)

    ncf_model = NeuralCFRecommender(
        n_users=n_users, n_movies=n_movies,
        embed_dim=NCF_EMBED_DIM, hidden_dims=NCF_HIDDEN,
        lr=NCF_LR, n_epochs=NCF_EPOCHS, batch_size=NCF_BATCH_SIZE, device=device
    )
    ncf_model.fit(train_data, n_users, n_movies, global_mean)

    models = {'SVD': svd_model, 'Temporal-MF': tmf_model, 'Neural-CF': ncf_model}
    print("\nAll models trained!")

    # 3. Model Evaluation Pipeline
    print("=" * 70)
    print("  MODEL EVALUATION")
    print("=" * 70)
    validation_set = data['val_data']
    train_user_items = data['train_user_items']
    results = {}

    for name, model in models.items():
        results[name] = evaluate_model(model, validation_set, train_user_items, n_movies, name, 
                                       TOP_K, RELEVANCE_THRESHOLD, MAP_SAMPLE_USERS, RANDOM_STATE)

    # 4. Construct Comparison Table
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)
    metrics_list = ['RMSE', 'MAE', 'MAP@10', 'Precision@10', 'Recall@10', 'NDCG@10', 'HitRate@10', 'Coverage']
    model_names = list(results.keys())
    
    header = f"  {'Metric':>15s}"
    for name in model_names:
        header += f"  {name:>12s}"
    header += f"  {'Best':>12s}"
    print(f"\n{header}")
    print("  " + "-" * (15 + 14 * (len(model_names) + 1)))
    
    best_scores = defaultdict(int)
    for metric in metrics_list:
        row = f"  {metric:>15s}"
        values = [results[name][metric] for name in model_names]
        for v in values:
            row += f"  {v:>12.6f}"
        best_idx = np.argmin(values) if metric in ['RMSE', 'MAE'] else np.argmax(values)
        best_name = model_names[best_idx]
        best_scores[best_name] += 1
        row += f"  {best_name:>12s}"
        print(row)
        
    overall_winner = max(best_scores, key=best_scores.get)
    print(f"\n  Winner Summary:")
    for name, score in sorted(best_scores.items(), key=lambda x: -x[1]):
        print(f"    {name}: {score}/{len(metrics_list)} metrics")
    print(f"\n  >>> BEST OVERALL MODEL: {overall_winner} <<<")

    # 5. Top-K Evaluation Output Visualizations
    print("\n" + "=" * 70)
    print("   TOP-K RECOMMENDATION EXAMPLES")
    print("=" * 70)
    df_movies = data['df_movies']
    movie_id_to_title = {}
    for _, row in df_movies.iterrows():
        mid = int(row['Movie_ID'])
        title = str(row['Title'])
        year = row.get('Year', '')
        movie_id_to_title[mid] = f"{title} ({int(year)})" if pd.notna(year) else title
        
    idx_to_movie = data['idx_to_movie']
    idx_to_user = data['idx_to_user']
    best_model = models[overall_winner]
    evaluation_dataset = data['val_data']
    
    eval_user_counts = evaluation_dataset.groupby('user_idx').size()
    rich_users = eval_user_counts[eval_user_counts >= 5].index.values  
    rng = np.random.RandomState(42)
    sample_users = rng.choice(rich_users, min(5, len(rich_users)), replace=False)
    
    for user_idx in sample_users:
        user_id = idx_to_user[int(user_idx)]
        user_train = train_data[train_data['user_idx'] == user_idx]
        user_eval = evaluation_dataset[evaluation_dataset['user_idx'] == user_idx]
        
        all_preds = best_model.predict_all_items(int(user_idx))
        train_items = train_user_items.get(int(user_idx), set())
        for item in train_items:
            all_preds[item] = -np.inf
        top_k = np.argsort(all_preds)[-TOP_K:][::-1]

        print(f"\n  User {user_id} ({len(user_train)} train / {len(user_eval)} validation ratings)")
        top_trained = user_train.nlargest(5, 'Rating')
        print(f"    Liked (training):")
        for _, r in top_trained.iterrows():
            mid = idx_to_movie[int(r['movie_idx'])]
            title = movie_id_to_title.get(mid, f"Movie {mid}")
            print(f"      {r['Rating']:.0f}/5  {title}")

        print(f"    Top-{TOP_K} Recommendations ({overall_winner}):")
        eval_items = dict(zip(user_eval['movie_idx'].values, user_eval['Rating'].values))
        successes, failures = 0, 0
        for rank, item_idx in enumerate(top_k, 1):
            mid = idx_to_movie[int(item_idx)]
            title = movie_id_to_title.get(mid, f"Movie {mid}")
            pred_r = all_preds[item_idx]
            if int(item_idx) in eval_items:
                actual = eval_items[int(item_idx)]
                if actual >= RELEVANCE_THRESHOLD:
                    successes += 1
                    tag = "[HIT]"
                else:
                    failures += 1
                    tag = "[miss]"
                print(f"      {rank:>2}. [pred:{pred_r:.2f}] {title}  -> actual:{actual:.0f}/5 {tag}")
            else:
                print(f"      {rank:>2}. [pred:{pred_r:.2f}] {title}")

        if successes + failures > 0:
            print(f"    Validation-verified accuracy: {successes}/{successes+failures} "
                  f"({100*successes/(successes+failures):.0f}%)")

    # 6. Generate Performance Figure Charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    if tmf_model.train_loss:
        ax.plot(range(1, len(tmf_model.train_loss)+1), tmf_model.train_loss, 'b-o', markersize=3, label='Temporal-MF', linewidth=2)
    if ncf_model.train_loss:
        ax.plot(range(1, len(ncf_model.train_loss)+1), ncf_model.train_loss, 'r-s', markersize=3, label='Neural-CF', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training RMSE', fontsize=12)
    ax.set_title('Training Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    pred_metrics = ['RMSE', 'MAE']
    x = np.arange(len(pred_metrics))
    width = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in pred_metrics]
        ax.bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(pred_metrics, fontsize=12)
    ax.set_title('Rating Prediction (lower = better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    rank_metrics = ['MAP@10', 'Precision@10', 'NDCG@10', 'HitRate@10']
    x = np.arange(len(rank_metrics))
    width = 0.22
    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in rank_metrics]
        ax.bar(x + i*width, vals, width, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(rank_metrics, fontsize=10, rotation=15)
    ax.set_title('Ranking Quality (higher = better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as model_comparison.png")

    print("\n" + "=" * 70)
    print(f"  FINAL RESULT: Best model is {overall_winner}")
    print(f"  Won {best_scores[overall_winner]}/{len(metrics_list)} metrics")
    print("=" * 70)

    print("\nFull Results (copy-paste friendly):")
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'MAP@10':<10} {'P@10':<10} {'R@10':<10} {'NDCG@10':<10} {'HR@10':<10} {'Coverage':<10}")
    print("-" * 95)
    for name in model_names:
        r = results[name]
        print(f"{name:<15} {r['RMSE']:<10.4f} {r['MAE']:<10.4f} {r['MAP@10']:<10.6f} {r['Precision@10']:<10.6f} {r['Recall@10']:<10.6f} {r['NDCG@10']:<10.6f} {r['HitRate@10']:<10.6f} {r['Coverage']:<10.6f}")

    # 7. Generate Vector Similarity Lookups
    get_item_similarities(overall_winner, models, data)

    # 8. Run Deployed Cold Start Module
    run_cold_start_module(models, data, n_movies)

if __name__ == '__main__':
    main()