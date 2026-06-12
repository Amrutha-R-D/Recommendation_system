# src/evaluation.py
"""
Evaluation Scripts 
Calculates statistical accuracy metrics and rank-sorted ranking evaluation metrics
against sparse validation test sets.
"""

import time
import numpy as np
from collections import defaultdict

def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def compute_ranking_metrics(model, test_data, train_user_items, n_movies,
                            k=10, threshold=3.5, sample_n=3000, RANDOM_STATE=42,
                            model_name="Model"):
    print(f"    Computing ranking metrics for {model_name} ({sample_n} users)...", end=" ")
    t0 = time.time()

    test_user_ratings = defaultdict(dict)
    for u, m, r in zip(test_data['user_idx'].values,
                       test_data['movie_idx'].values,
                       test_data['Rating'].values):
        test_user_ratings[int(u)][int(m)] = float(r)

    eligible_users = [u for u in test_user_ratings if len(test_user_ratings[u]) >= 1]
    rng = np.random.RandomState(RANDOM_STATE)
    if len(eligible_users) > sample_n:
        eligible_users = rng.choice(eligible_users, sample_n, replace=False).tolist()

    aps, precisions, recalls, ndcgs, hits = [], [], [], [], []
    all_recommended = set()

    for user_idx in eligible_users:
        all_preds = model.predict_all_items(user_idx)

        train_items = train_user_items.get(user_idx, set())
        mask = np.ones(n_movies, dtype=bool)
        for item in train_items:
            mask[item] = False
        candidate_preds = np.full(n_movies, -np.inf)
        candidate_preds[mask] = all_preds[mask]

        if np.all(candidate_preds == -np.inf):
            continue

        top_k_items = np.argpartition(candidate_preds, -k)[-k:]
        top_k_items = top_k_items[np.argsort(candidate_preds[top_k_items])[::-1]]
        all_recommended.update(top_k_items.tolist())

        actual = test_user_ratings[user_idx]
        relevances = [1 if (int(item) in actual and actual[int(item)] >= threshold) else 0
                      for item in top_k_items]
        total_relevant = sum(1 for r in actual.values() if r >= threshold)

        if total_relevant > 0:
            cum_rel, prec_sum = 0, 0.0
            for rank, rel in enumerate(relevances, 1):
                if rel == 1:
                    cum_rel += 1
                    prec_sum += cum_rel / rank
            aps.append(prec_sum / total_relevant)
        else:
            aps.append(0.0)

        n_rel_in_k = sum(relevances)
        precisions.append(n_rel_in_k / k)
        recalls.append(n_rel_in_k / total_relevant if total_relevant > 0 else 0.0)

        dcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(relevances, 1))
        idcg = sum(r / np.log2(i + 1) for i, r in enumerate(sorted(relevances, reverse=True), 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        hits.append(1 if n_rel_in_k > 0 else 0)

    print(f"done in {time.time()-t0:.1f}s")

    return {
        'MAP@10': np.mean(aps),
        'Precision@10': np.mean(precisions),
        'Recall@10': np.mean(recalls),
        'NDCG@10': np.mean(ndcgs),
        'HitRate@10': np.mean(hits),
        'Coverage': len(all_recommended) / n_movies,
    }

def evaluate_model(model, eval_dataset, train_user_items, n_movies, name, TOP_K, RELEVANCE_THRESHOLD, MAP_SAMPLE_USERS, RANDOM_STATE):
    print(f"\n  Evaluating {name}...")
    rating_col = None
    for col in ['Rating', 'rating']:
        if col in eval_dataset.columns:
            rating_col = col
            break

    if rating_col is None:
        raise KeyError(f"Could not find a rating column in evaluation data. Available columns: {list(eval_dataset.columns)}")

    users_eval = eval_dataset['user_idx'].values
    items_eval = eval_dataset['movie_idx'].values
    ratings_eval = eval_dataset[rating_col].values.astype(np.float32)

    if name == 'Temporal-MF' and 'time_bin' in eval_dataset.columns:
        preds = model.predict_batch(users_eval, items_eval, time_bins=eval_dataset['time_bin'].values)
    elif name == 'Neural-CF' and 'time_norm' in eval_dataset.columns:
        preds = model.predict_batch(users_eval, items_eval, time_norms=eval_dataset['time_norm'].values.astype(np.float32))
    else:
        preds = model.predict_batch(users_eval, items_eval)

    test_rmse = rmse_score(ratings_eval, preds)
    test_mae = mae_score(ratings_eval, preds)
    print(f"    Validation RMSE: {test_rmse:.4f}  |  Validation MAE: {test_mae:.4f}")

    ranking = compute_ranking_metrics(
        model, eval_dataset, train_user_items, n_movies,
        k=TOP_K, threshold=RELEVANCE_THRESHOLD,
        sample_n=MAP_SAMPLE_USERS, RANDOM_STATE=RANDOM_STATE, model_name=name
    )

    results = {'RMSE': test_rmse, 'MAE': test_mae, **ranking}
    return results