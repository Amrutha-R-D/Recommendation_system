# src/data_pipeline.py
"""
Data Processing Pipeline Module
-------------------------------
Handles raw dataset Parquet ingestion, feature engineering on release dates,
interaction density mapping constraints, and chronological matrix tokenizations.
"""

import os
import time
import numpy as np
import pandas as pd

def load_and_prepare_data(DATA_DIR, SAMPLE_USERS, MIN_USER_RATINGS, TMF_TIME_BINS):
    print("=" * 70)
    print("  DATA LOADING & PREPARATION")
    print("=" * 70)
    
    # 1. Load Data
    print("\n[1] Loading parquet files...")
    t0 = time.time()
    df_train = pd.read_parquet(os.path.join(DATA_DIR, 'df_train.parquet'))
    df_val = pd.read_parquet(os.path.join(DATA_DIR, 'df_val.parquet'))
    df_test_unlabeled = pd.read_parquet(os.path.join(DATA_DIR, 'df_test_unlabeled.parquet'))
    df_movies = pd.read_parquet(os.path.join(DATA_DIR, 'df_movies.parquet'))
    print(f"    Loaded in {time.time()-t0:.1f}s")

    print("\n[2] Merging Movie Metadata (Year) into training/val/test...")
    median_year = df_movies['Year'].median()
    df_movies['Year'] = df_movies['Year'].fillna(median_year)
    min_year, max_year = df_movies['Year'].min(), df_movies['Year'].max()
    df_movies['Year_Norm'] = ((df_movies['Year'] - min_year) / (max_year - min_year)).astype(np.float32)

    movie_year_map = df_movies.set_index('Movie_ID')['Year_Norm'].to_dict()

    for df in [df_train, df_val, df_test_unlabeled]:
        df['movie_year_norm'] = df['Movie_ID'].map(movie_year_map).fillna(0.5).astype(np.float32)
        
    user_counts = df_train['User_ID'].value_counts()
    active_users = user_counts[user_counts >= MIN_USER_RATINGS].index.values
    if SAMPLE_USERS is not None:
        rng = np.random.RandomState(42)
        active_users = rng.choice(active_users, min(SAMPLE_USERS, len(active_users)), replace=False)

    active_set = set(active_users.tolist())
    df_train = df_train[df_train['User_ID'].isin(active_set)].copy()
    df_val = df_val[df_val['User_ID'].isin(active_set)].copy()

    print("\n[3] Creating ID mappings...")
    all_users = set(df_train['User_ID']).union(set(df_val['User_ID'])).union(set(df_test_unlabeled['User_ID']))
    all_movies = set(df_train['Movie_ID']).union(set(df_val['Movie_ID'])).union(set(df_test_unlabeled['Movie_ID']))

    user_to_idx = {int(uid): i for i, uid in enumerate(sorted(all_users))}
    movie_to_idx = {int(mid): i for i, mid in enumerate(sorted(all_movies))}
    idx_to_user = {i: int(uid) for uid, i in user_to_idx.items()}
    idx_to_movie = {i: int(mid) for mid, i in movie_to_idx.items()}
    n_users, n_movies = len(user_to_idx), len(movie_to_idx)
    for df in [df_train, df_val, df_test_unlabeled]:
        df['user_idx'] = df['User_ID'].map(user_to_idx).astype(np.int32)
        df['movie_idx'] = df['Movie_ID'].map(movie_to_idx).astype(np.int32)

    print("\n[4] Engineering temporal features...")
    df_train['Date'] = pd.to_datetime(df_train['Date'])
    ref_date = df_train['Date'].min()
    max_date = df_train['Date'].max()
    total_days = max(1, (max_date - ref_date).days)

    for df in [df_train, df_val, df_test_unlabeled]:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['time_norm'] = ((df['Date'] - ref_date).dt.days.clip(0) / total_days).clip(0, 1).astype(np.float32)
            df['time_bin'] = (df['time_norm'] * (TMF_TIME_BINS - 1)).astype(np.int32).clip(0, TMF_TIME_BINS - 1)
            
    global_mean = float(df_train['Rating'].mean())
    train_user_items = df_train.groupby('user_idx')['movie_idx'].apply(set).to_dict()
    movie_idx_to_year = np.zeros(n_movies, dtype=np.float32)
    for idx, mid in idx_to_movie.items():
        movie_idx_to_year[idx] = movie_year_map.get(mid, 0.5)
        
    print(f"\n  DATA SUMMARY")
    print(f"    Users:            {n_users:>10,}")
    print(f"    Movies:           {n_movies:>10,}")
    print(f"    Train ratings:    {len(df_train):>10,}")
    print(f"    Val ratings:      {len(df_val):>10,}")
    print(f"    Test predictions: {len(df_test_unlabeled):>10,}")
    
    return {
        'train_data': df_train, 'val_data': df_val, 'test_data': df_test_unlabeled,
        'df_movies': df_movies, 'user_to_idx': user_to_idx, 'movie_to_idx': movie_to_idx,
        'idx_to_user': idx_to_user, 'idx_to_movie': idx_to_movie,
        'n_users': n_users, 'n_movies': n_movies, 'global_mean': global_mean,
        'train_user_items': train_user_items, 'movie_idx_to_year': movie_idx_to_year
    }