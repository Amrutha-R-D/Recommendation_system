# src/recommender.py
"""
Recommendation Generation Module
--------------------------------
Generates vector similarities across latent spaces and outputs fallback configurations
for cold-start execution paths.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_item_similarities(overall_winner, models, data):
    print("\n" + "=" * 70)
    print("  IDENTIFYING SIMILARITIES (Item-Item Collaborative Filtering)")
    print("=" * 70)
    
    best_model_name = overall_winner
    best_model = models[best_model_name]

    if best_model_name == 'Neural-CF':
        item_factors = best_model.model.item_embed.weight.detach().cpu().numpy()
    elif best_model_name == 'Temporal-MF':
        item_factors = best_model.Q
    else: 
        item_factors = best_model.Vt.T

    movie_to_idx = data['movie_to_idx']
    idx_to_movie = data['idx_to_movie']
    df_movies = data['df_movies']

    sample_movie_ids = list(movie_to_idx.keys())[:3]
    for target_movie_id in sample_movie_ids:
        target_idx = movie_to_idx[target_movie_id]
        target_vector = item_factors[target_idx].reshape(1, -1)
        
        similarities = cosine_similarity(target_vector, item_factors).flatten()
        similar_indices = np.argsort(similarities)
        similar_indices = [idx for idx in similar_indices if idx != target_idx][-5:][::-1]

        target_title_series = df_movies[df_movies['Movie_ID'] == target_movie_id]['Title']
        target_title = target_title_series.values[0] if len(target_title_series) > 0 else f"Movie ID {target_movie_id}"

        print(f"\nBecause you watched: '{target_title}'")
        print("  Our system recommends these similar titles:")

        for rank, idx in enumerate(similar_indices, 1):
            sim_score = similarities[idx]
            mid = idx_to_movie[idx]

            title_series = df_movies[df_movies['Movie_ID'] == mid]['Title']
            title = title_series.values[0] if len(title_series) > 0 else f"Movie ID {mid}"
            print(f"    {rank}. {title:<45s} (Match: {sim_score*100:5.1f}%)")


def run_cold_start_module(models, data, n_movies):
    print("\n" + "=" * 70)
    print("  HANDLING COLD START (New Users with 0 History)")
    print("=" * 70)
    
    if 'Temporal-MF' in models:
        item_biases = models['Temporal-MF'].b_i
    elif 'SVD' in models:
        item_biases = models['SVD'].item_bias
    else:
        item_biases = np.zeros(n_movies)

    movie_counts = data['train_data']['movie_idx'].value_counts()
    popular_movie_indices = movie_counts[movie_counts > 5000].index.values
    movie_means = data['train_data'].groupby('movie_idx')['Rating'].mean()

    popular_biases = {idx: item_biases[idx] for idx in popular_movie_indices}
    top_cold_start_indices = sorted(popular_biases, key=popular_biases.get, reverse=True)[:5]
    
    print("Simulating a brand new user (User ID: 999999999) with ZERO historical ratings.")
    print("Falling back to Global Popularity / Demographic recommendations:\n")
    
    for rank, idx in enumerate(top_cold_start_indices, 1):
        mid = data['idx_to_movie'][idx]
        title = data['df_movies'][data['df_movies']['Movie_ID'] == mid]['Title'].values[0]
        total_ratings = movie_counts[idx]
        avg_rating = movie_means[idx]

        print(f"  {rank}. {title}")
        print(f"     [Justification: Universally Acclaimed — Rated {total_ratings:,} times with an Average Score of {avg_rating:.2f}/5.0]")