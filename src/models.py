# src/models.py
"""
Model Architecture Blueprints Module
------------------------------------
Houses classes containing structural weights definitions and algebraic/deep learning 
optimization routines for collaborative filtering matrix operations.
"""

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SVDRecommender:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.name = "SVD"
        
    def fit(self, train_data, n_users, n_movies, global_mean):
        print(f"\n  Training SVD (k={self.n_factors})...")
        t0 = time.time()
        self.global_mean = global_mean
        self.n_users = n_users
        self.n_movies = n_movies

        # Step 1: Compute biases
        self.user_bias = np.zeros(n_users, dtype=np.float32)
        self.item_bias = np.zeros(n_movies, dtype=np.float32)
        for uid, mean_r in train_data.groupby('user_idx')['Rating'].mean().items():
            self.user_bias[uid] = mean_r - global_mean
        for mid, mean_r in train_data.groupby('movie_idx')['Rating'].mean().items():
            self.item_bias[mid] = mean_r - global_mean

        # Step 2: Build centered sparse matrix
        rows = train_data['user_idx'].values
        cols = train_data['movie_idx'].values
        vals = train_data['Rating'].values.astype(np.float32)
        centered = vals - global_mean - self.user_bias[rows] - self.item_bias[cols]
        R = sparse.csr_matrix((centered, (rows, cols)), shape=(n_users, n_movies))

        # Step 3: Truncated SVD
        k = min(self.n_factors, min(n_users, n_movies) - 2)
        self.U, self.sigma, self.Vt = svds(R.astype(np.float64), k=k)
        idx = np.argsort(-self.sigma)
        self.U = self.U[:, idx].astype(np.float32)
        self.sigma = self.sigma[idx].astype(np.float32)
        self.Vt = self.Vt[idx, :].astype(np.float32)

        # Precompute U*sigma for faster predictions
        self.U_sigma = self.U * self.sigma[np.newaxis, :]
        print(f"    Done in {time.time()-t0:.1f}s | Top singular values: {self.sigma[:5]}")
        self.train_loss = []

    def predict_batch(self, user_indices, movie_indices, **kwargs):
        preds = (self.global_mean + self.user_bias[user_indices]
                 + self.item_bias[movie_indices]
                 + np.sum(self.U_sigma[user_indices] * self.Vt[:, movie_indices].T, axis=1))
        return np.clip(preds, 1.0, 5.0)

    def predict_all_items(self, user_idx, **kwargs):
        preds = (self.global_mean + self.user_bias[user_idx]
                 + self.item_bias + self.U_sigma[user_idx] @ self.Vt)
        return np.clip(preds, 1.0, 5.0)


class TemporalMFRecommender:
    def __init__(self, n_factors=50, lr=0.005, reg=0.02, n_epochs=20,
                 n_time_bins=30, batch_size=8192, random_state=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_time_bins = n_time_bins
        self.batch_size = batch_size
        self.random_state = random_state
        self.name = "Temporal-MF"

    def fit(self, train_data, n_users, n_movies, global_mean):
        print(f"\n  Training Temporal MF (k={self.n_factors}, epochs={self.n_epochs}, "
              f"batch={self.batch_size})...")
        t0 = time.time()
        self.global_mean = global_mean
        self.n_users = n_users
        self.n_movies = n_movies

        rng = np.random.RandomState(self.random_state)
        scale = 0.1 / np.sqrt(self.n_factors)

        self.P = rng.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, scale, (n_movies, self.n_factors)).astype(np.float32)
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_movies, dtype=np.float32)
        self.b_ut = np.zeros((n_users, self.n_time_bins), dtype=np.float32)

        users = train_data['user_idx'].values.astype(np.int32)
        items = train_data['movie_idx'].values.astype(np.int32)
        ratings = train_data['Rating'].values.astype(np.float32)
        time_bins = train_data['time_bin'].values.astype(np.int32)
        n_ratings = len(ratings)

        self.train_loss = []

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_ratings)
            epoch_loss = 0.0
            lr = self.lr * (0.9 ** epoch)

            for start in range(0, n_ratings, self.batch_size):
                end = min(start + self.batch_size, n_ratings)
                batch_idx = perm[start:end]

                u = users[batch_idx]
                i = items[batch_idx]
                r = ratings[batch_idx]
                t = time_bins[batch_idx]

                pred = (self.global_mean + self.b_u[u] + self.b_i[i]
                        + self.b_ut[u, t]
                        + np.sum(self.P[u] * self.Q[i], axis=1))

                error = r - pred
                epoch_loss += np.sum(error ** 2)

                err_2d = error.reshape(-1, 1)

                P_update = lr * (err_2d * self.Q[i] - self.reg * self.P[u])
                Q_update = lr * (err_2d * self.P[u] - self.reg * self.Q[i])
                bu_update = lr * (error - self.reg * self.b_u[u])
                bi_update = lr * (error - self.reg * self.b_i[i])
                but_update = lr * (error - self.reg * self.b_ut[u, t])

                np.add.at(self.P, u, P_update)
                np.add.at(self.Q, i, Q_update)
                np.add.at(self.b_u, u, bu_update)
                np.add.at(self.b_i, i, bi_update)
                np.add.at(self.b_ut, (u, t), but_update)

            epoch_rmse = np.sqrt(epoch_loss / n_ratings)
            self.train_loss.append(epoch_rmse)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:>3}/{self.n_epochs}  "
                      f"RMSE: {epoch_rmse:.4f}  lr: {lr:.5f}  "
                      f"Time: {time.time()-t0:.0f}s")

        print(f"    Done in {time.time()-t0:.1f}s")

    def predict_batch(self, user_indices, movie_indices, time_bins=None, **kwargs):
        preds = self.global_mean + self.b_u[user_indices] + self.b_i[movie_indices]
        if time_bins is not None:
            preds += self.b_ut[user_indices, time_bins]
        else:
            preds += self.b_ut[user_indices, self.n_time_bins - 1]
        preds += np.sum(self.P[user_indices] * self.Q[movie_indices], axis=1)
        return np.clip(preds, 1.0, 5.0)

    def predict_all_items(self, user_idx, time_bin=None, **kwargs):
        if time_bin is None:
            time_bin = self.n_time_bins - 1
        preds = (self.global_mean + self.b_u[user_idx] + self.b_i
                 + self.b_ut[user_idx, time_bin] + self.P[user_idx] @ self.Q.T)
        return np.clip(preds, 1.0, 5.0)


class NeuralCFNet(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)

        layers = []
        input_dim = 2 * embed_dim + 1
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)

    def forward(self, user_ids, item_ids, time_norm):
        u_emb = self.user_embed(user_ids)          
        i_emb = self.item_embed(item_ids)          
        t = time_norm.unsqueeze(1)                 
        x = torch.cat([u_emb, i_emb, t], dim=1)   
        return self.mlp(x).squeeze(1)              


class NeuralCFRecommender:
    def __init__(self, n_users, n_movies, embed_dim=32, hidden_dims=None,
                 lr=0.001, n_epochs=15, batch_size=8192, device='cpu'):
        self.n_users = n_users
        self.n_movies = n_movies
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.name = "Neural-CF"
        self.model = None

    def fit(self, train_data, n_users, n_movies, global_mean):
        print(f"\n  Training Neural CF on {self.device} (embed={self.embed_dim}, "
              f"hidden={self.hidden_dims}, epochs={self.n_epochs})...")
        t0 = time.time()
        self.global_mean = global_mean

        self.model = NeuralCFNet(
            n_users, n_movies, self.embed_dim, self.hidden_dims
        ).to(self.device)

        users_t = torch.tensor(train_data['user_idx'].values, dtype=torch.long)
        items_t = torch.tensor(train_data['movie_idx'].values, dtype=torch.long)
        ratings_t = torch.tensor(train_data['Rating'].values, dtype=torch.float32)
        times_t = torch.tensor(train_data['time_norm'].values, dtype=torch.float32)

        dataset = TensorDataset(users_t, items_t, ratings_t, times_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=True, num_workers=0, pin_memory=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        criterion = nn.MSELoss()

        self.train_loss = []

        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0.0
            n_samples = 0

            for u_batch, i_batch, r_batch, t_batch in dataloader:
                u_batch = u_batch.to(self.device)
                i_batch = i_batch.to(self.device)
                r_batch = r_batch.to(device=self.device)
                t_batch = t_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(u_batch, i_batch, t_batch)
                loss = criterion(pred, r_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(r_batch)
                n_samples += len(r_batch)

            scheduler.step()
            epoch_rmse = np.sqrt(epoch_loss / n_samples)
            self.train_loss.append(epoch_rmse)

            if (epoch + 1) % 3 == 0 or epoch == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"    Epoch {epoch+1:>3}/{self.n_epochs}  "
                      f"RMSE: {epoch_rmse:.4f}  lr: {lr_now:.6f}  "
                      f"Time: {time.time()-t0:.0f}s")

        print(f"    Done in {time.time()-t0:.1f}s")
        self.model.eval()

    def predict_batch(self, user_indices, movie_indices, time_norms=None, **kwargs):
        if time_norms is None:
            time_norms = np.ones(len(user_indices), dtype=np.float32)
        with torch.no_grad():
            u = torch.tensor(user_indices, dtype=torch.long, device=self.device)
            i = torch.tensor(movie_indices, dtype=torch.long, device=self.device)
            t = torch.tensor(time_norms, dtype=torch.float32, device=self.device)
            pred = self.model(u, i, t).cpu().numpy()
        return np.clip(pred, 1.0, 5.0)

    def predict_all_items(self, user_idx, time_norm=1.0, **kwargs):
        n = self.n_movies
        with torch.no_grad():
            u = torch.full((n,), user_idx, dtype=torch.long, device=self.device)
            i = torch.arange(n, dtype=torch.long, device=self.device)
            t = torch.full((n,), time_norm, dtype=torch.float32, device=self.device)
            pred = self.model(u, i, t).cpu().numpy()
        return np.clip(pred, 1.0, 5.0)