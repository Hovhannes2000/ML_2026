"""FM второго порядка: аналитические градиенты и собственный AdaGrad на NumPy."""

import numpy as np
import pandas as pd
from scipy.special import expit
from .base import Recommender


class FactorizationMachine(Recommender):
    def __init__(self, factors=16, learning_rate=0.05, regularization=1e-4,
                 epochs=12, batch_size=2048, metadata=True, seed=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.metadata = metadata
        self.seed = seed

    def _history(self, frame):
        self.user_times = [np.empty(0, dtype=np.int64) for _ in range(self.shape[0])]
        self.item_times = [np.empty(0, dtype=np.int64) for _ in range(self.shape[1])]
        for idx, group in frame.groupby("user_idx", sort=False):
            self.user_times[idx] = np.sort(group.timestamp.to_numpy())
        for idx, group in frame.groupby("item_idx", sort=False):
            self.item_times[idx] = np.sort(group.timestamp.to_numpy())
        self.user_count = np.array([len(x) for x in self.user_times])
        self.item_count = np.array([len(x) for x in self.item_times])
        self.item_first = np.array([x[0] if len(x) else np.iinfo(np.int64).max for x in self.item_times])

    @staticmethod
    def _prefix_count(ids, times, histories):
        result = np.zeros(len(ids), dtype=np.float64)
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_ids)) + 1, len(ids)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            positions = order[left:right]
            result[positions] = np.searchsorted(histories[ids[positions[0]]], times[positions], side="left")
        return result

    def _features(self, users, items, times):
        n = len(users)
        if not self.metadata:
            return np.column_stack([users, self.shape[0] + items]), np.ones((n, 2))
        base = self.shape[0] + self.shape[1]
        dates = pd.to_datetime(times, unit="s", utc=True)
        user_count = self._prefix_count(users, times, self.user_times)
        item_count = self._prefix_count(items, times, self.item_times)
        age = np.maximum(0, (times - self.item_first[items]) / 86400.0)
        indices = np.column_stack([users, self.shape[0] + items,
                                   base + dates.month.to_numpy() - 1,
                                   base + 12 + dates.dayofweek.to_numpy(),
                                   np.full(n, base + 19), np.full(n, base + 20), np.full(n, base + 21)])
        values = np.column_stack([np.ones((n, 4)),
                                  np.log1p(user_count) / np.log(101),
                                  np.log1p(item_count) / np.log(101),
                                  np.log1p(age) / np.log(3651)])
        return indices.astype(np.int64), values

    def _negative_samples(self, frame, rng):
        positive = frame[frame.rating >= 4]
        users = positive.user_idx.to_numpy()
        times = positive.timestamp.to_numpy()
        item_order = np.argsort(self.item_first, kind="stable")
        # Товар должен уже встречаться до момента обучающего примера.
        available = np.searchsorted(self.item_first[item_order], times, side="left")
        known_keys = np.sort(frame.user_idx.to_numpy() * self.shape[1] + frame.item_idx.to_numpy())
        items = np.zeros(len(users), dtype=np.int64)
        pending = available > 0
        accepted = np.zeros(len(users), dtype=bool)
        for _ in range(30):
            ids = np.flatnonzero(pending)
            if not len(ids):
                break
            items[ids] = item_order[(rng.random(len(ids)) * available[ids]).astype(int)]
            keys = users[ids] * self.shape[1] + items[ids]
            pos = np.searchsorted(known_keys, keys)
            seen = (pos < len(known_keys)) & (known_keys[np.minimum(pos, len(known_keys) - 1)] == keys)
            good = ids[~seen]
            accepted[good] = True
            pending[good] = False
        return users[accepted], items[accepted], times[accepted]

    def _predict_features(self, indices, values):
        latent = self.V[indices] * values[..., None]
        summed = latent.sum(axis=1)
        score = self.bias + (self.w[indices] * values).sum(axis=1)
        score += 0.5 * (summed ** 2 - (latent ** 2).sum(axis=1)).sum(axis=1)
        return score, summed

    def _gradients(self, indices, values, labels):
        scores, summed = self._predict_features(indices, values)
        error = (expit(scores) - labels) / len(labels)
        unique, inverse = np.unique(indices, return_inverse=True)
        inverse = inverse.ravel()
        gw = np.zeros(len(unique))
        gv = np.zeros((len(unique), self.factors))
        dw = error[:, None] * values
        dv = (error[:, None, None] * values[..., None]
              * (summed[:, None, :] - self.V[indices] * values[..., None]))
        np.add.at(gw, inverse, dw.ravel())
        np.add.at(gv, inverse, dv.reshape(-1, self.factors))
        frequency = np.bincount(inverse, minlength=len(unique)) / len(labels)
        gw += self.regularization * frequency * self.w[unique]
        gv += self.regularization * frequency[:, None] * self.V[unique]
        return unique, gw, gv, float(error.sum())

    def fit(self, frame, shape, reference_time=None, **kwargs):
        self.shape = shape
        self.reference_time = reference_time
        self._history(frame)
        rng = np.random.default_rng(self.seed)
        negative_users, negative_items, negative_times = self._negative_samples(frame, rng)
        users = np.r_[frame.user_idx.to_numpy(), negative_users]
        items = np.r_[frame.item_idx.to_numpy(), negative_items]
        times = np.r_[frame.timestamp.to_numpy(), negative_times]
        labels = np.r_[(frame.rating.to_numpy() >= 4).astype(float), np.zeros(len(negative_users))]
        indices, values = self._features(users, items, times)
        feature_count = shape[0] + shape[1] + (22 if self.metadata else 0)
        self.w = np.zeros(feature_count)
        self.V = rng.normal(0, 0.02, (feature_count, self.factors))
        self.bias = 0.0
        aw = np.full_like(self.w, 1e-8)
        av = np.full_like(self.V, 1e-8)
        ab = 1e-8
        self.loss_history = []
        self.negative_count = len(negative_users)
        for _ in range(self.epochs):
            order = rng.permutation(len(labels))
            for start in range(0, len(order), self.batch_size):
                ids = order[start:start + self.batch_size]
                unique, gw, gv, gb = self._gradients(indices[ids], values[ids], labels[ids])
                aw[unique] += gw ** 2
                av[unique] += gv ** 2
                ab += gb ** 2
                self.w[unique] -= self.learning_rate * gw / np.sqrt(aw[unique])
                self.V[unique] -= self.learning_rate * gv / np.sqrt(av[unique])
                self.bias -= self.learning_rate * gb / np.sqrt(ab)
            loss = 0.0
            for start in range(0, len(labels), 8192):
                sl = slice(start, start + 8192)
                scores, _ = self._predict_features(indices[sl], values[sl])
                loss += np.sum(np.logaddexp(0, scores) - labels[sl] * scores)
            self.loss_history.append(loss / len(labels))
        return self

    def _group_score(self, indices, values):
        latent = self.V[indices] * values[..., None]
        summed = latent.sum(axis=1)
        bias = (self.w[indices] * values).sum(axis=1)
        bias += 0.5 * (summed ** 2 - (latent ** 2).sum(axis=1)).sum(axis=1)
        return bias, summed

    def score(self, user_ids, query_time=None):
        if query_time is None:
            query_time = self.reference_time
        if query_time < self.reference_time:
            raise ValueError("Для serving нельзя использовать историю из будущего")
        items = np.arange(self.shape[1])
        if self.metadata:
            base = sum(self.shape)
            date = pd.to_datetime(query_time, unit="s", utc=True)
            user_indices = np.column_stack([user_ids, np.full(len(user_ids), base + date.month - 1),
                                             np.full(len(user_ids), base + 12 + date.dayofweek),
                                             np.full(len(user_ids), base + 19)])
            user_values = np.column_stack([np.ones((len(user_ids), 3)),
                                            np.log1p(self.user_count[user_ids]) / np.log(101)])
            item_indices = np.column_stack([self.shape[0] + items, np.full(len(items), base + 20),
                                             np.full(len(items), base + 21)])
            age = np.maximum(0, (query_time - self.item_first) / 86400.0)
            item_values = np.column_stack([np.ones(len(items)), np.log1p(self.item_count) / np.log(101),
                                            np.log1p(age) / np.log(3651)])
        else:
            user_indices, user_values = user_ids[:, None], np.ones((len(user_ids), 1))
            item_indices, item_values = (self.shape[0] + items)[:, None], np.ones((len(items), 1))
        ub, uv = self._group_score(user_indices, user_values)
        ib, iv = self._group_score(item_indices, item_values)
        return self.bias + ub[:, None] + ib[None, :] + uv @ iv.T
