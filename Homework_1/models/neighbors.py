"""Item-based CF: cosine similarity, shrinkage и top-K соседей."""

import numpy as np
from scipy import sparse
from .base import Recommender, interaction_matrix


class ItemKNN(Recommender):
    def __init__(self, neighbors=50, shrinkage=10.0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, frame, shape, **kwargs):
        self.shape = shape
        self.history = interaction_matrix(frame[frame.rating >= 4], shape)
        gram = (self.history.T @ self.history).tocsc()
        degree = np.asarray(self.history.sum(axis=0)).ravel()
        indices, rows, values = [], [], []
        for target in range(shape[1]):
            start, end = gram.indptr[target:target + 2]
            candidates = gram.indices[start:end]
            common = gram.data[start:end]
            valid = candidates != target
            candidates, common = candidates[valid], common[valid]
            similarity = common / np.sqrt(degree[candidates] * degree[target])
            similarity *= common / (common + self.shrinkage)
            keep = np.lexsort((candidates, -similarity))[:self.neighbors]
            rows.extend(candidates[keep])
            indices.extend([target] * len(keep))
            values.extend(similarity[keep])
        self.weights = sparse.csr_matrix((values, (rows, indices)), shape=(shape[1], shape[1]))
        return self

    def score(self, user_ids, query_time=None):
        return (self.history[user_ids] @ self.weights).toarray()
