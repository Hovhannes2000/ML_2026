"""Weighted ALS с точным решением нормальных уравнений."""

import numpy as np
from scipy.linalg import solve
from .base import Recommender, interaction_matrix


class IALS(Recommender):
    def __init__(self, factors=24, regularization=1.0, alpha=20.0, iterations=10, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def targets(self, frame, reference_time):
        return (frame.rating.to_numpy() >= 4).astype(np.float64)

    def _step(self, observations, fixed):
        base = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        result = np.zeros((observations.shape[0], self.factors))
        for row in range(observations.shape[0]):
            start, end = observations.indptr[row:row + 2]
            ids = observations.indices[start:end]
            if not len(ids):
                continue
            local = fixed[ids]
            # Explicit zero targets remain stored: observed dislikes still have c=1+alpha.
            target = observations.data[start:end]
            lhs = base + self.alpha * (local.T @ local)
            rhs = (1.0 + self.alpha) * (local.T @ target)
            result[row] = solve(lhs, rhs, assume_a="pos", check_finite=False)
        return result

    def fit(self, frame, shape, reference_time=None, **kwargs):
        self.shape = shape
        matrix = interaction_matrix(frame, shape, self.targets(frame, reference_time))
        transpose = matrix.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.05, (shape[1], self.factors))
        for _ in range(self.iterations):
            self.user_factors = self._step(matrix, self.item_factors)
            self.item_factors = self._step(transpose, self.user_factors)
        return self

    def score(self, user_ids, query_time=None):
        return self.user_factors[user_ids] @ self.item_factors.T
