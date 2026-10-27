"""EASE: closed-form ridge regression с нулевой диагональю."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender, interaction_matrix


class EASE(Recommender):
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, frame, shape, **kwargs):
        self.shape = shape
        self.history = interaction_matrix(frame[frame.rating >= 4], shape)
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::shape[1] + 1] += self.regularization
        inverse = cho_solve(cho_factor(gram, check_finite=False), np.eye(shape[1]), check_finite=False)
        self.weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        return self

    def score(self, user_ids, query_time=None):
        return self.history[user_ids] @ self.weights
