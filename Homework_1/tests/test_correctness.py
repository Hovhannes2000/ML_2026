"""Численные проверки алгоритмов и защиты протокола от утечки."""

import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from models import IALS, EASE, ItemKNN, CustomTargetALS, FactorizationMachine
from models.base import interaction_matrix
from recsys.evaluation import ranking_metrics
from recsys.data import _text


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"user_idx": [0, 0, 1, 1, 2, 2], "item_idx": [0, 1, 1, 2, 0, 3],
                                   "rating": [5, 1, 4, 5, 2, 4], "timestamp": [100, 200, 200, 300, 400, 500]})
        self.shape = (3, 12)

    def test_als_normal_equations_with_observed_zero(self):
        model = IALS(factors=3, regularization=2, alpha=4, iterations=1)
        rng = np.random.default_rng(4)
        fixed = rng.normal(size=(12, 3))
        targets = interaction_matrix(self.frame, self.shape, model.targets(self.frame, 1000))
        result = model._step(targets, fixed)
        for user in range(3):
            observed = self.frame[self.frame.user_idx == user]
            confidence, preference = np.ones(12), np.zeros(12)
            confidence[observed.item_idx] = 5
            preference[observed.item_idx] = (observed.rating >= 4).astype(float)
            expected = np.linalg.solve(fixed.T @ (confidence[:, None] * fixed) + 2 * np.eye(3),
                                       fixed.T @ (confidence * preference))
            np.testing.assert_allclose(result[user], expected, atol=1e-12)

    def test_ease_constrained_gradient(self):
        model = EASE(regularization=2).fit(self.frame, self.shape)
        x = interaction_matrix(self.frame[self.frame.rating >= 4], self.shape).toarray()
        gradient = x.T @ (x @ model.weights - x) + 2 * model.weights
        np.fill_diagonal(gradient, 0)
        np.testing.assert_allclose(gradient, 0, atol=1e-12)
        np.testing.assert_array_equal(np.diag(model.weights), 0)

    def test_knn_similarity_and_exclusion(self):
        model = ItemKNN(neighbors=5, shrinkage=0).fit(self.frame, self.shape)
        x = interaction_matrix(self.frame[self.frame.rating >= 4], self.shape).toarray()
        expected = x.T @ x
        norm = np.sqrt(np.diag(expected))
        expected /= np.maximum(norm[:, None] * norm[None, :], 1)
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.weights.toarray(), expected)
        seen = interaction_matrix(self.frame, self.shape)
        recs = model.recommend(np.arange(3), seen, k=10)
        for user, rec in enumerate(recs):
            self.assertEqual(seen[user, rec].nnz, 0)
            self.assertEqual(len(set(rec)), 10)

    def test_custom_target_changes_preference_not_confidence(self):
        model = CustomTargetALS(half_life_days=1)
        frame = pd.DataFrame({"rating": [1, 3, 5, 5], "timestamp": [86400, 86400, 86400, 0]})
        np.testing.assert_allclose(model.targets(frame, 86400), [-1, 0, 1, .5])
        self.assertEqual(model.alpha, IALS().alpha)

    def test_fm_gradient_finite_differences(self):
        model = FactorizationMachine(factors=2, regularization=.03)
        rng = np.random.default_rng(5)
        model.w = rng.normal(size=4)
        model.V = rng.normal(size=(4, 2))
        model.bias = .2
        indices = np.array([[0, 1, 3], [0, 2, 3]])
        values = np.array([[1, 1, .4], [1, 1, .7]])
        labels = np.array([1., 0.])
        unique, gw, gv, gb = model._gradients(indices, values, labels)

        def objective():
            scores, _ = model._predict_features(indices, values)
            loss = np.mean(np.logaddexp(0, scores) - labels * scores)
            penalty = .5 * model.regularization * np.mean((model.w[indices] ** 2).sum(axis=1)
                                                          + (model.V[indices] ** 2).sum(axis=(1, 2)))
            return loss + penalty

        eps = 1e-6
        for pos, feature in enumerate(unique):
            original = model.w[feature]
            model.w[feature] = original + eps
            plus = objective()
            model.w[feature] = original - eps
            minus = objective()
            model.w[feature] = original
            self.assertAlmostEqual((plus - minus) / (2 * eps), gw[pos], places=7)
            for factor in range(2):
                original = model.V[feature, factor]
                model.V[feature, factor] = original + eps
                plus = objective()
                model.V[feature, factor] = original - eps
                minus = objective()
                model.V[feature, factor] = original
                self.assertAlmostEqual((plus - minus) / (2 * eps), gv[pos, factor], places=7)
        scores, _ = model._predict_features(indices, values)
        self.assertAlmostEqual(gb, np.mean(expit(scores) - labels))

    def test_fm_prefix_counts_are_strict_and_serving_formula_matches(self):
        model = FactorizationMachine(factors=3, epochs=1).fit(self.frame, self.shape, reference_time=600)
        counts = model._prefix_count(np.array([0, 0, 1]), np.array([100, 200, 200]), model.user_times)
        np.testing.assert_array_equal(counts, [0, 1, 0])
        users, items = np.array([0, 1, 2]), np.array([1, 2, 3])
        ids, values = model._features(users, items, np.array([600, 600, 600]))
        direct, _ = model._predict_features(ids, values)
        np.testing.assert_allclose(direct, model.score(users, query_time=600)[np.arange(3), items], atol=1e-12)

    def test_metrics_known_example(self):
        recs = np.array([[0, 2, 1, 3, 4, 5, 6, 7, 8, 9]])
        metrics, _ = ranking_metrics(recs, np.array([4]), {4: np.array([0, 1, 11])}, 12)
        self.assertAlmostEqual(metrics["Recall@10"], 2 / 3)
        self.assertAlmostEqual(metrics["NDCG@10"], 1.5 / (1 + 1 / np.log2(3) + .5))

    def test_metadata_html_cleaning(self):
        self.assertEqual(_text('<span class="a-size-medium"'), '')
        self.assertEqual(_text('<b>Rock &amp; Roll</b>'), 'Rock & Roll')
        self.assertEqual(_text(None), '')


if __name__ == "__main__":
    unittest.main()
