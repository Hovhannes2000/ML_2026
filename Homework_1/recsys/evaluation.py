"""Полный каталог кандидатов; одинаковые пользователи и метрики для всех моделей."""

import numpy as np
import pandas as pd
from models.base import interaction_matrix


def make_protocol(history, future, shape, query_time):
    seen = interaction_matrix(history, shape)
    relevant = future[future.rating >= 4]
    truth = {int(user): np.sort(group.item_idx.unique()) for user, group in relevant.groupby("user_idx")}
    truth = {u: items[~np.isin(items, seen.indices[seen.indptr[u]:seen.indptr[u + 1]])]
             for u, items in truth.items()}
    truth = {u: items for u, items in truth.items() if len(items)}
    users = np.array(sorted(truth), dtype=np.int64)
    if not len(users):
        raise ValueError("Нет пользователей с релевантными объектами")
    return {"seen": seen, "truth": truth, "users": users, "query_time": query_time, "shape": shape}


def ranking_metrics(recommendations, users, truth, n_items):
    k = recommendations.shape[1]
    discounts = 1 / np.log2(np.arange(2, k + 2))
    rows = []
    for user, predicted in zip(users, recommendations):
        relevant = truth[int(user)]
        hits = np.isin(predicted, relevant)
        ideal = discounts[:min(k, len(relevant))].sum()
        rows.append((int(user), float((hits * discounts).sum() / ideal), float(hits.sum() / len(relevant))))
    per_user = pd.DataFrame(rows, columns=["user_idx", "NDCG@10", "Recall@10"])
    metrics = {name: float(per_user[name].mean()) for name in ["NDCG@10", "Recall@10"]}
    metrics["Coverage@10"] = float(len(np.unique(recommendations)) / n_items)
    return metrics, per_user


def evaluate(model, protocol):
    recommendations = model.recommend(protocol["users"], protocol["seen"], k=10,
                                      query_time=protocol["query_time"])
    metrics, per_user = ranking_metrics(recommendations, protocol["users"], protocol["truth"], protocol["shape"][1])
    # Проверяем именно рекомендации, а не только предсказания.
    assert all(len(np.unique(row)) == 10 for row in recommendations)
    for user, row in zip(protocol["users"], recommendations):
        assert protocol["seen"][int(user), row].nnz == 0
    return metrics, per_user, recommendations


def paired_bootstrap(left, right, metric="NDCG@10", seed=42, repeats=2000):
    a = left.set_index("user_idx")[metric].sort_index()
    b = right.set_index("user_idx")[metric].sort_index()
    assert a.index.equals(b.index)
    difference = a.to_numpy() - b.to_numpy()
    rng = np.random.default_rng(seed)
    samples = np.array([rng.choice(difference, len(difference), replace=True).mean() for _ in range(repeats)])
    return {"difference": float(difference.mean()), "ci_low": float(np.quantile(samples, .025)),
            "ci_high": float(np.quantile(samples, .975)), "users": len(difference)}
