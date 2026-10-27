"""Общий интерфейс, построение разреженных матриц и точный top-K."""

import numpy as np
from scipy import sparse


def interaction_matrix(frame, shape, values=None):
    if values is None:
        values = np.ones(len(frame), dtype=np.float64)
    result = sparse.csr_matrix(
        (values, (frame.user_idx.to_numpy(), frame.item_idx.to_numpy())),
        shape=shape,
        dtype=np.float64,
    )
    result.sort_indices()
    return result


class Recommender:
    """Все модели используют один алгоритм исключения истории и ранжирования."""

    def recommend(self, user_ids, seen, k=10, query_time=None, batch_size=128):
        user_ids = np.asarray(user_ids, dtype=np.int64)
        if k < 1 or k > self.shape[1]:
            raise ValueError("Некорректное K")
        output = []
        for start in range(0, len(user_ids), batch_size):
            users = user_ids[start:start + batch_size]
            scores = np.asarray(self.score(users, query_time=query_time), dtype=np.float64)
            if scores.shape != (len(users), self.shape[1]) or not np.isfinite(scores).all():
                raise ValueError("Некорректные предсказания")
            for row, user in enumerate(users):
                known = seen.indices[seen.indptr[user]:seen.indptr[user + 1]]
                scores[row, known] = -np.inf
                if np.isfinite(scores[row]).sum() < k:
                    raise ValueError("Недостаточно кандидатов")
            # Колонки соответствуют лексикографическому порядку ASIN.
            # Stable sort разрешает равенства по item_idx, без случайного шума.
            output.append(np.argsort(-scores, axis=1, kind="stable")[:, :k])
        return np.vstack(output) if output else np.empty((0, k), dtype=np.int64)
