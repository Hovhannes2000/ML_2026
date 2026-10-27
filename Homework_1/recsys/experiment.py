"""Параметры фиксируются до test; подбор использует исключительно validation."""

from pathlib import Path
import itertools
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
from models import ItemKNN, IALS, EASE, FactorizationMachine, CustomTargetALS
from .data import TRAIN_END, VALID_END
from .evaluation import make_protocol, evaluate, paired_bootstrap

SEED = 42
PRIMARY_METRIC = "NDCG@10"
MODEL_CLASSES = {"ItemKNN": ItemKNN, "iALS": IALS, "EASE": EASE,
                 "FM": FactorizationMachine, "CustomALS": CustomTargetALS}


def product_grid(**grid):
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


SEARCH_SPACE = {
    "ItemKNN": product_grid(neighbors=[40, 100], shrinkage=[10.0, 50.0]),
    "iALS": product_grid(factors=[24, 48], regularization=[1.0, 10.0], alpha=[20.0], iterations=[10], seed=[SEED]),
    "EASE": product_grid(regularization=[20.0, 100.0, 500.0]),
    "FM": product_grid(factors=[16, 32], regularization=[1e-4, 1e-3], learning_rate=[0.05],
                       epochs=[12], batch_size=[2048], metadata=[True], seed=[SEED]),
    "CustomALS": product_grid(factors=[24, 48], regularization=[1.0, 10.0], alpha=[20.0],
                              iterations=[10], seed=[SEED], half_life_days=[730.0]),
}


def tune(bundle, output="results"):
    output = Path(output)
    output.mkdir(exist_ok=True)
    protocol = make_protocol(bundle["train"], bundle["validation"], bundle["shape"], TRAIN_END)
    records, selected = [], {}
    for name, configurations in SEARCH_SPACE.items():
        best_score = -np.inf
        for number, config in enumerate(configurations, 1):
            start = time.perf_counter()
            model = MODEL_CLASSES[name](**config).fit(bundle["train"], bundle["shape"], reference_time=TRAIN_END)
            fit_seconds = time.perf_counter() - start
            metrics, _, _ = evaluate(model, protocol)
            record = {"model": name, "trial": number, **metrics, "fit_seconds": fit_seconds,
                      "parameters": json.dumps(config, sort_keys=True)}
            records.append(record)
            print(f"{name} {number}/{len(configurations)}: val NDCG@10={metrics[PRIMARY_METRIC]:.6f}, "
                  f"Recall@10={metrics['Recall@10']:.6f}, fit={fit_seconds:.1f}s", flush=True)
            # В равенствах остаётся первый набор из заранее заданной сетки.
            if metrics[PRIMARY_METRIC] > best_score:
                best_score = metrics[PRIMARY_METRIC]
                selected[name] = config
            pd.DataFrame(records).to_csv(output / "validation.csv", index=False)
            del model
    frozen = {"seed": SEED, "primary_metric": PRIMARY_METRIC, "search_space": SEARCH_SPACE,
              "selected": selected, "validation_users": len(protocol["users"]),
              "train_end_utc": "2015-01-01", "validation_end_utc": "2016-01-01",
              "ablations": ["FM without metadata at selected FM parameters",
                            "CustomALS at selected baseline iALS parameters",
                            "Signed rating ALS without temporal decay at selected baseline iALS parameters"],
              "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
              "scipy": scipy.__version__}
    # Этот файл записывается ДО первого расчёта test-метрик.
    (output / "selected.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    return pd.DataFrame(records), selected, protocol


def final_comparison(bundle, selected, output="results"):
    output = Path(output)
    frozen = json.loads((output / "selected.json").read_text())
    assert selected == frozen["selected"]
    protocol = make_protocol(bundle["train_validation"], bundle["test"], bundle["shape"], VALID_END)
    records, per_users, recommendations, models = [], {}, {}, {}
    for name in MODEL_CLASSES:
        start = time.perf_counter()
        model = MODEL_CLASSES[name](**selected[name]).fit(bundle["train_validation"], bundle["shape"], reference_time=VALID_END)
        fit_seconds = time.perf_counter() - start
        metric_start = time.perf_counter()
        metrics, per_user, recs = evaluate(model, protocol)
        records.append({"model": name, **metrics, "fit_seconds": fit_seconds,
                        "evaluation_seconds": time.perf_counter() - metric_start,
                        "parameters": json.dumps(selected[name], sort_keys=True)})
        per_users[name], recommendations[name], models[name] = per_user, recs, model
        per_user.to_csv(output / f"test_users_{name}.csv", index=False)
        print(f"{name}: test NDCG@10={metrics[PRIMARY_METRIC]:.6f}, Recall@10={metrics['Recall@10']:.6f}", flush=True)
    table = pd.DataFrame(records).sort_values(PRIMARY_METRIC, ascending=False, kind="stable").reset_index(drop=True)
    table.to_csv(output / "test.csv", index=False)
    return table, per_users, recommendations, models, protocol


def ablations(bundle, selected, final_models, final_per_users, protocol, output="results"):
    output = Path(output)
    controlled = {
        "FM_ID_only": (FactorizationMachine, {**selected["FM"], "metadata": False}),
        "CustomALS_matched": (CustomTargetALS, {**selected["iALS"], "half_life_days": 730.0}),
        "SignedALS_no_decay": (CustomTargetALS, {**selected["iALS"], "half_life_days": float("inf")}),
    }
    per_users = dict(final_per_users)
    rows = []
    for name, (cls, config) in controlled.items():
        if name == "CustomALS_matched" and config == selected["CustomALS"]:
            model = final_models["CustomALS"]
        else:
            model = cls(**config).fit(bundle["train_validation"], bundle["shape"], reference_time=VALID_END)
        metrics, per_user, _ = evaluate(model, protocol)
        per_users[name] = per_user
        rows.append({"model": name, **metrics, "parameters": json.dumps(config, sort_keys=True)})
        per_user.to_csv(output / f"test_users_{name}.csv", index=False)
        print(f"Ablation {name}: NDCG@10={metrics[PRIMARY_METRIC]:.6f}", flush=True)
    comparisons = []
    for left, right in [("CustomALS", "iALS"), ("CustomALS_matched", "iALS"),
                        ("CustomALS_matched", "SignedALS_no_decay"), ("FM", "FM_ID_only")]:
        comparisons.append({"comparison": f"{left} - {right}", **paired_bootstrap(per_users[left], per_users[right])})
    table, intervals = pd.DataFrame(rows), pd.DataFrame(comparisons)
    table.to_csv(output / "ablations.csv", index=False)
    intervals.to_csv(output / "paired_bootstrap.csv", index=False)
    return table, intervals
