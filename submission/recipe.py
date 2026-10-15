"""
submission/recipe.py -- training recipe for the undisclosed benchmarks (Track C1).

train(train_path, vali_path, out_model_path, seed=0) is run by staff once per hidden benchmark, then the written
model is scored through our own prepare() / rerank_with_ltr() on that benchmark's held-out split.

Steps:
  1. fast-read train + vali, pad both to one width (sparse files can drop trailing all-zero features)
  2. size-based rules instead of MSLR constants (min_data_in_leaf, feature_fraction, tree budget)
  3. small config search, each config early-stopped on the given vali file by nDCG@10
  4. optional refit on train+vali with the picked config and a scaled tree count
  5. save as lightgbm text with our feature names (they encode the feature transform)

Deterministic given the seed: every decision depends on data size, never on wall clock. The only
clock-based thing is an emergency guard that should never fire at <= 500k rows.
OPTIONS holds the shipped choices; experiments/c1_proxy.py compares alternatives via _train(**opts).
"""
import time

import numpy as np

from submission.ltr_common import (apply_transform, feature_names, group_sizes, make_contiguous, pad_to,
                                   read_ltr)

TIME_GUARD_SECONDS = 16 * 60          # staff cap is 20 min
# tree budget in (rows x features used x trees); ~6 min of training on a fast laptop at the worst case,
# so a 2-3x slower grading box still fits the cap
TREE_WORK_BUDGET = 3.3e11

OPTIONS = {
    "transform": "raw",       # 'raw' | 'raw+qz'
    "search": "small",        # 'single' | 'small'
    "refit": "none",          # 'none' | 'trainval'
    "learning_rate": 0.05,
    "es_rounds": 100,
}


def _feature_fraction(n_feat: int) -> float:
    # wide data: sample features per tree (decorrelates trees, cheaper)
    if n_feat <= 50:
        return 0.9
    if n_feat <= 150:
        return 0.7
    return 0.5


def _min_data(n_rows: int) -> int:
    return int(np.clip(n_rows / 2000, 20, 200))


def _configs(n_rows: int, search: str):
    """num_leaves candidates by data size"""
    if n_rows < 30_000:
        leaves = [7, 15, 31]
    elif n_rows < 150_000:
        leaves = [15, 31, 63]
    else:
        leaves = [31, 63, 127]
    if search == "single":
        leaves = [leaves[1]]
    return [{"num_leaves": l, "min_data_in_leaf": _min_data(n_rows)} for l in leaves]


def _base_params(seed: int, n_feat: int, lr: float) -> dict:
    return {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "learning_rate": lr, "lambda_l2": 1.0,
        "feature_fraction": _feature_fraction(n_feat),
        "bagging_fraction": 0.8, "bagging_freq": 1,
        "seed": seed, "deterministic": True, "force_row_wise": True,
        "num_threads": 4, "verbose": -1,
    }


class _TimeGuard:
    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, env):
        if time.monotonic() > self.deadline:
            import lightgbm as lgb
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    _train(train_path, vali_path, out_model_path, seed, **OPTIONS)


def _load(train_path, vali_path):
    Xtr, ytr, qtr = read_ltr(train_path)
    Xva, yva, qva = read_ltr(vali_path)
    width = max(Xtr.shape[1], Xva.shape[1])
    Xtr, ytr, qtr = make_contiguous(pad_to(Xtr, width), ytr, qtr)
    Xva, yva, qva = make_contiguous(pad_to(Xva, width), yva, qva)
    return (Xtr, ytr, qtr), (Xva, yva, qva), width


def _train(train_path, vali_path, out_model_path, seed=0, *, transform="raw", search="small", refit="none",
           learning_rate=0.05, es_rounds=100, return_info=False):
    import lightgbm as lgb

    t_start = time.monotonic()
    (Xtr, ytr, qtr), (Xva, yva, qva), width = _load(train_path, vali_path)
    names = feature_names(width, transform)
    Ttr, Tva = apply_transform(Xtr, qtr, transform), apply_transform(Xva, qva, transform)

    base = _base_params(seed, Ttr.shape[1], learning_rate)
    configs = _configs(len(ytr), search)
    work_per_tree = len(ytr) * Ttr.shape[1] * base["feature_fraction"]
    fits = len(configs) + (1 if refit == "trainval" else 0)
    rounds = int(np.clip(TREE_WORK_BUDGET / work_per_tree / fits, 100, 5000))
    guard = _TimeGuard(t_start + TIME_GUARD_SECONDS)

    dtrain = lgb.Dataset(Ttr, label=ytr, group=group_sizes(qtr), feature_name=names, free_raw_data=False)
    dvali = lgb.Dataset(Tva, label=yva, group=group_sizes(qva), feature_name=names, reference=dtrain)
    tried = []
    for cfg in configs:
        params = dict(base, **cfg)
        booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvali], valid_names=["vali"],
                            callbacks=[lgb.early_stopping(es_rounds, verbose=False), guard])
        tried.append((booster.best_score["vali"]["ndcg@10"], booster.best_iteration, cfg, booster))
    best_score, best_iter, best_cfg, best_booster = max(tried, key=lambda t: t[0])

    if refit == "trainval":
        # same config, more data: trees scaled by the data ratio
        n_trees = int(round(best_iter * (len(ytr) + len(yva)) / len(ytr)))
        Tall = np.vstack([Ttr, Tva])
        yall = np.concatenate([ytr, yva])
        gall = group_sizes(qtr) + group_sizes(qva)
        model = lgb.train(dict(base, **best_cfg), lgb.Dataset(Tall, label=yall, group=gall, feature_name=names),
                          num_boost_round=max(n_trees, 1), callbacks=[guard])
        model.save_model(out_model_path)
    else:
        best_booster.save_model(out_model_path, num_iteration=best_iter)

    if return_info:
        return {"rounds_cap": rounds, "tried": [(round(s, 4), it, c) for s, it, c, _ in tried],
                "picked": best_cfg, "best_iter": best_iter, "seconds": time.monotonic() - t_start}
