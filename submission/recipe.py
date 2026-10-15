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

from submission.ltr_common import (WidthError, apply_transform, feature_names, group_sizes, make_contiguous,
                                   pad_to, read_into, read_ltr)

TIME_GUARD_SECONDS = 16 * 60          # staff cap is 20 min
# tree budget in (rows x features used x trees). 3.3e11 took 6 min on a fast laptop at the worst case
# (500k x 300, query-z doubles it to 600), too close to 20 min on a 2-3x slower box -> 2.2e11 (~4 min here)
TREE_WORK_BUDGET = 2.2e11
HIST_POOL_MB = 1024  # cap lightgbm's per-leaf histogram cache (it grows with leaves x features x bins)

# picked on 5 MSLR-TRAIN-derived proxies (experiments/c1_proxy.py, notes F9): query-z features +0.006,
# train+vali refit +0.005, both +0.009 mean test nDCG@10, best on 4/5 proxies
OPTIONS = {
    "transform": "raw+qz",    # 'raw' | 'raw+qz'
    "search": "small",        # 'single' | 'small'
    "refit": "trainval",      # 'none' | 'trainval'
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
        "num_threads": 4, "verbose": -1, "histogram_pool_size": HIST_POOL_MB,
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


def _load_matrix(train_path, vali_path, transform):
    """train + vali stacked into one model-input matrix (train rows first) plus per-file query sizes.
    Normal path parses straight into one preallocated array (peak ~1 copy of the data); if a file
    lists feature indices out of order, fall back to the simpler reader (needs ~3 copies)."""
    try:
        T, y, (g_tr, g_va), width = read_into([train_path, vali_path], transform)
        return T, y, g_tr, g_va, width
    except WidthError:
        Xtr, ytr, qtr = read_ltr(train_path)
        Xva, yva, qva = read_ltr(vali_path)
        width = max(Xtr.shape[1], Xva.shape[1])
        Xtr, ytr, qtr = make_contiguous(pad_to(Xtr, width), ytr, qtr)
        Xva, yva, qva = make_contiguous(pad_to(Xva, width), yva, qva)
        g_tr, g_va = group_sizes(qtr), group_sizes(qva)
        X = np.vstack([Xtr, Xva])
        del Xtr, Xva
        T = apply_transform(X, None, transform, sizes=g_tr + g_va)
        return T, np.concatenate([ytr, yva]), g_tr, g_va, width


def _train(train_path, vali_path, out_model_path, seed=0, *, transform="raw", search="small", refit="none",
           learning_rate=0.05, es_rounds=100, return_info=False):
    import lightgbm as lgb

    t_start = time.monotonic()
    T_all, y_all, g_tr, g_va, width = _load_matrix(train_path, vali_path, transform)
    names = feature_names(width, transform)
    n_tr = sum(g_tr)
    Ttr, Tva = T_all[:n_tr], T_all[n_tr:]  # row views, no copies
    ytr, yva = y_all[:n_tr], y_all[n_tr:]

    base = _base_params(seed, T_all.shape[1], learning_rate)
    configs = _configs(n_tr, search)
    work_per_tree = n_tr * T_all.shape[1] * base["feature_fraction"]
    fits = len(configs) + (1 if refit == "trainval" else 0)
    rounds = int(np.clip(TREE_WORK_BUDGET / work_per_tree / fits, 100, 5000))
    guard = _TimeGuard(t_start + TIME_GUARD_SECONDS)

    dtrain = lgb.Dataset(Ttr, label=ytr, group=g_tr, feature_name=names)
    dvali = lgb.Dataset(Tva, label=yva, group=g_va, feature_name=names, reference=dtrain)
    tried, best = [], None
    for cfg in configs:
        params = dict(base, **cfg)
        booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvali], valid_names=["vali"],
                            callbacks=[lgb.early_stopping(es_rounds, verbose=False), guard])
        score = booster.best_score["vali"]["ndcg@10"]
        tried.append((score, booster.best_iteration, cfg))
        if best is None or score > best[0]:  # ties keep the earlier config
            best = (score, booster.best_iteration, cfg, booster)
        del booster  # only the best booster stays alive (each one holds its own histogram cache)
    best_score, best_iter, best_cfg, best_booster = best
    del best

    if refit == "trainval":
        # same config, more data: trees scaled by the data ratio
        del dtrain, dvali, best_booster
        n_trees = int(round(best_iter * len(y_all) / n_tr))
        model = lgb.train(dict(base, **best_cfg),
                          lgb.Dataset(T_all, label=y_all, group=g_tr + g_va, feature_name=names),
                          num_boost_round=max(n_trees, 1), callbacks=[guard])
        model.save_model(out_model_path)
    else:
        best_booster.save_model(out_model_path, num_iteration=best_iter)

    if return_info:
        return {"rounds_cap": rounds, "tried": [(round(s, 4), it, c) for s, it, c in tried],
                "picked": best_cfg, "best_iter": best_iter, "seconds": time.monotonic() - t_start}
