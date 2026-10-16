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
# tree budget in (rows x features used x trees), split over the search fits + refit.
# 2.2e11 = 249 s here on the synthetic worst case (500k x 300 -> 600 cols with query-z, no early stop).
# Tried 2.9e11 + raising lr when the cap binds (notes F13): on real-shaped 500k x 300 data the picked
# 127-leaf config early-stops well under the cap anyway, and the higher lr made it overshoot -> test
# nDCG@10 0.4799 vs 0.4855. Kept 2.2e11 and a fixed lr.
TREE_WORK_BUDGET = 2.2e11
MIN_TREES = 50        # floor: early stopping on a saturated vali can otherwise ship 1-8 trees (critique M3)
HIST_POOL_MB = 512    # cap lightgbm's per-leaf histogram cache (it grows with leaves x features x bins)

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
    """emergency stop near the staff time cap; remembers that it fired so the caller can fall back"""

    def __init__(self, deadline):
        self.deadline = deadline
        self.fired = False

    def __call__(self, env):
        if time.monotonic() > self.deadline:
            import lightgbm as lgb
            self.fired = True
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

    configs = _configs(n_tr, search)
    work_per_tree = n_tr * T_all.shape[1] * _feature_fraction(T_all.shape[1])
    fits = len(configs) + (1 if refit == "trainval" else 0)
    rounds = int(np.clip(TREE_WORK_BUDGET / work_per_tree / fits, 100, 5000))
    base = _base_params(seed, T_all.shape[1], learning_rate)
    guard = _TimeGuard(t_start + TIME_GUARD_SECONDS)

    dtrain = lgb.Dataset(Ttr, label=ytr, group=g_tr, feature_name=names)
    dvali = lgb.Dataset(Tva, label=yva, group=g_va, feature_name=names, reference=dtrain)
    tried, best = [], None   # best = (vali score, trees to keep, config, model text)
    for cfg in configs:
        if guard.fired:
            break  # out of time: keep what we have
        # keep_training_booster: otherwise lgb.train trims the returned model to best_iteration and the
        # tree floor below would have nothing extra to keep
        booster = lgb.train(dict(base, **cfg), dtrain, num_boost_round=rounds, valid_sets=[dvali],
                            valid_names=["vali"], keep_training_booster=True,
                            callbacks=[lgb.early_stopping(es_rounds, verbose=False), guard])
        score = booster.best_score["vali"]["ndcg@10"]
        # floor on the tree count, but never more trees than were actually grown
        keep = max(booster.best_iteration, min(MIN_TREES, booster.current_iteration()))
        tried.append((score, booster.best_iteration, cfg))
        if best is None or score > best[0]:  # ties keep the earlier config
            # keep the best model as text, not as a live booster (that one holds its own histogram cache)
            best = (score, keep, cfg, booster.model_to_string(num_iteration=keep))
        del booster
    best_score, keep, best_cfg, best_text = best

    refit_done = False
    if refit == "trainval" and not guard.fired:
        # same config, more data: trees scaled by the data ratio
        del dtrain, dvali
        n_trees = int(round(keep * len(y_all) / n_tr))
        model = lgb.train(dict(base, **best_cfg),
                          lgb.Dataset(T_all, label=y_all, group=g_tr + g_va, feature_name=names),
                          num_boost_round=n_trees, callbacks=[guard])
        if not guard.fired:  # a refit cut short by the guard is worse than the finished search model
            model.save_model(out_model_path)
            refit_done = True
    if not refit_done:
        with open(out_model_path, "w") as f:
            f.write(best_text)

    if return_info:
        return {"rounds_cap": rounds, "lr": learning_rate, "tried": [(round(s, 4), it, c) for s, it, c in tried],
                "picked": best_cfg, "best_iter": keep, "refit_done": refit_done, "guard_fired": guard.fired,
                "seconds": time.monotonic() - t_start}
