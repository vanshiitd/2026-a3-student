"""
submission/recipe.py -- training recipe for the undisclosed benchmarks (Track C1).

train(train_path, vali_path, out_model_path, seed=0) is run by staff once per hidden benchmark, then the written
model is scored through our own prepare() / rerank_with_ltr() on that benchmark's held-out split.

What the recipe does (v1, numbers get tuned on MSLR TRAIN proxies, see README):
  1. fast-read train + vali, pad both to one width (sparse files can drop trailing all-zero features)
  2. size-based hyperparameter rules instead of MSLR constants (leaves / min_data / feature_fraction / rounds)
  3. small config search, each config early-stopped on the given vali file by nDCG@10
  4. save best model as lightgbm text with our feature names (encodes the feature transform)

No wall-clock based decisions in normal runs, so it's deterministic given the seed. There's only an
emergency time guard that should never fire on <= 500k rows.
"""
import time

import numpy as np

from submission.ltr_common import (apply_transform, feature_names, group_sizes, make_contiguous, pad_to,
                                   read_ltr)

TIME_GUARD_SECONDS = 17 * 60  # staff cap is 20 min; leave room for reading + saving


def _base_params(seed: int, n_rows: int, n_feat: int) -> dict:
    return {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "learning_rate": 0.05, "lambda_l2": 1.0,
        "feature_fraction": 0.8 if n_feat <= 100 else 0.5,
        "bagging_fraction": 0.8, "bagging_freq": 1,
        "seed": seed, "deterministic": True, "force_row_wise": True,
        "num_threads": 4, "verbose": -1,
    }


def _configs(n_rows: int):
    """(num_leaves, min_data_in_leaf) candidates, scaled with data size"""
    min_data = int(np.clip(n_rows / 2000, 20, 200))
    if n_rows < 50_000:
        return [(15, min_data), (31, min_data)]
    return [(31, min_data), (63, min_data), (127, min_data)]


def _max_rounds(n_rows: int, n_feat: int) -> int:
    # rough cost model so big wide data can't blow the time cap
    work = n_rows * max(n_feat, 1)
    return int(np.clip(3e9 / work, 300, 3000))


class _TimeGuard:
    def __init__(self, deadline):
        self.deadline = deadline

    def __call__(self, env):
        if time.monotonic() > self.deadline:
            import lightgbm as lgb
            raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)


def train(train_path: str, vali_path: str, out_model_path: str, seed: int = 0) -> None:
    import lightgbm as lgb

    t_start = time.monotonic()
    Xtr, ytr, qtr = read_ltr(train_path)
    Xva, yva, qva = read_ltr(vali_path)
    width = max(Xtr.shape[1], Xva.shape[1])
    Xtr, Xva = pad_to(Xtr, width), pad_to(Xva, width)
    Xtr, ytr, qtr = make_contiguous(Xtr, ytr, qtr)
    Xva, yva, qva = make_contiguous(Xva, yva, qva)

    transform = "raw"
    names = feature_names(width, transform)
    Ttr, Tva = apply_transform(Xtr, qtr, transform), apply_transform(Xva, qva, transform)

    dtrain = lgb.Dataset(Ttr, label=ytr, group=group_sizes(qtr), feature_name=names, free_raw_data=False)
    dvali = lgb.Dataset(Tva, label=yva, group=group_sizes(qva), feature_name=names, reference=dtrain)

    base = _base_params(seed, len(ytr), width)
    rounds = _max_rounds(len(ytr), Ttr.shape[1])
    guard = _TimeGuard(t_start + TIME_GUARD_SECONDS)

    best = None
    for leaves, min_data in _configs(len(ytr)):
        params = dict(base, num_leaves=leaves, min_data_in_leaf=min_data)
        booster = lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dvali], valid_names=["vali"],
                            callbacks=[lgb.early_stopping(100, verbose=False), guard])
        score = booster.best_score["vali"]["ndcg@10"]
        if best is None or score > best[0]:
            best = (score, booster)
    best[1].save_model(out_model_path, num_iteration=best[1].best_iteration)
