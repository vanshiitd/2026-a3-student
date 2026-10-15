"""experiments/robustness.py -- a guess-free hedge for Track C2.

Staff score the shipped model on five undisclosed "versions" of the MSLR held-out split. We can't know how
they're built, so this only checks whether configs that tie on clean grouped CV differ under a few generic
shifts a test set could plausibly have. It's a tie-breaker between CV-equivalent configs, not an objective
to optimise on its own.

Shifts (applied to a CV validation fold only; training data untouched):
  noise    each value *(1 + 0.1 * N(0,1))
  dropcol  10% of feature columns zeroed for every row
  missing  10% of individual values zeroed
  scale    20% of columns multiplied by a random factor in [0.5, 2]
  subpool  each query keeps a random 50% of its candidates (labels kept)

    python experiments/robustness.py --configs experiments/robust_configs.json --folds 0 1 2
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_split, ndcg10  # noqa: E402

sys.path.insert(0, ROOT)
from harness.cv import grouped_kfold  # noqa: E402
from submission.ltr_common import apply_transform, group_sizes  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results", "robustness.jsonl")
SHIFTS = ["clean", "noise", "dropcol", "missing", "scale", "subpool"]


def shift(name, X, y, q, rng):
    if name == "clean":
        return X, y, q
    X = X.copy()
    if name == "noise":
        X *= (1 + 0.1 * rng.randn(*X.shape)).astype(np.float32)
    elif name == "dropcol":
        X[:, rng.choice(X.shape[1], X.shape[1] // 10, replace=False)] = 0
    elif name == "missing":
        X[rng.rand(*X.shape) < 0.1] = 0
    elif name == "scale":
        cols = rng.choice(X.shape[1], X.shape[1] // 5, replace=False)
        X[:, cols] *= rng.uniform(0.5, 2, len(cols)).astype(np.float32)
    elif name == "subpool":
        keep = rng.rand(len(y)) < 0.5
        return X[keep], y[keep], q[keep]
    return X, y, q


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True, help="json list of {name, params, transform, trees}")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    configs = json.load(open(args.configs))
    X, y, q = load_split("train")
    folds = grouped_kfold(q, k=5, seed=0)
    for fold in args.folds:
        tr, va = folds[fold]
        for cfg in configs:
            tf = cfg.get("transform", "raw")
            params = dict(objective="lambdarank", metric="None", seed=0, deterministic=True, force_row_wise=True,
                          verbose=-1, num_threads=args.threads, bagging_freq=1, **cfg["params"])
            model = lgb.train(params, lgb.Dataset(apply_transform(X[tr], q[tr], tf), label=y[tr],
                                                  group=group_sizes(q[tr])), num_boost_round=cfg["trees"])
            rec = {"name": cfg["name"], "fold": fold}
            for s in SHIFTS:
                rng = np.random.RandomState(1000 + fold)
                Xs, ys, qs = shift(s, X[va], y[va], q[va], rng)
                rec[s] = ndcg10(model.predict(apply_transform(Xs, qs, tf)), ys, qs)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec["name"], fold, {s: round(rec[s], 4) for s in SHIFTS}, flush=True)


if __name__ == "__main__":
    main()
