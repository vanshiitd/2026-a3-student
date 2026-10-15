"""Train the LambdaMART artifact that submission/ranking.py loads (offline; grading never retrains).

    python scripts/train_model.py --train data/train --dev data/dev --out submission/model.txt

Defaults = what we ship (picked by grouped 5-fold CV on TRAIN only, experiments/cv_sweep.py, notes F8-F12).
--train takes a split file or a directory holding train.txt. --dev is only used to PRINT a dev score after
training; nothing is tuned or early-stopped on it. Pass --use-dev to ALSO train on dev (raises dev nDCG,
Tracks A/B, without helping on unseen queries; declare it, Section 8) -- the shipped model does NOT.

Reproducible: same data + same flags -> byte-identical model.txt (deterministic=True, fixed seed and
thread count). The model's feature names encode the feature transform, which prepare() reads back.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import numpy as np

from harness.metrics import mean_ndcg_from_scores
from submission.ltr_common import apply_transform, feature_names, group_sizes, read_ltr


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default=None, help="dev file/dir; printed score only, unless --use-dev")
    ap.add_argument("--out", default="submission/model.txt")
    ap.add_argument("--transform", choices=["raw", "raw+qz"], default="raw+qz")
    ap.add_argument("--trees", type=int, default=600)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=200)
    ap.add_argument("--feature-fraction", type=float, default=0.5)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--use-dev", action="store_true", help="also train on dev (declare it)")
    args = ap.parse_args()

    import lightgbm as lgb

    X, y, q = read_ltr(_resolve(args.train, "train.txt"))
    dev = None
    if args.dev:
        dev = read_ltr(_resolve(args.dev, "vali.txt"), n_features=X.shape[1])
    if args.use_dev:
        if dev is None:
            raise SystemExit("--use-dev needs --dev")
        X, y, q = np.vstack([X, dev[0]]), np.concatenate([y, dev[1]]), np.concatenate([q, dev[2]])
    width = X.shape[1]
    params = {
        "objective": "lambdarank", "metric": "None",
        "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf, "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction, "bagging_freq": 1, "lambda_l2": args.lambda_l2,
        "seed": args.seed, "deterministic": True, "force_row_wise": True, "num_threads": args.threads,
        "verbose": -1,
    }
    T = apply_transform(X, q, args.transform)
    model = lgb.train(params, lgb.Dataset(T, label=y, group=group_sizes(q),
                                          feature_name=feature_names(width, args.transform)),
                      num_boost_round=args.trees)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save_model(args.out)
    print(f"trained on {len(set(q.tolist()))} queries ({'train+dev' if args.use_dev else 'train only'}), "
          f"transform={args.transform}, {args.trees} trees -> {args.out}")
    print(f"train nDCG@10 = {mean_ndcg_from_scores(model.predict(T), y, q):.4f}")
    if dev is not None and not args.use_dev:
        Xd, yd, qd = dev
        print(f"dev   nDCG@10 = {mean_ndcg_from_scores(model.predict(apply_transform(Xd, qd, args.transform)), yd, qd):.4f}")


if __name__ == "__main__":
    main()
