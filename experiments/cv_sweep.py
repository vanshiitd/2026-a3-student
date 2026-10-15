"""experiments/cv_sweep.py -- grouped 5-fold CV over LambdaMART configs on MSLR TRAIN only (spec 2.4)

One fit per (config, fold) gives the whole validation curve over tree counts (predict with num_iteration=n),
so tree count is swept over fixed values and nothing early-stops on the reported fold.
The standardiser question doesn't come up for trees; the query-z transform is computed per query so it
can't leak across folds either.

    python experiments/cv_sweep.py --grid stage1 --workers 5 --threads 2

Rows go to experiments/results/cv_<grid>.jsonl (one per config x fold x tree count), then
summarise with experiments/cv_summary.py.
"""
import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_split, ndcg10_per_query, query_bounds  # noqa: E402

sys.path.insert(0, ROOT)
from harness.cv import grouped_kfold  # noqa: E402
from submission.ltr_common import apply_transform, group_sizes  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results")

BASE = {"objective": "lambdarank", "metric": "None", "learning_rate": 0.1, "num_leaves": 63,
        "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 1.0, "transform": "raw", "max_trees": 1000}


def _grid(name):
    """list of config dicts (override BASE)"""
    if name == "stage1":
        # coarse shape search at lr 0.1: capacity vs regularisation
        out = []
        for leaves, md, ff in itertools.product([15, 31, 63, 127, 255], [50, 200, 800], [0.5, 0.8]):
            out.append({"num_leaves": leaves, "min_data_in_leaf": md, "feature_fraction": ff, "max_trees": 800})
        return out
    if name == "transform":
        return [{"transform": t, "num_leaves": l, "min_data_in_leaf": 200, "max_trees": 800}
                for t in ("raw", "raw+qz") for l in (31, 127)]
    if name == "smoke":
        return [{"num_leaves": 15, "max_trees": 20}, {"num_leaves": 31, "max_trees": 20}]
    raise SystemExit(f"unknown grid {name}")


def _checkpoints(max_trees):
    pts = [10, 25, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 4000]
    return [p for p in pts if p <= max_trees]


def _job(args):
    cfg, fold, n_folds, seed, threads = args
    import lightgbm as lgb
    X, y, q = load_split("train")
    tr_idx, va_idx = grouped_kfold(q, k=n_folds, seed=seed)[fold]
    cfg = dict(BASE, **cfg)
    Xtr, ytr, qtr = X[tr_idx], y[tr_idx], q[tr_idx]
    Xva, yva, qva = X[va_idx], y[va_idx], q[va_idx]
    Xtr = apply_transform(Xtr, qtr, cfg["transform"])
    Xva = apply_transform(Xva, qva, cfg["transform"])
    params = {k: v for k, v in cfg.items() if k not in ("transform", "max_trees")}
    params.update({"seed": seed, "deterministic": True, "force_row_wise": True, "verbose": -1,
                   "num_threads": threads})
    t0 = time.time()
    booster = lgb.train(params, lgb.Dataset(Xtr, label=ytr, group=group_sizes(qtr)),
                        num_boost_round=cfg["max_trees"])
    fit_s = time.time() - t0
    bva, btr = query_bounds(qva), query_bounds(qtr)
    rows = []
    for n in _checkpoints(cfg["max_trees"]):
        va = ndcg10_per_query(booster.predict(Xva, num_iteration=n), yva, bva)
        rec = {"cfg": cfg, "fold": fold, "trees": n, "val": float(va.mean()), "fit_s": fit_s}
        if n in (50, 100, 200, 400, 800, 1500, 3000) or n == cfg["max_trees"]:
            rec["train"] = float(ndcg10_per_query(booster.predict(Xtr, num_iteration=n), ytr, btr).mean())
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    load_split("train")  # build the cache once before forking workers
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"cv_{args.grid}.jsonl")
    jobs = [(c, f, args.folds, args.seed, args.threads) for c in _grid(args.grid) for f in range(args.folds)]
    print(f"{len(jobs)} fits -> {out_path}", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(args.workers) as pool, open(out_path, "a") as out:
        for i, rows in enumerate(pool.imap_unordered(_job, jobs), start=1):
            for r in rows:
                out.write(json.dumps(r) + "\n")
            out.flush()
            print(f"  {i}/{len(jobs)} done  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
