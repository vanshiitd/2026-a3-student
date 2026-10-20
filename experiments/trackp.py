"""experiments/trackp.py -- pointwise / pairwise MLP variants for Track P.

Early stopping uses a 10% query holdout carved out of TRAIN (never DEV), so DEV stays an untouched
read for the report. Each run prints holdout nDCG@10 per epoch and the DEV nDCG@10 of the restored
best weights. Results append to experiments/results/trackp.jsonl.

    python experiments/trackp.py --variant base --variant lambda ...
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_split, ndcg10  # noqa: E402

sys.path.insert(0, ROOT)
import torch  # noqa: E402

from harness.cv import grouped_kfold  # noqa: E402
from submission import pointwise_pairwise as pp  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results", "trackp.jsonl")

VARIANTS = {
    "pointwise": {"kind": "pointwise"},
    "base": {},
    "lambda": {"weighting": "lambda"},
    "lambda10": {"weighting": "lambda", "lambda_top": 10},
    "gain": {"sampler": "gain"},
    "robust": {"scaler": "robust"},
    "wide": {"hidden": [256, 128]},
    "deep": {"hidden": [256, 256, 128]},
    "ppq1000": {"pairs_per_query": 1000},
    "lambda_deep": {"weighting": "lambda", "hidden": [256, 256, 128]},
    "lambda_deep_robust": {"weighting": "lambda", "hidden": [256, 256, 128], "scaler": "robust"},
    "lr3": {"lr": 3e-4},
    "robust_lr3": {"scaler": "robust", "lr": 3e-4},
    "deep_lr3": {"hidden": [256, 256, 128], "lr": 3e-4},
    "deep_robust_lr3": {"hidden": [256, 256, 128], "scaler": "robust", "lr": 3e-4},
    "lambda_deep_robust_lr3": {"weighting": "lambda", "hidden": [256, 256, 128], "scaler": "robust", "lr": 3e-4},
    "bs4096": {"batch_size": 4096},
    "ppq1000_lr3": {"pairs_per_query": 1000, "lr": 3e-4},
    "gain_lr3": {"sampler": "gain", "lr": 3e-4},
    "wide_lr3": {"hidden": [256, 128], "lr": 3e-4},
    "deep_ppq1000_lr3": {"hidden": [256, 256, 128], "pairs_per_query": 1000, "lr": 3e-4},
    "gain2_lr3": {"sampler": "gain2", "lr": 3e-4},
    "gain_ppq600_lr3": {"sampler": "gain", "pairs_per_query": 600, "lr": 3e-4},
    "gain_ppq150_lr3": {"sampler": "gain", "pairs_per_query": 150, "lr": 3e-4},
    "gain_lambda_lr3": {"sampler": "gain", "weighting": "lambda", "lr": 3e-4},
    "gain_lr3_e50": {"sampler": "gain", "lr": 3e-4, "epochs": 50, "patience": 6},
    "gain_wide_lr3": {"sampler": "gain", "hidden": [128, 64], "lr": 3e-4},
    "gain_ema": {"sampler": "gain", "lr": 3e-4, "epochs": 50, "patience": 6, "ema": 0.999},
    "gain_cos": {"sampler": "gain", "lr": 3e-4, "epochs": 30, "patience": 30, "cosine": True},
    "gain_ema_lr1e3": {"sampler": "gain", "lr": 1e-3, "epochs": 50, "patience": 6, "ema": 0.999},
    "gain_ema_wide": {"sampler": "gain", "lr": 3e-4, "epochs": 50, "patience": 6, "ema": 0.999, "hidden": [256, 128]},
}
DEFAULTS = {"kind": "pairwise", "epochs": 30, "patience": 4, "lr": 1e-3, "batch_size": 1024,
            "pairs_per_query": 300, "hidden": 64}


def split_train(seed=0):
    X, y, q = load_split("train")
    tr_idx, ho_idx = grouped_kfold(q, k=10, seed=seed)[0]
    return (X[tr_idx], y[tr_idx], q[tr_idx]), (X[ho_idx], y[ho_idx], q[ho_idx])


def run(name, seed, threads):
    torch.set_num_threads(threads)
    cfg = dict(DEFAULTS, **VARIANTS[name])
    kind = cfg.pop("kind")
    train, holdout = split_train()
    t0 = time.time()
    net, std, hist = pp.train_model(kind, train, holdout, seed=seed, verbose=True, **cfg)
    secs = time.time() - t0
    Xd, yd, qd = load_split("dev")
    dev = ndcg10(pp.make_scorer(net, std)(Xd), yd, qd)
    best = max(hist, key=lambda h: h["dev_ndcg@10"])
    rec = {"variant": name, "seed": seed, "cfg": {k: v for k, v in cfg.items()}, "kind": kind,
           "holdout_best": best["dev_ndcg@10"], "best_epoch": best["epoch"], "epochs_run": len(hist),
           "dev_ndcg10": dev, "seconds": secs, "history": hist}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{name:<20} holdout {best['dev_ndcg@10']:.4f} @ep{best['epoch']}  DEV {dev:.4f}  {secs:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    for v in args.variant:
        run(v, args.seed, args.threads)


if __name__ == "__main__":
    main()
