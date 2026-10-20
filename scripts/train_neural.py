"""Train the pointwise and/or pairwise (RankNet) MLP scorers and save pointwise_model.pt / pairwise_model.pt.

    python scripts/train_neural.py --kind both --train data/train --dev data/dev --out-dir submission

Reproduces the shipped artifacts (defaults below = what we ship). Early stopping is on a 10% query holdout
carved out of TRAIN (grouped_kfold, seed 0), NOT on dev, so dev stays an untouched read; pass
--early-stop dev to early-stop on --dev instead (that counts as tuning on dev -- declare it).

Shipped settings (picked on the TRAIN holdout, experiments/trackp.py, notes F6-F7):
  pairwise : RankNet loss, pairs sampled per query with probability ~ 2^l_hi - 2^l_lo (gain sampler),
             300 pairs/query/epoch, 64 hidden, Adam lr 3e-4, batch 1024, <= 50 epochs, patience 6
  pointwise: textbook squared error, same net/optimiser settings, uniform over documents
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import numpy as np

from harness.cv import grouped_kfold
from harness.feature_io import read_svmlight_ltr, stack
from submission.pointwise_pairwise import dev_ndcg, make_scorer, save_scorer, train_model


def _resolve(path, name):
    return os.path.join(path, name) if os.path.isdir(path) else path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["pointwise", "pairwise", "both"], default="both")
    ap.add_argument("--train", default="data/toy")
    ap.add_argument("--dev", default="data/toy")
    ap.add_argument("--out-dir", default="submission")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pairs-per-query", type=int, default=300)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64],
                    help="hidden layer widths, e.g. --hidden 128 64 (at most 3 layers of at most 256 units)")
    ap.add_argument("--sampler", choices=["uniform", "gain", "gain2"], default="gain")
    ap.add_argument("--weighting", choices=["none", "lambda"], default="none")
    ap.add_argument("--early-stop", choices=["train-holdout", "dev"], default="train-holdout")
    ap.add_argument("--use-dev", action="store_true",
                    help="also train on dev (declare it); early stopping then uses a 10%% holdout of TRAIN+DEV")
    args = ap.parse_args()

    train = stack(read_svmlight_ltr(_resolve(args.train, "train.txt")))
    dev = stack(read_svmlight_ltr(_resolve(args.dev, "vali.txt")))
    if args.use_dev:
        if args.early_stop == "dev":
            raise SystemExit("--use-dev with --early-stop dev would stop on training data")
        train = tuple(np.concatenate([a, b]) for a, b in zip(train, dev))
    if args.early_stop == "train-holdout":
        X, y, q = train
        fit_idx, ho_idx = grouped_kfold(q, k=10, seed=0)[0]
        fit, stop = (X[fit_idx], y[fit_idx], q[fit_idx]), (X[ho_idx], y[ho_idx], q[ho_idx])
    else:
        fit, stop = train, dev

    os.makedirs(args.out_dir, exist_ok=True)
    for kind in (["pointwise", "pairwise"] if args.kind == "both" else [args.kind]):
        extra = {"sampler": args.sampler, "weighting": args.weighting} if kind == "pairwise" else {}
        net, std, hist = train_model(kind, fit, stop, seed=args.seed, epochs=args.epochs,
                                     batch_size=args.batch_size, lr=args.lr,
                                     pairs_per_query=args.pairs_per_query, patience=args.patience,
                                     hidden=args.hidden, verbose=True, **extra)
        path = os.path.join(args.out_dir, f"{kind}_model.pt")
        save_scorer(path, net, std, args.hidden)
        best = max(h["dev_ndcg@10"] for h in hist)
        print(f"{kind}: epochs run = {len(hist)}, best early-stop-split nDCG@10 = {best:.4f}, "
              f"dev nDCG@10 = {dev_ndcg(make_scorer(net, std), *dev):.4f} -> {path}")


if __name__ == "__main__":
    main()
