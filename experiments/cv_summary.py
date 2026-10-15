"""experiments/cv_summary.py -- summarise cv_sweep.jsonl: per config x tree count,
val mean +- sd over folds, train mean, val/train ratio. Best rows first.

    python experiments/cv_summary.py experiments/results/cv_stage1.jsonl --top 20
"""
import argparse
import json
from collections import defaultdict

import numpy as np

KEYS = ["transform", "learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
        "bagging_fraction", "lambda_l2", "max_depth", "lambdarank_truncation_level", "min_sum_hessian_in_leaf",
        "extra_trees", "label_gain"]


def cfg_key(cfg):
    return tuple((k, json.dumps(cfg.get(k))) for k in KEYS if k in cfg)


def load(path):
    groups = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        groups[cfg_key(r["cfg"])][r["trees"]].append(r)
    return groups


def table(groups):
    n_folds = max(len(recs) for by_trees in groups.values() for recs in by_trees.values())
    rows = []
    for key, by_trees in groups.items():
        for trees, recs in by_trees.items():
            if len(recs) < n_folds:
                continue
            val = np.array([r["val"] for r in recs])
            tr = [r["train"] for r in recs if "train" in r]
            rows.append({"cfg": dict((k, json.loads(v)) for k, v in key), "trees": trees,
                         "val": val.mean(), "sd": val.std(ddof=1), "train": np.mean(tr) if tr else np.nan})
    return sorted(rows, key=lambda r: -r["val"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    rows = table(load(args.path))
    print(f"{'val':>7} {'sd':>6} {'train':>7} {'ratio':>6} {'trees':>6}  config")
    for r in rows[:args.top]:
        ratio = r["val"] / r["train"] if r["train"] == r["train"] else float("nan")
        c = {k: v for k, v in r["cfg"].items() if k not in ("bagging_fraction",)}
        print(f"{r['val']:.4f} {r['sd']:.4f} {r['train']:7.4f} {ratio:6.3f} {r['trees']:>6}  {c}")


if __name__ == "__main__":
    main()
