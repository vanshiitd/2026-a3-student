"""Train the LambdaMART artifact that submission/ranking.py loads (offline; grading never retrains).

    python scripts/train_model.py --train data/train --dev data/dev --out submission/model.txt

Defaults = what we ship: a blend of (a) the query-z LambdaMART picked by grouped 5-fold CV on TRAIN
(experiments/cv_sweep.py) and (b) a raw-features-only LambdaMART, merged into one model file. The blend was
chosen among 3 candidates by TRAIN CV under generic shifts AND the live leaderboard's C2 score
(experiments/c2_candidates.py, notes F14-F15). --no-blend gives the single query-z model.
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


def merge_models(text_a: str, text_b: str) -> str:
    """one lightgbm text model whose prediction is model_a + model_b: trees of b appended after a's.
    Both must share the same feature list; per-feature ranges are unioned."""
    def split(text):
        head, rest = text.split("\nTree=0\n", 1)
        trees, tail = rest.split("\nend of trees\n", 1)
        blocks = ("Tree=0\n" + trees).split("\n\n")
        return head, [b.strip("\n") for b in blocks if b.strip()], tail

    head_a, trees_a, tail_a = split(text_a)
    head_b, trees_b, _ = split(text_b)
    lines_a, lines_b = head_a.split("\n"), head_b.split("\n")

    def get(lines, key):
        return next(l for l in lines if l.startswith(key + "="))

    assert get(lines_a, "feature_names") == get(lines_b, "feature_names"), "different feature lists"
    fa = get(lines_a, "feature_infos")[len("feature_infos="):].split(" ")
    fb = get(lines_b, "feature_infos")[len("feature_infos="):].split(" ")
    infos = []
    for a, b in zip(fa, fb):
        if a == "none" or b == "none":
            infos.append(b if a == "none" else a)
        else:
            lo = min(float(a[1:-1].split(":")[0]), float(b[1:-1].split(":")[0]))
            hi = max(float(a[1:-1].split(":")[1]), float(b[1:-1].split(":")[1]))
            infos.append(f"[{lo:g}:{hi:g}]")
    blocks = [f"Tree={i}\n" + blk.split("\n", 1)[1] for i, blk in enumerate(trees_a + trees_b)]
    sizes = [len(b) + 2 for b in blocks]
    head = []
    for l in lines_a:
        if l.startswith("feature_infos="):
            l = "feature_infos=" + " ".join(infos)
        elif l.startswith("tree_sizes="):
            l = "tree_sizes=" + " ".join(map(str, sizes))
        head.append(l)
    return "\n".join(head) + "\n" + "\n\n".join(blocks) + "\n\n\nend of trees\n" + tail_a


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
    ap.add_argument("--no-blend", dest="blend", action="store_false",
                    help="single model only. Default (what ships): also train a raw-features-only member and "
                         "merge both into ONE model file (prediction = main model + raw member)")
    ap.add_argument("--blend-raw-trees", type=int, default=800)
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
    names = feature_names(width, args.transform)
    model = lgb.train(params, lgb.Dataset(T, label=y, group=group_sizes(q), feature_name=names),
                      num_boost_round=args.trees)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.blend:
        if args.transform != "raw+qz":
            raise SystemExit("--blend expects --transform raw+qz")
        # raw member: same layout, query-z block held constant so its trees only use raw features
        T_raw = T.copy()
        T_raw[:, width:] = 0.0
        member = lgb.train(params, lgb.Dataset(T_raw, label=y, group=group_sizes(q), feature_name=names),
                           num_boost_round=args.blend_raw_trees)
        del T_raw
        merged = merge_models(model.model_to_string(), member.model_to_string())
        with open(args.out, "w") as f:  # write the merged text as is (re-saving through lightgbm reformats it)
            f.write(merged)
        model = lgb.Booster(model_str=merged)
    else:
        model.save_model(args.out)
    print(f"trained on {len(set(q.tolist()))} queries ({'train+dev' if args.use_dev else 'train only'}), "
          f"transform={args.transform}, {args.trees} trees"
          f"{f' + raw member {args.blend_raw_trees} trees (merged)' if args.blend else ''} -> {args.out}")
    print(f"train nDCG@10 = {mean_ndcg_from_scores(model.predict(T), y, q):.4f}")
    if dev is not None and not args.use_dev:
        Xd, yd, qd = dev
        print(f"dev   nDCG@10 = {mean_ndcg_from_scores(model.predict(apply_transform(Xd, qd, args.transform)), yd, qd):.4f}")


if __name__ == "__main__":
    main()
