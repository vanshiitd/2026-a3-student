"""experiments/c2_aug.py -- feature-dropout augmentation for robustness (Track C2), TRAIN CV only.

Training data = the original queries + K augmented copies of every query; each copy gets ONE corruption drawn
from a fixed mixture (whole 11-feature non-text block zeroed / one whole field zeroed / 30% of columns zeroed /
30% of values zeroed / multiplicative noise sigma 0.5). Copies are separate query groups. The query-z block is
computed AFTER corrupting (that's what rerank_with_ltr sees at test time). Evaluated on the c2_probe suite.

    python experiments/c2_aug.py --fold 0 --copies 0 1 2
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c2_probe import apply, cols_field, perturbations  # noqa: E402
from common import ROOT, load_split, ndcg10, query_bounds  # noqa: E402

sys.path.insert(0, ROOT)
from harness.cv import grouped_kfold  # noqa: E402
from harness.feature_names import FIELDS  # noqa: E402
from submission.ltr_common import apply_transform, feature_names, group_sizes  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results", "c2_aug.jsonl")
SINGLES_COLS = list(range(125, 136))
HARD = ["drop_singles(126-136)", "drop_clicks(133-136)", "drop_quality+rank(130-132)"] + \
       [f"drop_field:{f}" for f in FIELDS] + ["dropcol_rand:0.3", "dropcol_rand:0.5", "missing:0.3", "missing:0.5",
                                             "noise_mult:0.3", "noise_mult:1.0", "scale_cols:0.5", "scale_cols:1.0"]


def corrupt_query(Xq, rng):
    """one corruption per query copy, drawn from the mixture"""
    Xq = Xq.copy()
    kind = rng.randint(5)
    if kind == 0:
        Xq[:, SINGLES_COLS] = 0
    elif kind == 1:
        Xq[:, cols_field(FIELDS[rng.randint(5)])] = 0
    elif kind == 2:
        Xq[:, rng.choice(Xq.shape[1], int(0.3 * Xq.shape[1]), replace=False)] = 0
    elif kind == 3:
        Xq[rng.rand(*Xq.shape) < 0.3] = 0
    else:
        Xq *= (1 + 0.5 * rng.randn(*Xq.shape)).astype(np.float32)
    return Xq


def augment(X, y, q, copies, seed):
    if copies == 0:
        return X, y, q
    rng = np.random.RandomState(seed)
    b = query_bounds(q)
    Xs, ys, qs = [X], [y], [q]
    for c in range(copies):
        blocks = [corrupt_query(X[b[i]:b[i + 1]], rng) for i in range(len(b) - 1)]
        Xs.append(np.vstack(blocks))
        ys.append(y)
        qs.append(np.array([f"{qq}#aug{c}" for qq in q], dtype=object))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(qs)


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--copies", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    X, y, q = load_split("train")
    tr, va = grouped_kfold(q, k=5, seed=0)[args.fold]
    common = dict(objective="lambdarank", metric="None", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=0,
                  deterministic=True, force_row_wise=True, verbose=-1, num_threads=args.threads)
    names = feature_names(136, "raw+qz")
    perts = perturbations()
    for K in args.copies:
        Xa, ya, qa = augment(X[tr], y[tr], q[tr], K, seed=100 + args.fold)
        T = apply_transform(Xa, qa, "raw+qz")
        m_qz = lgb.train(common, lgb.Dataset(T, label=ya, group=group_sizes(qa), feature_name=names), 1200)
        T[:, 136:] = 0
        m_raw = lgb.train(common, lgb.Dataset(T, label=ya, group=group_sizes(qa), feature_name=names), 1600)
        del T, Xa
        for nq, nr in ((600, 800), (900, 1200), (1200, 1600)):
            rec = {"fold": args.fold, "copies": K, "trees": [nq, nr]}
            for name, (kind, arg) in perts.items():
                rng = np.random.RandomState(7)
                Xs, ys, qs = apply(kind, arg, X[va], y[va], q[va], rng)
                Ts = apply_transform(Xs, qs, "raw+qz")
                rec[name] = ndcg10(m_qz.predict(Ts, num_iteration=nq) + m_raw.predict(Ts, num_iteration=nr), ys, qs)
            rec["hard_mean"] = float(np.mean([rec[h] for h in HARD]))
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"copies={K} trees={nq}/{nr}  clean {rec['clean']:.4f}  hard_mean {rec['hard_mean']:.4f}  "
                  f"drop_singles {rec['drop_singles(126-136)']:.4f}  missing.3 {rec['missing:0.3']:.4f}  "
                  f"noise1 {rec['noise_mult:1.0']:.4f}  dropcol.5 {rec['dropcol_rand:0.5']:.4f}", flush=True)


if __name__ == "__main__":
    main()
