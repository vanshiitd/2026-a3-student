"""experiments/c2_probe.py -- which kind of test-set change could cost ~0.13 nDCG@10 (our clean held-out ~0.49
vs live C2 0.3606)? TRAIN CV fold only; trains the shipped blend's two members on 4/5 of TRAIN and scores the
held-out fold under many perturbation families at several strengths. Nothing here touches DEV or held-out.

    python experiments/c2_probe.py --fold 0

Rows go to experiments/results/c2_probe.jsonl.
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
from harness.feature_names import FIELDS, GROUPS, SINGLES  # noqa: E402
from submission.ltr_common import apply_transform, feature_names, group_sizes  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results", "c2_probe.jsonl")


def cols_field(field):  # 0-based columns of all 25 groups for one field
    return [5 * g + FIELDS.index(field) for g in range(len(GROUPS))]


def cols_group(group):  # one feature type, all 5 fields
    return [5 * GROUPS.index(group) + f for f in range(5)]


def cols_singles(names):
    return [125 + SINGLES.index(n) for n in names]


def perturbations():
    """name -> function(X, rng) returning a perturbed copy (or (X, keep_mask) for pool changes)"""
    P = {}
    for f in FIELDS:
        P[f"drop_field:{f}"] = ("cols", cols_field(f))
    P["drop_singles(126-136)"] = ("cols", list(range(125, 136)))
    P["drop_clicks(133-136)"] = ("cols", cols_singles(["query_url_click_count", "url_click_count", "url_dwell_time"]))
    P["drop_quality+rank(130-132)"] = ("cols", cols_singles(["pagerank", "siterank", "quality_score", "quality_score_2"]))
    for g in ("bm25", "lmir_abs", "lmir_dir", "lmir_jm", "vector_space_model", "sum_tfidf", "covered_query_term_ratio"):
        P[f"drop_group:{g}"] = ("cols", cols_group(g))
    for frac in (0.1, 0.3, 0.5):
        P[f"dropcol_rand:{frac}"] = ("randcols", frac)
    for frac in (0.1, 0.3, 0.5):
        P[f"missing:{frac}"] = ("missing", frac)
    for s in (0.1, 0.3, 1.0):
        P[f"noise_mult:{s}"] = ("noise", s)
    for frac in (0.2, 0.5, 1.0):
        P[f"scale_cols:{frac}"] = ("scale", frac)
    for frac in (0.5, 0.25):
        P[f"subpool:{frac}"] = ("subpool", frac)
    P["clean"] = ("none", None)
    return P


def apply(kind, arg, X, y, q, rng):
    X = X.copy()
    if kind == "cols":
        X[:, arg] = 0
    elif kind == "randcols":
        X[:, rng.choice(X.shape[1], int(X.shape[1] * arg), replace=False)] = 0
    elif kind == "missing":
        X[rng.rand(*X.shape) < arg] = 0
    elif kind == "noise":
        X *= (1 + arg * rng.randn(*X.shape)).astype(np.float32)
    elif kind == "scale":
        cols = rng.choice(X.shape[1], int(X.shape[1] * arg), replace=False)
        X[:, cols] *= np.exp(rng.uniform(np.log(0.1), np.log(10), len(cols))).astype(np.float32)
    elif kind == "subpool":
        keep = rng.rand(len(y)) < arg
        return X[keep], y[keep], q[keep]
    return X, y, q


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    X, y, q = load_split("train")
    tr, va = grouped_kfold(q, k=5, seed=0)[args.fold]
    common = dict(objective="lambdarank", metric="None", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=0,
                  deterministic=True, force_row_wise=True, verbose=-1, num_threads=args.threads)
    T = apply_transform(X[tr], q[tr], "raw+qz")
    names = feature_names(136, "raw+qz")
    m_qz = lgb.train(common, lgb.Dataset(T, label=y[tr], group=group_sizes(q[tr]), feature_name=names), 600)
    T[:, 136:] = 0
    m_raw = lgb.train(common, lgb.Dataset(T, label=y[tr], group=group_sizes(q[tr]), feature_name=names), 800)
    del T
    for name, (kind, arg) in perturbations().items():
        rng = np.random.RandomState(7)
        Xs, ys, qs = apply(kind, arg, X[va], y[va], q[va], rng)
        Ts = apply_transform(Xs, qs, "raw+qz")
        s_qz, s_raw = m_qz.predict(Ts), m_raw.predict(Ts)
        rec = {"fold": args.fold, "pert": name, "qz": ndcg10(s_qz, ys, qs), "rawmember": ndcg10(s_raw, ys, qs),
               "blend": ndcg10(s_qz + s_raw, ys, qs)}
        with open(RESULTS, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{name:<32} qz {rec['qz']:.4f}  raw {rec['rawmember']:.4f}  blend {rec['blend']:.4f}", flush=True)
    imp = m_qz.feature_importance("gain")
    top = np.argsort(-imp)[:15]
    print("top gain features (qz model):", [(names[i], round(float(imp[i] / imp.sum()), 3)) for i in top], flush=True)


if __name__ == "__main__":
    main()
