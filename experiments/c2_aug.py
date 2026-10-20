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


FRAC = 0.3  # corruption fraction for mix='fixed' (overridable with --frac)


def corrupt_query(Xq, rng, mix="fixed"):
    """one corruption per query copy, drawn from the mixture. mix='fixed' = the shipped one (30% / sigma .5);
    mix='varied' draws the strength per copy (fraction U(.1,.6), sigma U(.2,1.0))"""
    Xq = Xq.copy()
    kind = rng.randint(5)
    frac = FRAC if mix == "fixed" else rng.uniform(0.1, 0.6)
    sigma = 0.5 if mix == "fixed" else rng.uniform(0.2, 1.0)
    if kind == 0:
        Xq[:, SINGLES_COLS] = 0
    elif kind == 1:
        Xq[:, cols_field(FIELDS[rng.randint(5)])] = 0
    elif kind == 2:
        Xq[:, rng.choice(Xq.shape[1], int(frac * Xq.shape[1]), replace=False)] = 0
    elif kind == 3:
        Xq[rng.rand(*Xq.shape) < frac] = 0
    else:
        Xq *= (1 + sigma * rng.randn(*Xq.shape)).astype(np.float32)
    return Xq


def augment(X, y, q, copies, seed, mix="fixed"):
    if copies == 0:
        return X, y, q
    rng = np.random.RandomState(seed)
    b = query_bounds(q)
    Xs, ys, qs = [X], [y], [q]
    for c in range(copies):
        blocks = [corrupt_query(X[b[i]:b[i + 1]], rng, mix) for i in range(len(b) - 1)]
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
    ap.add_argument("--mix", choices=["fixed", "varied"], default="fixed")
    ap.add_argument("--ff", type=float, default=0.5)
    ap.add_argument("--only-final-trees", action="store_true", help="evaluate 1200/1600 only")
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="use only this fraction of the training-fold queries (learning-curve check)")
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--extra-trees", action="store_true")
    ap.add_argument("--nosingles-member", action="store_true",
                    help="third member that never sees the 11 non-text features (zeroed in both blocks)")
    ap.add_argument("--third", action="store_true",
                    help="also train a query-z-only member (raw block held constant) and report the 3-member blend")
    args = ap.parse_args()
    global FRAC
    FRAC = args.frac
    X, y, q = load_split("train")
    tr, va = grouped_kfold(q, k=5, seed=0)[args.fold]
    if args.subsample < 1.0:  # keep a random subset of whole training queries
        uq = np.array(list(dict.fromkeys(q[tr].tolist())))
        keep = set(np.random.RandomState(42).choice(uq, int(len(uq) * args.subsample), replace=False).tolist())
        tr = np.array([i for i in tr if q[i] in keep])
    common = dict(objective="lambdarank", metric="None", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=args.ff, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=0,
                  deterministic=True, force_row_wise=True, verbose=-1, num_threads=args.threads,
                  extra_trees=args.extra_trees)
    names = feature_names(136, "raw+qz")
    perts = perturbations()
    for K in args.copies:
        Xa, ya, qa = augment(X[tr], y[tr], q[tr], K, seed=100 + args.fold, mix=args.mix)
        T = apply_transform(Xa, qa, "raw+qz")
        m_qz = lgb.train(common, lgb.Dataset(T, label=ya, group=group_sizes(qa), feature_name=names), 1200)
        m_qzo = None
        if args.third:  # query-z-only member: raw block constant
            Tq = T.copy()
            Tq[:, :136] = 0
            m_qzo = lgb.train(common, lgb.Dataset(Tq, label=ya, group=group_sizes(qa), feature_name=names), 1200)
            del Tq
        elif args.nosingles_member:  # member that never uses the 11 non-text features
            Tq = T.copy()
            Tq[:, SINGLES_COLS] = 0
            Tq[:, [136 + c for c in SINGLES_COLS]] = 0
            m_qzo = lgb.train(common, lgb.Dataset(Tq, label=ya, group=group_sizes(qa), feature_name=names), 1200)
            del Tq
        T[:, 136:] = 0
        m_raw = lgb.train(common, lgb.Dataset(T, label=ya, group=group_sizes(qa), feature_name=names), 1600)
        del T, Xa
        for nq, nr in (((1200, 1600),) if args.only_final_trees else ((600, 800), (900, 1200), (1200, 1600))):
            rec = {"fold": args.fold, "copies": K, "trees": [nq, nr], "mix": args.mix, "ff": args.ff,
                   "subsample": args.subsample, "frac": args.frac, "extra_trees": args.extra_trees,
                   "third": "qzonly" if args.third else ("nosingles" if args.nosingles_member else None)}
            for name, (kind, arg) in perts.items():
                rng = np.random.RandomState(7)
                Xs, ys, qs = apply(kind, arg, X[va], y[va], q[va], rng)
                Ts = apply_transform(Xs, qs, "raw+qz")
                s2 = m_qz.predict(Ts, num_iteration=nq) + m_raw.predict(Ts, num_iteration=nr)
                rec[name] = ndcg10(s2, ys, qs)
                if m_qzo is not None:
                    rec["3m:" + name] = ndcg10(s2 + m_qzo.predict(Ts, num_iteration=nq), ys, qs)
            rec["hard_mean"] = float(np.mean([rec[h] for h in HARD]))
            if m_qzo is not None:
                rec["3m:hard_mean"] = float(np.mean([rec["3m:" + h] for h in HARD]))
                print(f"  3-member blend: clean {rec['3m:clean']:.4f}  hard_mean {rec['3m:hard_mean']:.4f}", flush=True)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"frac={args.frac} et={args.extra_trees} sub={args.subsample} mix={args.mix} ff={args.ff} copies={K} trees={nq}/{nr}  clean {rec['clean']:.4f}  hard_mean {rec['hard_mean']:.4f}  "
                  f"drop_singles {rec['drop_singles(126-136)']:.4f}  missing.3 {rec['missing:0.3']:.4f}  "
                  f"noise1 {rec['noise_mult:1.0']:.4f}  dropcol.5 {rec['dropcol_rand:0.5']:.4f}", flush=True)


if __name__ == "__main__":
    main()
