"""experiments/common.py -- shared helpers for the offline experiments (not used at grading time)

MSLR text files take a while to parse, so load_split() caches them as .npz under data/cache/ (gitignored).
Only Fold1 train (TRAIN) and vali (DEV) are ever read; the held-out test split is not on this machine.
"""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from submission.ltr_common import read_ltr  # noqa: E402

PATHS = {"train": os.path.join(ROOT, "data", "train", "train.txt"),
         "dev": os.path.join(ROOT, "data", "dev", "vali.txt")}
CACHE = os.path.join(ROOT, "data", "cache")


def load_split(name: str):
    """(X float32, y int64, qids str array) for 'train' or 'dev'; queries contiguous, file order"""
    path = os.path.join(CACHE, f"{name}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return z["X"], z["y"], z["q"].astype(object)
    X, y, q = read_ltr(PATHS[name], n_features=136)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, X=X, y=y, q=q.astype(str))
    return X, y, q


def query_bounds(qids):
    """start offsets of each query block (+ final end) for contiguous qids"""
    qids = np.asarray(qids)
    starts = np.r_[0, np.nonzero(qids[1:] != qids[:-1])[0] + 1]
    return np.r_[starts, len(qids)]


def ndcg10_per_query(scores, labels, bounds, k=10):
    """nDCG@k per query, same convention as harness.metrics (gain 2^l-1, log2 discount,
    stable ties, all-zero query = 0). bounds from query_bounds()."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    out = np.zeros(len(bounds) - 1)
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        lab = labels[a:b]
        order = np.argsort(-scores[a:b], kind="stable")[:k]
        dcg = ((2.0 ** lab[order] - 1) * disc[:len(order)]).sum()
        ideal = np.sort(lab)[::-1][:k]
        idcg = ((2.0 ** ideal - 1) * disc[:len(ideal)]).sum()
        out[i] = dcg / idcg if idcg > 0 else 0.0
    return out


def ndcg10(scores, labels, qids):
    return float(ndcg10_per_query(scores, labels, query_bounds(qids)).mean())
