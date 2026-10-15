"""
submission/ltr_common.py -- shared bits for recipe.py, scripts/train_model.py and ranking.py

  * read_ltr()        fast svmlight/letor reader (numpy per line, no per-token python loop)
  * group_sizes()     lightgbm wants rows per query, queries contiguous
  * feature transform the model was trained with is written into the model's feature names,
                      so rerank_with_ltr can rebuild the same inputs for any dimension
"""
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- reading

def read_ltr(path: str, n_features: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """returns (X float32 [rows, D], y int64, qids object array), file order kept.
    sparse lines ok (missing features = 0), trailing '# comment' ignored.
    D = n_features if given else the largest feature index seen."""
    labels, qids, rows = [], [], []
    max_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 2 or not parts[1].startswith("qid:"):
                raise ValueError(f"{path}:{line_no}: expected '<label> qid:<id> ...'")
            labels.append(int(float(parts[0])))
            qids.append(parts[1][4:])
            if len(parts) == 3:
                # "i:v i:v ..." -> "i v i v ..." then one C-level parse
                arr = np.array(parts[2].replace(":", " ").split(), dtype=np.float64)
                idx = arr[0::2].astype(np.int64)
                if len(idx) and idx.max() > max_idx:
                    max_idx = int(idx.max())
                rows.append((idx, arr[1::2]))
            else:
                rows.append((np.zeros(0, np.int64), np.zeros(0)))
    width = max_idx if n_features is None else int(n_features)
    if max_idx > width:
        raise ValueError(f"{path}: feature index {max_idx} > n_features={width}")
    X = np.zeros((len(rows), width), dtype=np.float32)
    for r, (idx, val) in enumerate(rows):
        if len(idx):
            X[r, idx - 1] = val
    return X, np.asarray(labels, dtype=np.int64), np.asarray(qids, dtype=object)


def pad_to(X: np.ndarray, width: int) -> np.ndarray:
    """zero-pad (or cut) columns so a file whose last features were all-zero still matches the model"""
    if X.shape[1] == width:
        return X
    if X.shape[1] > width:
        return X[:, :width]
    out = np.zeros((X.shape[0], width), dtype=X.dtype)
    out[:, :X.shape[1]] = X
    return out


def group_sizes(qids) -> List[int]:
    """rows per query in order of appearance; rows of a query must be contiguous"""
    sizes, last = [], object()
    for q in qids:
        if q == last:
            sizes[-1] += 1
        else:
            sizes.append(1)
            last = q
    return sizes


def make_contiguous(X, y, qids):
    """stable reorder so each query's rows sit together (lightgbm needs that)"""
    first = {}
    for i, q in enumerate(qids):
        first.setdefault(q, i)
    order = np.argsort(np.array([first[q] for q in qids]), kind="stable")
    return X[order], y[order], qids[order]


# ---------------------------------------------------------------- feature transform
# model feature names encode the transform:
#   "f<i>"   raw feature i
#   "qz<i>"  feature i standardised within the query's candidate pool
# so prepare() only needs the model file.

def feature_names(n_raw: int, transform: str) -> List[str]:
    names = [f"f{i}" for i in range(n_raw)]
    if transform == "raw+qz":
        names += [f"qz{i}" for i in range(n_raw)]
    elif transform != "raw":
        raise ValueError(f"unknown transform {transform!r}")
    return names


def parse_names(names: List[str]) -> Tuple[int, str]:
    """(n_raw, transform) from a model's feature names; models without our names count as raw"""
    n_raw = sum(1 for n in names if n.startswith("f") and n[1:].isdigit())
    if n_raw == 0:
        return len(names), "raw"
    has_qz = any(n.startswith("qz") for n in names)
    return n_raw, ("raw+qz" if has_qz else "raw")


def query_zscore(Xq: np.ndarray) -> np.ndarray:
    """z-score each column inside one query's pool (constant column -> 0)"""
    mu = Xq.mean(axis=0)
    sd = Xq.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return ((Xq - mu) / sd).astype(np.float32)


def apply_transform(X: np.ndarray, qids, transform: str) -> np.ndarray:
    """X rows grouped by query (contiguous) -> model inputs"""
    if transform == "raw":
        return X
    out = np.empty((X.shape[0], 2 * X.shape[1]), dtype=np.float32)
    out[:, :X.shape[1]] = X
    start = 0
    for size in group_sizes(qids):
        out[start:start + size, X.shape[1]:] = query_zscore(X[start:start + size])
        start += size
    return out
