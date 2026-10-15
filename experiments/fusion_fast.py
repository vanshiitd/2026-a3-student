"""experiments/fusion_fast.py -- vectorised copy of submission/fusion.py + harness nDCG@10 / MAP@10, for grid
search only. `check` asserts it matches the real functions and the harness metric before any number is used.

    python experiments/fusion_fast.py check
    python experiments/fusion_fast.py tune      # TRAIN only; prints top configs, then ONE dev read of the pick
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_split, query_bounds  # noqa: E402
from fusion_eval import pools, score, make_fn  # noqa: E402

sys.path.insert(0, ROOT)
from harness.feature_names import BASE_RANKER_CHANNELS  # noqa: E402

CHANNELS = list(BASE_RANKER_CHANNELS)
RESULTS = os.path.join(ROOT, "experiments", "results", "fusion_grid.jsonl")


class Prepared:
    """per-row channel ranks (1-based, stable) and min-max scores within each query"""

    def __init__(self, split, max_queries=None):
        X, y, q = load_split(split)
        b = query_bounds(q)
        if max_queries is not None:
            b = b[:max_queries + 1]
            X, y, q = X[:b[-1]], y[:b[-1]], q[:b[-1]]
        self.y, self.bounds = y, b
        n = len(y)
        self.code = np.repeat(np.arange(len(b) - 1), np.diff(b))
        self.pos = np.arange(n) - b[self.code]            # row position within query (candidate order)
        self.rank = np.zeros((n, len(CHANNELS)))
        self.norm = np.zeros((n, len(CHANNELS)))
        for j, name in enumerate(CHANNELS):
            s = X[:, BASE_RANKER_CHANNELS[name] - 1].astype(np.float64)
            order = np.lexsort((self.pos, -s, self.code))  # best first, ties by candidate order
            r = np.empty(n)
            r[order] = self.pos[order] * 0 + _pos_in_group(self.code[order]) + 1
            self.rank[:, j] = r
            lo = np.minimum.reduceat(s, b[:-1])[self.code]
            hi = np.maximum.reduceat(s, b[:-1])[self.code]
            span = hi - lo
            self.norm[:, j] = np.where(span > 0, (s - lo) / np.where(span > 0, span, 1), 0.0)
        # per-query ideal DCG@10 and relevant counts for the metric
        gain = 2.0 ** y - 1
        order = np.lexsort((-gain, self.code))
        p = _pos_in_group(self.code[order])
        disc = np.where(p < 10, 1.0 / np.log2(p + 2.0), 0.0)
        self.idcg = np.bincount(self.code[order], weights=gain[order] * disc, minlength=len(b) - 1)
        self.n_rel = np.bincount(self.code, weights=(y > 0).astype(float), minlength=len(b) - 1)

    def fused(self, cfg):
        w = np.array([cfg["weights"][c] for c in CHANNELS])
        if cfg["method"] == "rrf":
            return ((w > 0) * w / (cfg["c"] + self.rank)).sum(axis=1)
        s = (self.norm * w).sum(axis=1)
        if cfg["method"] == "combsum":
            return s
        hits = ((self.norm > 0) & (w > 0)).sum(axis=1)
        return s * hits

    def metrics(self, f):
        order = np.lexsort((self.pos, -f, self.code))     # stable sort by fused score inside each query
        p = _pos_in_group(self.code[order])
        top = p < 10
        rows, pos = order[top], p[top]
        codes = self.code[rows]
        gain = 2.0 ** self.y[rows] - 1
        dcg = np.bincount(codes, weights=gain / np.log2(pos + 2.0), minlength=len(self.idcg))
        nd = np.where(self.idcg > 0, dcg / np.where(self.idcg > 0, self.idcg, 1), 0.0)
        rel = (self.y[rows] > 0).astype(float)
        hits = _cumsum_in_group(rel, codes)
        ap = np.bincount(codes, weights=rel * hits / (pos + 1.0), minlength=len(self.idcg))
        mp = np.where(self.n_rel > 0, ap / np.where(self.n_rel > 0, self.n_rel, 1), 0.0)
        return float(nd.mean()), float(mp.mean())


def _pos_in_group(sorted_codes):
    n = len(sorted_codes)
    starts = np.r_[True, sorted_codes[1:] != sorted_codes[:-1]]
    first = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    return np.arange(n) - first


def _cumsum_in_group(v, sorted_codes):
    c = np.cumsum(v)
    starts = np.r_[True, sorted_codes[1:] != sorted_codes[:-1]]
    base = np.maximum.accumulate(np.where(starts, np.arange(len(v)), 0))
    return c - (c[base] - v[base])


def candidates():
    ws = [0.0, 0.5, 1.0, 2.0]
    for w in itertools.product(ws, repeat=len(CHANNELS)):
        if max(w) == 0:
            continue
        nz = [x for x in w if x > 0]
        if len(set(nz)) == 1 and nz[0] != 1.0:
            continue  # pure rescale of an equal-weight subset
        weights = dict(zip(CHANNELS, w))
        for c in (1, 5, 10, 20, 40, 60, 120):
            yield {"method": "rrf", "c": c, "weights": weights}
        yield {"method": "combsum", "weights": weights}
        yield {"method": "combmnz", "weights": weights}


def check():
    prep = Prepared("dev", max_queries=300)
    data = pools("dev", max_queries=300)
    rng = np.random.RandomState(0)
    cands = list(candidates())
    for i in rng.choice(len(cands), 25, replace=False):
        cfg = cands[i]
        fast = prep.metrics(prep.fused(cfg))
        slow = score(data, make_fn(cfg))
        assert np.allclose(fast, slow, atol=1e-12), (cfg, fast, slow)
    print("vectorised fusion + metrics match fusion.py + harness on 25 random configs x 300 queries")


def tune():
    prep = Prepared("train")
    rows = []
    for cfg in candidates():
        nd, mp = prep.metrics(prep.fused(cfg))
        rows.append(dict(cfg, train_ndcg=nd, train_map=mp))
    rows.sort(key=lambda r: -(0.8 * r["train_ndcg"] + 0.2 * r["train_map"]))
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} configs on TRAIN; top 12:")
    for r in rows[:12]:
        print(f"  {r['method']:<8} c={str(r.get('c', '-')):<4} w={[r['weights'][c] for c in CHANNELS]}  "
              f"nDCG {r['train_ndcg']:.4f}  MAP {r['train_map']:.4f}")
    for m in ("rrf", "combsum", "combmnz"):
        best = next(r for r in rows if r["method"] == m)
        eq = next(r for r in rows if r["method"] == m and set(r["weights"].values()) == {1.0}
                  and r.get("c", 60) == 60)
        print(f"  best {m:<8} {best['train_ndcg']:.4f}/{best['train_map']:.4f}   "
              f"equal-weight default {eq['train_ndcg']:.4f}/{eq['train_map']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "tune"])
    args = ap.parse_args()
    check() if args.cmd == "check" else tune()


if __name__ == "__main__":
    main()
