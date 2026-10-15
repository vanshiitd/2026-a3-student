"""experiments/fusion_eval.py -- score fusion variants with the real submission/fusion.py functions and the
harness metric code (nDCG@10, MAP@10). Tuning reads TRAIN only; DEV is printed for the record.

    python experiments/fusion_eval.py baselines
    python experiments/fusion_eval.py tune          # grid on TRAIN, then one DEV read of the pick
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_split, query_bounds  # noqa: E402

sys.path.insert(0, ROOT)
from harness.feature_names import BASE_RANKER_CHANNELS  # noqa: E402
from harness.metrics import evaluate_run  # noqa: E402
from submission import fusion  # noqa: E402

CHANNELS = list(BASE_RANKER_CHANNELS)
RESULTS = os.path.join(ROOT, "experiments", "results", "fusion.jsonl")


def pools(split, max_queries=None):
    """[(qid, base_scores dict, qrels dict)] exactly as the harness builds them"""
    X, y, q = load_split(split)
    b = query_bounds(q)
    out = []
    for i in range(len(b) - 1 if max_queries is None else min(max_queries, len(b) - 1)):
        a, e = b[i], b[i + 1]
        qid = str(q[a])
        ids = [f"{qid}:{r}" for r in range(e - a)]
        base = {d: {name: float(X[a + r, idx - 1]) for name, idx in BASE_RANKER_CHANNELS.items()}
                for r, d in enumerate(ids)}
        out.append((qid, base, {d: int(y[a + r]) for r, d in enumerate(ids)}))
    return out


def score(data, fn):
    run, qrels = {}, {}
    for qid, base, qr in data:
        fused = fn(base)
        ranked = sorted(base.keys(), key=lambda d: -fused[d])[:10]  # stable, like ranking.py
        run[qid] = [(d, fused[d]) for d in ranked]
        qrels[qid] = qr
    agg = evaluate_run(run, qrels)["aggregate"]
    return agg["ndcg@10"], agg["map@10"]


def baselines():
    for split in ("train", "dev"):
        data = pools(split)
        print(f"== {split} ({len(data)} queries)")
        for ch in CHANNELS:
            print(f"  single {ch:<26} nDCG@10 %.4f  MAP@10 %.4f" % score(data, lambda b, ch=ch: {d: s[ch] for d, s in b.items()}))
        print("  RRF c=60                          nDCG@10 %.4f  MAP@10 %.4f" % score(data, fusion.rrf_scores))
        print("  CombSUM                           nDCG@10 %.4f  MAP@10 %.4f" % score(data, fusion.combsum_scores))
        print("  CombMNZ                           nDCG@10 %.4f  MAP@10 %.4f" % score(data, fusion.combmnz_scores))


def candidates():
    """method x c x channel weights (weights in {0, .5, 1, 2}, at least one non-zero)"""
    ws = [0.0, 0.5, 1.0, 2.0]
    for w in itertools.product(ws, repeat=len(CHANNELS)):
        if max(w) == 0 or (max(w) > 1 and min(x for x in w if x > 0) == max(w)):
            continue  # all-zero, or a pure rescale of an all-equal weighting
        weights = dict(zip(CHANNELS, w))
        for c in (5, 20, 60, 120):
            yield {"method": "rrf", "c": c, "weights": weights}
        yield {"method": "combsum", "weights": weights}
        yield {"method": "combmnz", "weights": weights}


def make_fn(cfg):
    if cfg["method"] == "rrf":
        return lambda b: fusion.rrf_scores(b, c=cfg["c"], weights=cfg["weights"])
    if cfg["method"] == "combsum":
        return lambda b: fusion.combsum_scores(b, weights=cfg["weights"])
    return lambda b: fusion.combmnz_scores(b, weights=cfg["weights"])


def track_a(nd, mp):
    return 0.8 * nd + 0.2 * mp  # within-track weighting (percentiles in reality)


def tune():
    train = pools("train")
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    rows = []
    cands = list(candidates())
    print(f"{len(cands)} candidates on TRAIN ({len(train)} queries)", flush=True)
    for i, cfg in enumerate(cands):
        nd, mp = score(train, make_fn(cfg))
        rows.append(dict(cfg, train_ndcg=nd, train_map=mp))
        if i % 100 == 0:
            print(f"  {i}/{len(cands)}", flush=True)
    rows.sort(key=lambda r: -track_a(r["train_ndcg"], r["train_map"]))
    with open(RESULTS, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows[:15]:
        print(f"  {r['method']:<8} c={r.get('c', '-'):<4} w={list(r['weights'].values())}  "
              f"train nDCG {r['train_ndcg']:.4f} MAP {r['train_map']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["baselines", "tune"])
    args = ap.parse_args()
    baselines() if args.cmd == "baselines" else tune()


if __name__ == "__main__":
    main()
