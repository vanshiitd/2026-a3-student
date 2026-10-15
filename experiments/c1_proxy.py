"""experiments/c1_proxy.py -- stand-in benchmarks for Track C1, built ONLY from MSLR TRAIN queries.

Other public LTR datasets are off limits (they could be the hidden benchmarks; spec Section 3), so the proxies
are reshaped slices of MSLR TRAIN: different query counts, random feature subsets, random monotone rescaling,
pure-noise columns, shuffled column order, 0-4 or collapsed 0-2 labels, sparse files. They are a sanity
check on how the recipe behaves across sizes/shapes, not a claim about the real benchmarks.

    python experiments/c1_proxy.py build          # writes data/cache/proxies/<name>/{train,vali,test}.txt
    python experiments/c1_proxy.py eval --variant base --variant qz ...

Results append to experiments/results/c1_proxy.jsonl.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_split, ndcg10  # noqa: E402

sys.path.insert(0, ROOT)
from submission.ltr_common import apply_transform, pad_to, parse_names, read_ltr  # noqa: E402

PROXY_DIR = os.path.join(ROOT, "data", "cache", "proxies")
RESULTS = os.path.join(ROOT, "experiments", "results", "c1_proxy.jsonl")

# name -> (n_queries, n_real_features, n_noise_features, label_scheme, seed)
PROXIES = {
    "p1_small_46f_l3": (350, 46, 0, "0-2", 11),
    "p2_small_60f_l5": (600, 60, 10, "0-4", 12),
    "p3_mid_90f_l5": (1500, 90, 20, "0-4", 13),
    "p4_mid_136f_l3": (2000, 136, 0, "0-2", 14),
    "p5_big_220f_l5": (4000, 136, 84, "0-4", 15),
}

VARIANTS = {
    "base": {},
    "qz": {"transform": "raw+qz"},
    "single": {"search": "single"},
    "refit": {"refit": "trainval"},
    "qz_refit": {"transform": "raw+qz", "refit": "trainval"},
    "lr10": {"learning_rate": 0.1},
}


def _monotone(col, rng):
    kind = rng.randint(4)
    if kind == 0:
        return col
    if kind == 1:
        return np.sign(col) * np.log1p(np.abs(col))
    if kind == 2:
        return col * rng.uniform(0.01, 100) + rng.uniform(-50, 50)
    return np.sign(col) * np.sqrt(np.abs(col))


def _write(path, X, y, q):
    with open(path, "w") as f:
        for i in range(len(y)):
            nz = np.nonzero(X[i])[0]
            body = " ".join(f"{j + 1}:{X[i, j]:.6g}" for j in nz)
            f.write(f"{y[i]} qid:{q[i]} {body}\n")


TEST_QUERIES = 1500  # big disjoint test set so proxy noise (SE ~0.006) is below the effects we compare


def build():
    """train/vali = the proxy's own size (that's what the recipe's size rules see); test = 1500 OTHER
    MSLR TRAIN queries pushed through the same column pick / rescale / noise / label map"""
    X, y, q = load_split("train")
    uq = np.array(list(dict.fromkeys(q.tolist())))
    for name, (n_q, n_f, n_noise, labels, seed) in PROXIES.items():
        out = os.path.join(PROXY_DIR, name)
        if os.path.exists(os.path.join(out, "test.txt")):
            continue
        rng = np.random.RandomState(seed)
        picked = rng.choice(uq, n_q + TEST_QUERIES, replace=False)
        fit_q, test_q = picked[:n_q], picked[n_q:]
        n_train = int(round(0.75 * n_q))
        split_of = {qq: "train" for qq in fit_q[:n_train]}
        split_of.update({qq: "vali" for qq in fit_q[n_train:]})
        split_of.update({qq: "test" for qq in test_q})
        mask = np.isin(q, picked)
        Xp, yp, qp = X[mask], y[mask].copy(), q[mask]
        cols = np.sort(rng.choice(136, n_f, replace=False))
        Xp = np.column_stack([_monotone(Xp[:, c].astype(np.float64), rng) for c in cols] +
                             [rng.randn(len(yp)) for _ in range(n_noise)])
        Xp = Xp[:, rng.permutation(Xp.shape[1])].astype(np.float32)
        if labels == "0-2":
            yp = np.array([0, 1, 1, 2, 2])[yp]
        qmap = {qq: str(70000 + i) for i, qq in enumerate(picked)}  # opaque new qids
        os.makedirs(out, exist_ok=True)
        for split in ("train", "vali", "test"):
            m = np.array([split_of[qq] == split for qq in qp])
            _write(os.path.join(out, f"{split}.txt"), Xp[m], yp[m], np.array([qmap[qq] for qq in qp[m]]))
        print(f"built {name}: {n_q} fit queries (+{TEST_QUERIES} test), {Xp.shape[1]} features, "
              f"labels {labels}", flush=True)


def _score(model_path, test_path):
    import lightgbm as lgb
    model = lgb.Booster(model_file=model_path)
    n_raw, transform = parse_names(model.feature_name())
    Xt, yt, qt = read_ltr(test_path)
    return ndcg10(model.predict(apply_transform(pad_to(Xt, n_raw), qt, transform)), yt, qt)


def evaluate(variants, proxies, seed):
    import submission.recipe as recipe
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    for name in proxies:
        d = os.path.join(PROXY_DIR, name)
        for v in variants:
            opts = dict(recipe.OPTIONS, **VARIANTS[v])
            model_path = os.path.join(d, f"model_{v}.txt")
            t0 = time.time()
            info = recipe._train(os.path.join(d, "train.txt"), os.path.join(d, "vali.txt"), model_path, seed,
                                 return_info=True, **opts)
            secs = time.time() - t0
            test = _score(model_path, os.path.join(d, "test.txt"))
            rec = {"proxy": name, "variant": v, "seed": seed, "test_ndcg10": test, "seconds": secs,
                   "picked": info["picked"], "best_iter": info["best_iter"], "rounds_cap": info["rounds_cap"],
                   "tried": info["tried"]}
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"{name:<18} {v:<10} test nDCG@10 {test:.4f}  {secs:5.0f}s  picked {info['picked']} "
                  f"iter {info['best_iter']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--variant", action="append", default=None)
    ap.add_argument("--proxy", action="append", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    else:
        evaluate(args.variant or ["base"], args.proxy or list(PROXIES), args.seed)


if __name__ == "__main__":
    main()
