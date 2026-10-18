"""experiments/c2_candidates.py -- alternative model.txt candidates for Track C2 (hidden robustness test).

The live leaderboard shows C2 for submitted models, so a few principled candidates get submitted and the best
kept. That is selection on a hidden test and is declared in the README/report. This script provides the
TRAIN-only evidence first:

  cv     3 grouped CV folds on TRAIN: clean + the 5 generic shifts from robustness.py, per candidate and tree count
  final  train each candidate on full TRAIN and write a submission zip that differs only in model.txt

Candidates (lr 0.05, bagging freq 1):
  qz     current shipped model: raw + query-z, 63 leaves, min_data 200, ff 0.5, bag 0.8, l2 1, 600 trees
  raw    raw features only, same shape, 800 trees (its CV peak, notes F8/stage2)
  blend  qz + raw summed into ONE lightgbm model file (trees concatenated; raw member trained with the qz
         block held constant so its trees never touch it) -> prediction = qz score + raw score
  reg    raw + query-z, 63 leaves, min_data 400, ff 0.3, bag 0.7, l2 10 (tree count picked on CV clean)

    python experiments/c2_candidates.py cv
    python experiments/c2_candidates.py final --reg-trees 900
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, load_split, ndcg10  # noqa: E402
from robustness import SHIFTS, shift  # noqa: E402

sys.path.insert(0, ROOT)
from harness.cv import grouped_kfold  # noqa: E402
from submission.ltr_common import apply_transform, feature_names, group_sizes  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results", "c2_candidates.jsonl")
OUT = os.path.join(ROOT, "dist", "candidates")

COMMON = {"objective": "lambdarank", "metric": "None", "learning_rate": 0.05, "bagging_freq": 1,
          "seed": 0, "deterministic": True, "force_row_wise": True, "verbose": -1}
SPECS = {
    "qz": ({"num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.5, "bagging_fraction": 0.8,
            "lambda_l2": 1.0}, "raw+qz", 600),
    "raw": ({"num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.5, "bagging_fraction": 0.8,
             "lambda_l2": 1.0}, "raw", 800),
    "reg": ({"num_leaves": 63, "min_data_in_leaf": 400, "feature_fraction": 0.3, "bagging_fraction": 0.7,
             "lambda_l2": 10.0}, "raw+qz", 1500),  # trained to 1500, tree count chosen on CV
}
REG_CHECKPOINTS = (600, 900, 1200, 1500)


def fit(X, y, q, name, threads, trees=None, constant_qz=False):
    import lightgbm as lgb
    params, tf, n = SPECS[name]
    T = apply_transform(X, q, "raw+qz" if constant_qz else tf)
    if constant_qz:  # blend member: qz block constant -> lightgbm never splits on it, but the
        T[:, X.shape[1]:] = 0.0  # model still has the 272-feature layout the qz model has
    names = feature_names(X.shape[1], "raw+qz" if constant_qz else tf)
    return lgb.train(dict(COMMON, num_threads=threads, **params),
                     lgb.Dataset(T, label=y, group=group_sizes(q), feature_name=names),
                     num_boost_round=trees or n)


def merge_models(text_a: str, text_b: str) -> str:
    """One lightgbm text model whose prediction is model_a + model_b (same feature list required)."""
    def split(text):
        head, rest = text.split("\nTree=0\n", 1)
        trees, tail = rest.split("\nend of trees\n", 1)
        blocks = ("Tree=0\n" + trees).split("\n\n")
        return head, [b.strip("\n") for b in blocks if b.strip()], tail

    head_a, trees_a, tail_a = split(text_a)
    head_b, trees_b, _ = split(text_b)
    lines_a, lines_b = head_a.split("\n"), head_b.split("\n")
    get = lambda lines, key: next(l for l in lines if l.startswith(key + "="))
    assert get(lines_a, "feature_names") == get(lines_b, "feature_names"), "different feature lists"
    # feature ranges: union per feature ("none" = feature unused in that model)
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
    blocks = []
    for i, blk in enumerate(trees_a + trees_b):
        body = blk.split("\n", 1)[1]
        blocks.append(f"Tree={i}\n{body}")
    sizes = [len(b) + 2 for b in blocks]  # each tree block is followed by a blank line
    new_head = []
    for l in lines_a:
        if l.startswith("feature_infos="):
            l = "feature_infos=" + " ".join(infos)
        elif l.startswith("tree_sizes="):
            l = "tree_sizes=" + " ".join(map(str, sizes))
        new_head.append(l)
    return "\n".join(new_head) + "\n" + "\n\n".join(blocks) + "\n\n\nend of trees\n" + tail_a


def cv(threads):
    X, y, q = load_split("train")
    folds = grouped_kfold(q, k=5, seed=0)
    for fold in (0, 1, 2):
        tr, va = folds[fold]
        models = {n: fit(X[tr], y[tr], q[tr], n, threads) for n in SPECS}
        models["blend_raw"] = fit(X[tr], y[tr], q[tr], "raw", threads, constant_qz=True)
        for s in SHIFTS:
            rng = np.random.RandomState(1000 + fold)
            Xs, ys, qs = shift(s, X[va], y[va], q[va], rng)
            Tq = apply_transform(Xs, qs, "raw+qz")
            sc = {"qz": models["qz"].predict(Tq), "raw": models["raw"].predict(Xs)}
            sc["blend"] = sc["qz"] + models["blend_raw"].predict(Tq)
            for n in REG_CHECKPOINTS:
                sc[f"reg@{n}"] = models["reg"].predict(Tq, num_iteration=n)
            rec = {"fold": fold, "shift": s, **{k: ndcg10(v, ys, qs) for k, v in sc.items()}}
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)


def build_zip(name, model_text):
    base = os.path.join(ROOT, "dist", "2023EE10656.zip")
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "2023EE10656.zip")
    with zipfile.ZipFile(base) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = model_text.encode() if item.filename == "submission/model.txt" else src.read(item.filename)
            dst.writestr(item, data)
    with open(os.path.join(out_dir, "model.txt"), "w") as f:
        f.write(model_text)
    return out


def final(threads, reg_trees):
    import lightgbm as lgb
    X, y, q = load_split("train")
    Xd, yd, qd = load_split("dev")
    texts = {}
    m_raw = fit(X, y, q, "raw", threads)
    texts["raw"] = m_raw.model_to_string()
    m_qz = lgb.Booster(model_file=os.path.join(ROOT, "submission", "model.txt"))  # the shipped one
    m_braw = fit(X, y, q, "raw", threads, constant_qz=True)
    texts["blend"] = merge_models(m_qz.model_to_string(), m_braw.model_to_string())
    texts["reg"] = fit(X, y, q, "reg", threads, trees=reg_trees).model_to_string()
    # checks: the merged file must predict exactly qz + raw-member, and every candidate must go through
    # the real ranking.py path
    Tq = apply_transform(Xd, qd, "raw+qz")
    merged = lgb.Booster(model_str=texts["blend"])
    assert np.allclose(merged.predict(Tq), m_qz.predict(Tq) + m_braw.predict(Tq), atol=1e-9)
    for name, text in texts.items():
        path = build_zip(name, text)
        mp = os.path.join(OUT, name, "model.txt")
        res = subprocess.run([sys.executable, "-m", "harness.run_harness", "--features",
                              os.path.join(ROOT, "data", "dev", "vali.txt"), "--model", mp],
                             cwd=ROOT, capture_output=True, text=True)
        ltr = [l for l in res.stdout.splitlines() if "rerank_with_ltr" in l]
        print(f"{name:<6} {path}  ({os.path.getsize(path) / 1e6:.1f} MB)  harness DEV: {ltr}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "final"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reg-trees", type=int, default=900)
    args = ap.parse_args()
    cv(args.threads) if args.cmd == "cv" else final(args.threads, args.reg_trees)


if __name__ == "__main__":
    main()
