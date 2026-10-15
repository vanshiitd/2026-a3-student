"""Edge cases the hidden C1 benchmarks could throw at recipe.py / rerank_with_ltr that the toy sets don't:
sparse lines, files of different widths, out-of-order feature indices, split-up queries, comments,
fewer label levels. A crash on any of these scores 0 on that benchmark."""
import os

import numpy as np
import pytest

from harness.feature_io import read_svmlight_ltr, stack
from harness.recipe_eval import run_recipe_and_evaluate
from submission import ltr_common


def _write(path, rows):
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def _random_rows(rng, n_q, docs, width, max_label=4, sparse=True, drop_last=False, comment=False, qid0=1):
    rows = []
    for q in range(n_q):
        for _ in range(docs):
            x = rng.randn(width)
            label = int(np.clip(round(x[0] + 0.5 * rng.randn() + 1), 0, max_label))
            if sparse:
                x[rng.rand(width) < 0.3] = 0.0
            if drop_last:
                x[-1] = 0.0
            toks = [f"{j + 1}:{v:.5g}" for j, v in enumerate(x) if not (sparse and v == 0.0)]
            line = f"{label} qid:{qid0 + q} " + " ".join(toks)
            rows.append(line + ("  # doc comment" if comment else ""))
    return rows


def test_read_into_matches_simple_reader(tmp_path):
    rng = np.random.RandomState(0)
    tr, va = tmp_path / "train.txt", tmp_path / "vali.txt"
    _write(tr, _random_rows(rng, 30, 12, 25, comment=True))
    _write(va, _random_rows(rng, 10, 12, 25, drop_last=True, qid0=500))  # narrower file
    for transform in ("raw", "raw+qz"):
        T, y, (g_tr, g_va), width = ltr_common.read_into([str(tr), str(va)], transform)
        Xtr, ytr, qtr = ltr_common.read_ltr(str(tr))
        Xva, yva, qva = ltr_common.read_ltr(str(va))
        X = np.vstack([ltr_common.pad_to(Xtr, width), ltr_common.pad_to(Xva, width)])
        expect = ltr_common.apply_transform(X, None, transform, sizes=ltr_common.group_sizes(qtr) +
                                            ltr_common.group_sizes(qva))
        assert width == 25
        assert np.array_equal(T, expect) and np.array_equal(y, np.concatenate([ytr, yva]))


def test_read_ltr_matches_harness_reader(tmp_path):
    rng = np.random.RandomState(1)
    p = tmp_path / "f.txt"
    _write(p, _random_rows(rng, 8, 9, 17, comment=True))
    X, y, q = ltr_common.read_ltr(str(p))
    X2, y2, q2 = stack(read_svmlight_ltr(str(p)))
    assert np.array_equal(X, X2) and np.array_equal(y, y2) and list(q) == list(q2)


def test_unordered_indices_fall_back(tmp_path):
    rng = np.random.RandomState(2)
    tr, va = tmp_path / "train.txt", tmp_path / "vali.txt"
    def largest_first(rows):
        # move the largest index to the front of every line, so the cheap scan underestimates the width
        out = []
        for row in rows:
            parts = row.split()
            out.append(" ".join(parts[:2] + [parts[-1]] + parts[2:-1]))
        return out

    _write(tr, largest_first(_random_rows(rng, 40, 10, 12, sparse=False)))
    _write(va, largest_first(_random_rows(rng, 15, 10, 12, sparse=False, qid0=900)))
    with pytest.raises(ltr_common.WidthError):
        ltr_common.read_into([str(tr), str(va)], "raw")
    r = run_recipe_and_evaluate(str(tr), str(va), str(va), str(tmp_path / "m.txt"), seed=0)
    assert r["ndcg@10"] > 0.5


def test_split_queries_and_three_label_levels(tmp_path):
    rng = np.random.RandomState(3)
    rows = _random_rows(rng, 40, 10, 15, max_label=2)
    rows = rows[::2] + rows[1::2]  # every query's rows now split into two non-adjacent runs
    tr, va = tmp_path / "train.txt", tmp_path / "vali.txt"
    _write(tr, rows)
    _write(va, _random_rows(rng, 15, 10, 15, max_label=2, qid0=900))
    T, y, (g_tr, g_va), width = ltr_common.read_into([str(tr), str(va)], "raw")
    assert len(g_tr) == 40 and sum(g_tr) == 400  # regrouped, still 40 queries
    r = run_recipe_and_evaluate(str(tr), str(va), str(va), str(tmp_path / "m.txt"), seed=0)
    assert r["ndcg@10"] > 0.5


def test_eval_file_narrower_than_training(tmp_path):
    """held-out file whose last feature is zero everywhere -> rerank_with_ltr must pad, not crash"""
    rng = np.random.RandomState(4)
    tr, va, te = tmp_path / "train.txt", tmp_path / "vali.txt", tmp_path / "test.txt"
    _write(tr, _random_rows(rng, 40, 10, 20))
    _write(va, _random_rows(rng, 15, 10, 20, qid0=500))
    _write(te, _random_rows(rng, 15, 10, 20, drop_last=True, qid0=900))
    r = run_recipe_and_evaluate(str(tr), str(va), str(te), str(tmp_path / "m.txt"), seed=0)
    assert r["num_eval_queries"] == 15 and r["ndcg@10"] > 0.5
