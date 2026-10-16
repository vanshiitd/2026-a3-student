"""
submission/pointwise_pairwise.py -- the pointwise / pairwise models (assignment Sections 4.1, 4.3 and 7).
Read docs/RANKNET_PRIMER.md first; no prior neural-network experience is assumed.

Both models use the SAME scoring network (harness/neural.py: ScoringNetwork), so the only difference between
them is the loss you write:
  pointwise: squared error between the network's score and the graded label
  pairwise : RankNet -- for a pair (i more relevant than j), loss =
             log(1 + exp(-(s_i - s_j))), i.e. -log sigmoid(s_i - s_j)

PROVIDED : ScoringNetwork, make_scorer, save_scorer, load_scorer (harness/neural.py), dev_ndcg, and the
           overall shape of train_model().
YOU WRITE: every `TODO(...)` below (roughly 100-150 lines in total):
  TODO(1) set_seed            TODO(2) pointwise_loss     TODO(3) make_pairs
  TODO(4) ranknet_loss        TODO(5) pointwise_step / pairwise_step
  TODO(6) the training loop inside train_model(): minibatching, per-epoch dev
          nDCG@10, early stopping.
ALLOWED  : PyTorch tensors, autograd, nn layers, activations, optimizers
           (Adam/SGD), random-seed utilities.
NOT ALLOWED: any ready-made ranking loss or LTR trainer (allRank, TF-Ranking,
           PT-Ranking, RankLib, ...) and LightGBM/XGBoost for these two
           models. torch's MarginRankingLoss is a HINGE loss, not RankNet.

TWO JOBS, KEEP THEM SEPARATE.
  1. Correctness (hand-checked): the functions below are tested by NAME and SIGNATURE (docs/GRADING.md) against
     their textbook definitions. Keep them exactly as declared -- in particular ranknet_loss must stay the plain,
     unweighted RankNet loss.
  2. Track P (leaderboard): your submitted pairwise_model.pt is scored on held-out data. To make it rank better you
     MAY change, in your own extra functions / train_model options: how pairs are sampled (e.g. favour large
     label gaps or top-ranked documents), pairs per query, epochs, learning rate, batch size, the MLP's width and
     depth (at most MAX_HIDDEN_LAYERS hidden layers of at most MAX_WIDTH units: pass e.g. hidden=[128, 64]), and
     per-pair weights such as |delta nDCG| (which turns RankNet into LambdaRank) -- write such weighting as a
     separate function (e.g. weighted_ranknet_loss), never inside ranknet_loss.
     You may NOT: train on anything but relevance labels (no distillation from LambdaMART or any other model's
     scores), or change the artifact format: staff load it with their own copy of harness/neural.py, and an
     artifact that breaks the bounds scores 0 on Track P. Declare the splits you trained and tuned on.

Artifacts to ship in submission/: pointwise_model.pt and pairwise_model.pt (see scripts/train_neural.py).
"""
import copy
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from harness.metrics import mean_ndcg_from_scores
from harness.preprocessing import Standardizer
# PROVIDED, and also what staff use to load your artifacts (assignment Section 7, Track P):
from harness.neural import MAX_HIDDEN_LAYERS, MAX_WIDTH, ScoringNetwork, load_scorer, make_scorer, save_scorer  # noqa: F401


def set_seed(seed: int) -> None:
    """Seed python `random`, numpy and torch so two runs with the same seed
    give identical weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pointwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean squared error between predicted scores and graded labels
    (both 1-D tensors of equal length). Returns a scalar tensor.
    Example: scores=[1, 2], labels=[0, 2] -> ((1-0)^2 + (2-2)^2) / 2 = 0.5"""
    scores, labels = scores.reshape(-1), labels.reshape(-1).to(scores.dtype)  # (n,1) vs (n,) would broadcast
    if scores.shape != labels.shape:
        raise ValueError(f"pointwise_loss: {scores.numel()} scores vs {labels.numel()} labels")
    return ((scores - labels) ** 2).mean()


def make_pairs(qids, labels, pairs_per_query: int, rng: np.random.RandomState) -> np.ndarray:
    """TODO(3): build within-query preference pairs.

    qids  : one query id per ROW (any hashable; rows of a query may be anywhere)
    labels: one graded relevance label per row
    Return an int array of shape (P, 2) of ROW INDICES: column 0 is the
    strictly MORE relevant document of the pair, column 1 the less relevant one.
    Rules (all checked by staff): (a) both rows from the SAME query; (b) only
    pairs with different labels (ties carry no preference); (c) oriented
    more-relevant first; (d) at most `pairs_per_query` pairs per query -- sample
    without replacement using `rng` (a NumPy RandomState) so runs are
    reproducible. Never build the pairs of the whole TRAIN split at once.
    Example: qids=[a,a,a,b,b], labels=[2,0,1,1,1], pairs_per_query large ->
    the 3 pairs (0,1), (0,2), (2,1); query b has only ties -> no pairs."""
    lab_all = np.asarray(labels)  # once, not per query (lists would make this O(rows x queries))
    out = []
    for rows in _query_rows(qids):
        cand = _candidate_pairs(rows, lab_all[rows])
        if len(cand) == 0:
            continue
        if len(cand) > pairs_per_query:
            cand = cand[rng.choice(len(cand), pairs_per_query, replace=False)]
        out.append(cand)
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(out).astype(np.int64)


def _query_rows(qids) -> List[np.ndarray]:
    """row indices of each query, queries in first-seen order (rows can be anywhere)"""
    groups: Dict = {}
    for i, q in enumerate(qids):
        groups.setdefault(q, []).append(i)
    return [np.asarray(r, dtype=np.int64) for r in groups.values()]


def _candidate_pairs(rows: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """all strictly ordered pairs of one query, more relevant row first"""
    a, b = np.triu_indices(len(rows), k=1)
    keep = lab[a] != lab[b]
    a, b = a[keep], b[keep]
    swap = lab[a] < lab[b]
    hi = np.where(swap, b, a)
    lo = np.where(swap, a, b)
    return np.stack([rows[hi], rows[lo]], axis=1)


def ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor) -> torch.Tensor:
    """RankNet loss. For each pair, log(1 + exp(-(s_i - s_j))), where
    s_preferred = s_i is the score of the more relevant document, averaged over
    the pairs. softplus(-d) is the stable form of log(1 + e^-d).
    Example: s_preferred=[2, 0], s_other=[0, 0] ->
    mean(log(1+e^-2), log 2) = mean(0.126928, 0.693147) = 0.410038"""
    s_preferred, s_other = s_preferred.reshape(-1), s_other.reshape(-1)  # no silent (n,1) x (n,) broadcast
    if s_preferred.shape != s_other.shape:
        raise ValueError(f"ranknet_loss: {s_preferred.numel()} vs {s_other.numel()} scores")
    return torch.nn.functional.softplus(-(s_preferred - s_other)).mean()


def pointwise_step(network: nn.Module, optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    """ONE optimisation step on a minibatch of documents: zero grads, forward,
    pointwise_loss, backward, step. Returns the loss as a float."""
    network.train()
    optimizer.zero_grad()
    loss = pointwise_loss(network(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def pairwise_step(network: nn.Module, optimizer, x_preferred: torch.Tensor, x_other: torch.Tensor) -> float:
    """ONE optimisation step on a minibatch of PAIRS. Both sides go through the
    SAME network (shared weights), then ranknet_loss, backward, step. Returns
    the loss as a float."""
    network.train()
    optimizer.zero_grad()
    loss = ranknet_loss(network(x_preferred), network(x_other))
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ---- Track P extras (not graded by name; ranknet_loss / make_pairs above stay textbook) ----------

def weighted_ranknet_loss(s_preferred: torch.Tensor, s_other: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """RankNet loss with a weight per pair, normalised by the weight sum. With w = |delta nDCG| this is
    LambdaRank's gradient direction. w = all ones gives back plain ranknet_loss."""
    per_pair = torch.nn.functional.softplus(-(s_preferred - s_other))
    return (w * per_pair).sum() / w.sum().clamp_min(1e-12)


def weighted_pairwise_step(network: nn.Module, optimizer, x_preferred, x_other, w) -> float:
    network.train()
    optimizer.zero_grad()
    loss = weighted_ranknet_loss(network(x_preferred), network(x_other), w)
    loss.backward()
    optimizer.step()
    return float(loss.item())


class _QueryIndex:
    """per-row integer query codes + per-query ideal DCG, built once per training run"""

    def __init__(self, qids, labels):
        codes, uniq = {}, []
        self.code = np.empty(len(qids), dtype=np.int64)
        for i, q in enumerate(qids):
            if q not in codes:
                codes[q] = len(uniq)
                uniq.append(q)
            self.code[i] = codes[q]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.gain = (2.0 ** self.labels) - 1.0
        n_q = len(uniq)
        # ideal DCG over the full list (untruncated), per query
        order = np.lexsort((-self.gain, self.code))
        pos = _positions_within_group(self.code[order])
        disc = 1.0 / np.log2(pos + 1.0)
        self.idcg = np.bincount(self.code[order], weights=self.gain[order] * disc, minlength=n_q)

    def ranks(self, scores: np.ndarray) -> np.ndarray:
        """1-based rank of each row inside its query under `scores` (ties by row order)"""
        order = np.lexsort((np.arange(len(scores)), -scores, self.code))
        r = np.empty(len(scores), dtype=np.float64)
        r[order] = _positions_within_group(self.code[order])
        return r


def _positions_within_group(sorted_codes: np.ndarray) -> np.ndarray:
    """1,2,3.. restarting at every new code (codes already sorted)"""
    n = len(sorted_codes)
    starts = np.r_[True, sorted_codes[1:] != sorted_codes[:-1]]
    first = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    return (np.arange(n) - first + 1).astype(np.float64)


def lambda_weights(pairs: np.ndarray, qindex: "_QueryIndex", scores: np.ndarray, top: int = 0) -> np.ndarray:
    """|delta nDCG| if the two docs of each pair swapped places under the current scores.
    top > 0 truncates the discount at that rank (nDCG@top); 0 = full-list discount."""
    r = qindex.ranks(scores)
    hi, lo = pairs[:, 0], pairs[:, 1]

    def disc(rank):
        d = 1.0 / np.log2(rank + 1.0)
        return np.where(rank <= top, d, 0.0) if top > 0 else d

    dg = np.abs(qindex.gain[hi] - qindex.gain[lo])
    dd = np.abs(disc(r[hi]) - disc(r[lo]))
    idcg = qindex.idcg[qindex.code[hi]]
    return dg * dd / np.maximum(idcg, 1e-12)


def make_pairs_gain_sampled(qids, labels, pairs_per_query: int, rng: np.random.RandomState,
                            power: float = 1.0) -> np.ndarray:
    """like make_pairs, but when a query has more candidate pairs than the cap, sample them with
    probability proportional to (2^l_hi - 2^l_lo)^power (favours pairs with a highly relevant doc)"""
    lab_all = np.asarray(labels)
    out = []
    for rows in _query_rows(qids):
        cand = _candidate_pairs(rows, lab_all[rows])
        if len(cand) == 0:
            continue
        if len(cand) > pairs_per_query:
            gap = ((2.0 ** lab_all[cand[:, 0]]) - (2.0 ** lab_all[cand[:, 1]])) ** power
            cand = cand[rng.choice(len(cand), pairs_per_query, replace=False, p=gap / gap.sum())]
        out.append(cand)
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(out).astype(np.int64)


def fit_robust_standardizer(X: np.ndarray) -> Standardizer:
    """Standardizer variant for heavy-tailed features: centre on the median, scale by a high-quantile
    spread instead of the std, so a few huge values (click counts etc.) don't squash everything else.
    Same {mean, std} artifact format, so the staff loader is unchanged."""
    med = np.median(X, axis=0).astype(np.float64)
    q_lo, q_hi = np.percentile(X, [5, 95], axis=0)
    spread = ((q_hi - q_lo) / 3.29).astype(np.float64)  # 90% range of a normal is 3.29 sd
    std = X.std(axis=0).astype(np.float64)
    spread = np.where(spread < 1e-8, std, spread)
    spread[spread < 1e-8] = 1.0
    return Standardizer(med, spread)


# ---- provided helpers -------------------------------------------------------

def dev_ndcg(scorer: Callable[[np.ndarray], np.ndarray], X, y, qids) -> float:
    """PROVIDED. Mean nDCG@10 of `scorer` over the queries of (X, y, qids). It scores
    WHOLE queries and sorts them -- evaluation is not batched like training."""
    return mean_ndcg_from_scores(scorer(X), y, qids, k=10)


# ---- training ---------------------------------------------------------------

def train_model(kind: str, train, dev, *, seed: int = 0, epochs: int = 20, batch_size: int = 1024,
                lr: float = 1e-3, pairs_per_query: int = 300, patience: int = 3, hidden=64,
                verbose: bool = False, sampler: str = "uniform", weighting: str = "none",
                lambda_top: int = 0, weight_decay: float = 0.0, scaler: str = "standard"):
    """Train a 'pointwise' or 'pairwise' scorer. train / dev are (X_raw, y, qids)
    tuples. Returns (network, standardizer, history) with the best-dev-nDCG
    weights restored.

    dev=None -> no early stopping, run exactly `epochs` epochs (used for the final refit
    on all of TRAIN once the epoch count is picked on a TRAIN holdout).
    Track P options (defaults = textbook RankNet):
      sampler   'uniform' (make_pairs) | 'gain' / 'gain2' (make_pairs_gain_sampled, power 1 / 2)
      weighting 'none' | 'lambda' (|delta nDCG| per pair, recomputed every epoch -> LambdaRank)
      lambda_top  truncation rank for the lambda discount, 0 = whole list
      weight_decay  Adam L2
      scaler    'standard' | 'robust' (median / quantile spread, same artifact format)"""
    if kind not in ("pointwise", "pairwise"):
        raise ValueError("kind must be 'pointwise' or 'pairwise'")
    X, y, q = train
    set_seed(seed)
    if scaler == "robust":
        standardizer = fit_robust_standardizer(X)  # fit on TRAIN only
    else:
        standardizer = Standardizer().fit(X)       # fit on TRAIN only
    Xt = torch.from_numpy(standardizer.transform(X))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    network = ScoringNetwork(X.shape[1], hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.RandomState(seed)
    if sampler == "gain":
        pair_fn = make_pairs_gain_sampled
    elif sampler == "gain2":
        def pair_fn(qq, yy, ppq, r):
            return make_pairs_gain_sampled(qq, yy, ppq, r, power=2.0)
    else:
        pair_fn = make_pairs
    qindex = _QueryIndex(q, y) if (kind == "pairwise" and weighting == "lambda") else None

    history = []
    best_nd, best_state, bad_epochs = -1.0, copy.deepcopy(network.state_dict()), 0
    for epoch in range(1, epochs + 1):
        if kind == "pairwise":
            examples = pair_fn(q, y, pairs_per_query, rng)
        else:
            examples = np.arange(len(y))
        examples = examples[rng.permutation(len(examples))]  # reshuffle every epoch

        weights = None
        if qindex is not None:
            # scores frozen at the start of the epoch, good enough for the swap weights
            network.eval()
            with torch.no_grad():
                cur = network(Xt).numpy().astype(np.float64)
            weights = torch.from_numpy(lambda_weights(examples, qindex, cur, lambda_top).astype(np.float32))

        network.train()
        losses = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            bt = torch.from_numpy(batch)
            if kind == "pairwise" and weights is not None:
                losses.append(weighted_pairwise_step(network, optimizer, Xt[bt[:, 0]], Xt[bt[:, 1]],
                                                     weights[start:start + batch_size]))
            elif kind == "pairwise":
                losses.append(pairwise_step(network, optimizer, Xt[bt[:, 0]], Xt[bt[:, 1]]))
            else:
                losses.append(pointwise_step(network, optimizer, Xt[bt], yt[bt]))

        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan")}
        if dev is not None:
            rec["dev_ndcg@10"] = dev_ndcg(make_scorer(network, standardizer), *dev)
        history.append(rec)
        if verbose:
            print(f"  {kind} epoch {epoch:2d}  loss {rec['train_loss']:.4f}  "
                  f"dev nDCG@10 {rec.get('dev_ndcg@10', float('nan')):.4f}", flush=True)

        if dev is None:
            continue
        # early stopping on the dev split given (keep best weights)
        nd = rec["dev_ndcg@10"]
        if nd > best_nd:
            best_nd, best_state, bad_epochs = nd, copy.deepcopy(network.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if dev is not None:
        network.load_state_dict(best_state)
    return network, standardizer, history
