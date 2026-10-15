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
    labels = labels.to(scores.dtype)
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
    out = []
    for rows in _query_rows(qids):
        cand = _candidate_pairs(rows, np.asarray(labels)[rows])
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


# ---- provided helpers -------------------------------------------------------

def dev_ndcg(scorer: Callable[[np.ndarray], np.ndarray], X, y, qids) -> float:
    """PROVIDED. Mean nDCG@10 of `scorer` over the queries of (X, y, qids). It scores
    WHOLE queries and sorts them -- evaluation is not batched like training."""
    return mean_ndcg_from_scores(scorer(X), y, qids, k=10)


# ---- training ---------------------------------------------------------------

def train_model(kind: str, train, dev, *, seed: int = 0, epochs: int = 20, batch_size: int = 1024,
                lr: float = 1e-3, pairs_per_query: int = 300, patience: int = 3, hidden=64,
                verbose: bool = False):
    """Train a 'pointwise' or 'pairwise' scorer. train / dev are (X_raw, y, qids)
    tuples. Returns (network, standardizer, history) with the best-dev-nDCG
    weights restored. The setup below is PROVIDED; the loop is yours (TODO(6))."""
    if kind not in ("pointwise", "pairwise"):
        raise ValueError("kind must be 'pointwise' or 'pairwise'")
    X, y, q = train
    Xd, yd, qd = dev
    set_seed(seed)                                    # TODO(1) makes this meaningful
    standardizer = Standardizer().fit(X)              # fit on TRAIN only
    Xt = torch.from_numpy(standardizer.transform(X))
    yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
    network = ScoringNetwork(X.shape[1], hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    rng = np.random.RandomState(seed)

    history = []
    best_nd, best_state, bad_epochs = -1.0, copy.deepcopy(network.state_dict()), 0
    for epoch in range(1, epochs + 1):
        network.train()
        if kind == "pairwise":
            examples = make_pairs(q, y, pairs_per_query, rng)
        else:
            examples = np.arange(len(y))
        examples = examples[rng.permutation(len(examples))]  # reshuffle every epoch

        losses = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            if kind == "pairwise":
                bt = torch.from_numpy(batch)
                losses.append(pairwise_step(network, optimizer, Xt[bt[:, 0]], Xt[bt[:, 1]]))
            else:
                bt = torch.from_numpy(batch)
                losses.append(pointwise_step(network, optimizer, Xt[bt], yt[bt]))

        nd = dev_ndcg(make_scorer(network, standardizer), Xd, yd, qd)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else float("nan"),
                        "dev_ndcg@10": nd})
        if verbose:
            print(f"  {kind} epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  dev nDCG@10 {nd:.4f}")

        # early stopping on the dev split given (keep best weights)
        if nd > best_nd:
            best_nd, best_state, bad_epochs = nd, copy.deepcopy(network.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    network.load_state_dict(best_state)
    return network, standardizer, history
