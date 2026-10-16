"""
submission/fusion.py -- rank fusion over the named base-ranker scores that
fuse_candidates() receives (assignment Sections 3.1 and 4.1).

Input convention for every function here: `base_scores` maps each candidate
doc_id to {channel_name: score}, HIGHER = BETTER for every channel (LMIR
scores are log-probabilities, i.e. negative -- larger is still better).
Output: {doc_id: fused_score}, higher = better. Ties in a channel are broken
by the dict's insertion order (the candidate order), so results are
deterministic.

What you must write (Section 4.1): RRF (TODO(fusion-0)) and at least ONE
score-based method, CombSUM or CombMNZ (TODO(fusion-1)). Hand-written: no
fusion library. Staff test these functions against small hand-checkable examples
(docs/GRADING.md).
"""
from typing import Dict, Iterable, List, Optional


def _channels(base_scores: Dict[str, Dict[str, float]], channels: Optional[Iterable[str]]) -> List[str]:
    if channels is not None:
        return list(channels)
    first = next(iter(base_scores.values()), {})
    return list(first.keys())


def _weight(weights: Optional[Dict[str, float]], ch: str) -> float:
    return 1.0 if weights is None else float(weights.get(ch, 0.0))


def rrf_scores(base_scores: Dict[str, Dict[str, float]], channels: Optional[Iterable[str]] = None,
               c: float = 60.0, weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Reciprocal rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009):
    score(d) = sum over channels of 1 / (c + rank_channel(d)), ranks 1-based,
    best score = rank 1. Needs only each channel's ORDER, never its scale.
    `weights` (optional) scales each channel's 1/(c+rank) term; missing channel = 0."""
    docs = list(base_scores.keys())
    fused = {d: 0.0 for d in docs}
    for ch in _channels(base_scores, channels):
        w = _weight(weights, ch)
        if w == 0.0:
            continue
        # python sort is stable so equal scores keep candidate order
        order = sorted(docs, key=lambda d: -base_scores[d][ch])
        for rank, d in enumerate(order, start=1):
            fused[d] += w / (c + rank)
    return fused


def minmax_normalize(values: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise {doc: score} to [0, 1] within one channel. A channel
    where every document has the same score normalises to all zeros."""
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {d: 0.0 for d in values}
    return {d: (v - lo) / (hi - lo) for d, v in values.items()}


def _normalised_channels(base_scores, channels):
    """{channel: {doc: minmax score}} over the candidate pool."""
    return {ch: minmax_normalize({d: s[ch] for d, s in base_scores.items()})
            for ch in _channels(base_scores, channels)}


def combsum_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None,
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """CombSUM (Fox & Shaw, TREC-2 1994): sum of the per-channel MIN-MAX-
    normalised scores. `weights` (optional) scales each channel; missing channel = 0."""
    norm = _normalised_channels(base_scores, channels)
    fused = {d: 0.0 for d in base_scores}
    for ch, vals in norm.items():
        w = _weight(weights, ch)
        for d, v in vals.items():
            fused[d] += w * v
    return fused


def combmnz_scores(base_scores: Dict[str, Dict[str, float]],
                   channels: Optional[Iterable[str]] = None,
                   weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """CombMNZ: CombSUM multiplied by the number of channels in which the
    document has a normalised score > 0. (Every candidate here has a score
    from every channel, so 'appears in the list' is defined as 'normalised
    score > 0' -- see docs/SUBMISSION_INTERFACE.md.)
    With `weights`, the sum is weighted and the count only includes channels with weight > 0; channels with
    weight <= 0 are left out entirely (CombSUM, by contrast, would add a negative weight's term -- the shipped
    weights are all >= 0, so the two never differ in practice)."""
    norm = _normalised_channels(base_scores, channels)
    total = {d: 0.0 for d in base_scores}
    hits = {d: 0 for d in base_scores}
    for ch, vals in norm.items():
        w = _weight(weights, ch)
        if w <= 0.0:
            continue
        for d, v in vals.items():
            total[d] += w * v
            if v > 0.0:
                hits[d] += 1
    return {d: total[d] * hits[d] for d in base_scores}
