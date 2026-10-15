# A3 -- Rank Fusion + LTR Arena

Vansh Saini, 2023EE10656. MSLR-WEB10K Fold1, TRAIN = `train.txt`, DEV = `vali.txt`. The held-out `test.txt`
(and every other fold, which contains it) was never extracted or looked at.

| component | what ships | DEV nDCG@10 |
|---|---|---|
| `fuse_candidates` (Track A) | CombSUM over min-max scores, equal weights, LMIR.JM left out | 0.3105 (MAP@10 0.1302) |
| `rerank_with_ltr` (Tracks B, C2) | LightGBM lambdarank, raw + query-z features, 63 leaves, 600 trees | 0.4899 (MAP@10 0.1535) |
| `pairwise_model.pt` (Track P) | RankNet MLP, gain-proportional pair sampling | 0.4419 |
| `pointwise_model.pt` | squared-error MLP, same net | 0.4201 |
| `recipe.py` (Track C1) | size-scaled LambdaMART search + query-z features + train/vali refit | n/a (hidden data) |

For reference: best single base channel 0.2887, textbook RankNet (64 hidden, uniform pairs) about 0.43.

## Reproduce
```
pip install -r requirements.txt
python scripts/prepare_mslr.py --fold-dir <MSLR-WEB10K/Fold1>     # only Fold1 train/vali
python scripts/train_model.py  --train data/train --dev data/dev --out submission/model.txt   # byte-identical
python scripts/train_neural.py --kind both --train data/train --dev data/dev --out-dir submission
python -m harness.run_harness --features data/dev/vali.txt --model submission/model.txt
bash scripts/smoke_test.sh        # toy end-to-end + tests (trains into runs/, never touches submission/)
python scripts/run_recipe.py      # C1 dry run on the synthetic transfer sets
```
Experiments behind every choice: `experiments/` (scripts + raw results in `experiments/results/`).
Local macOS note: torch and lightgbm in one process clash (two libomp copies); run tests with
`OMP_NUM_THREADS=1`. Linux is unaffected.

## Training-data declaration
| model | trained on | tuned on | early-stopped on |
|---|---|---|---|
| LambdaMART `model.txt` | TRAIN only | grouped 5-fold CV inside TRAIN | nothing (tree count fixed from CV) |
| `pointwise_model.pt` | 90% of TRAIN queries | TRAIN holdout | 10% TRAIN query holdout |
| `pairwise_model.pt` | 90% of TRAIN queries | TRAIN holdout | 10% TRAIN query holdout |
| fusion weights | n/a (no learning) | TRAIN | n/a |
| `recipe.py` (staff run it on hidden benchmarks) | each benchmark's train, then train+vali refit | MSLR-TRAIN-derived proxies + synthetic transfer sets | each benchmark's vali |

DEV was never trained on, tuned on, or early-stopped on. It was read once per shipped component, after the
choice was made, to report the numbers above.

## Fusion method and why
CombSUM: min-max normalise each channel within the candidate pool, sum with equal weights over BM25 (whole
document), BM25 (title) and LMIR.DIR; LMIR.JM gets weight 0. Picked from 2025 configs (RRF with c in
{1..120}, CombSUM, CombMNZ, channel weights in {0, .5, 1, 2}) scored on TRAIN. The top of TRAIN is a flat
plateau of CombSUM/CombMNZ configs without JM, so I took the simplest one. JM is the weakest channel on
its own and the same model family as DIR, so it mostly re-counts DIR's opinion. Score-based beats RRF here
(TRAIN 0.3000 vs best RRF 0.2979) because the channels are close in quality, so the score gaps carry
information that RRF throws away.

## Feature engineering
One addition, used by both `model.txt` and `recipe.py`: for every feature, its z-score within the query's
candidate pool (`submission/ltr_common.py:query_zscore`), appended to the raw features (136 -> 272).
Trees split on absolute thresholds, but how a document compares with the other candidates of the same
query is what ranking needs, and raw values like BM25 swing a lot from query to query. The transform is
recorded in the model's feature names, so `prepare()` rebuilds it for any feature dimension.
CV on TRAIN: +0.005 (0.4806 vs 0.4752 best raw). It also held up better under generic test-set shifts
(rescaled columns, dropped columns, noise) and was about neutral when half the candidate pool was
removed (`experiments/robustness.py`).

## LambdaMART hyperparameters and how chosen
`objective=lambdarank, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.5,
bagging_fraction=0.8 (freq 1), lambda_l2=1, 600 trees, truncation level 30 (default)`.
Grouped 5-fold CV on TRAIN (`harness.cv.grouped_kfold`, folds by query), tree count swept at fixed values,
no early stopping on any reported fold:
- stage 1 (lr 0.1): leaves {31, 127, 255} x min_data {50, 400} x feature_fraction {.5, .8}. Small trees
  generalise best: 31 leaves 0.4730 (train 0.60), 127 leaves 0.4707 (train 0.69), 255 leaves 0.4682 at
  only 100 trees. Bigger trees fit TRAIN far harder for a lower validation score.
- stage 2: lr 0.05 +0.002; query-z features +0.004; lambdarank truncation 10 or 20 hurts vs default 30.
- stage 3 (query-z, lr 0.05): 63 leaves / min_data 200 peaks at 0.4806 ± 0.0068 with a flat plateau from
  300 to 800 trees (train 0.63 -> 0.71); 31-leaf variants reach 0.4798 at 1500-2000 trees.

## Final configuration: dev accuracy vs generalisation
I took 63 leaves with min_data_in_leaf 200 and 600 trees, not the configuration that fits TRAIN hardest.
Every config in the top group is within one fold standard error of the others. This one has the widest
plateau, so the exact tree count matters least. min_data 200 (about two queries' worth of documents per
leaf) stops leaves from memorising single queries, and 600 sits mid-plateau, adjusted for the full-TRAIN
refit having 25% more data than a CV fold. Training on DEV as well would raise DEV scores but says nothing
about unseen queries, so it wasn't done.

## AI-use disclosure
An AI assistant (Claude, by Anthropic, through Claude Code) was used throughout: planning, writing and
debugging code in `submission/`, `scripts/`, `experiments/` and `tests/`, running the experiments, and
drafting this README. I directed the work, made the decisions, and checked every result reported here
against the logged experiment output in `experiments/results/`.

## Code provenance
No external code was copied. Formulas and definitions from: Cormack, Clarke and Buettcher (SIGIR 2009) for
RRF; Fox and Shaw (TREC-2 1994) for CombSUM/CombMNZ; Burges et al. (ICML 2005) for the RankNet loss;
Burges (MSR-TR-2010-82) for the |delta nDCG| pair weighting (LambdaRank); LightGBM documentation for the
`lambdarank` objective and its parameters. The starter repository's harness, metrics, CV split and
neural-net loader are used unchanged.

---

## Starter layout (unchanged)
`harness/` metrics, I/O, CV, leaderboard formulas, runner (staff run an identical copy) - `submission/` the
entry - `scripts/` tooling - `experiments/` my experiments - `data/toy/` and `data/toy_transfer/`
synthetic data - `tests/` (starter tests + `tests/test_recipe_robustness.py`).
Docs: `docs/SUBMISSION_INTERFACE.md`, `docs/GRADING.md`, `docs/RANKNET_PRIMER.md`, `docs/DATA.md`,
`docs/TRANSFER_TRACK.md`, `docs/SUBMISSION_NOTES.md`.
