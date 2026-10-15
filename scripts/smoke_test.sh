#!/usr/bin/env bash
# End-to-end sanity check on toy data: regenerate data, train, run the harness, run the tests.
# Trains into runs/ (gitignored), NOT submission/model.txt, so running this never overwrites the shipped model.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
python scripts/make_toy_data.py --out data/toy
python scripts/train_model.py --train data/toy --out runs/toy_model.txt --trees 30
python -m harness.run_harness --features data/toy/vali.txt --model runs/toy_model.txt
python -m harness.run_harness --features data/toy/vali.txt --model submission/model.txt
python -m pytest tests --ignore=tests/test_student_todos.py -q
