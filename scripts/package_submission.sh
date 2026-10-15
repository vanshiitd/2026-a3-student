#!/usr/bin/env bash
# Build the submission ZIP from the committed tree only (git archive), so untracked files -- the MSLR
# data in data/train, data/dev, data/cache, the venv, runs/ -- can never end up in it.
# Extracting the zip gives the repository's directory structure at its root.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain -- submission scripts harness tests)" ]; then
  echo "uncommitted changes under submission/ scripts/ harness/ tests/ -- commit first" >&2
  exit 1
fi
mkdir -p dist
out="dist/${1:-2023EE10656}.zip"
rm -f "$out"
git archive --format=zip -o "$out" HEAD
echo "wrote $out ($(du -h "$out" | cut -f1)) from $(git rev-parse --short HEAD)"
