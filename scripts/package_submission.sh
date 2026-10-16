#!/usr/bin/env bash
# Build the submission ZIP from the committed tree only (git archive), so untracked files -- the MSLR
# data in data/train, data/dev, data/cache, the venv, runs/ -- can never end up in it.
# Extracting the zip gives the repository's directory structure at its root.
set -euo pipefail
cd "$(dirname "$0")/.."
# whole tree, not just code dirs: git archive only packs what's committed, so an uncommitted README or
# experiments edit would silently be missing from the zip
if [ -n "$(git status --porcelain)" ]; then
  echo "uncommitted changes in the repo -- commit (or stash) them first:" >&2
  git status --short >&2
  exit 1
fi
mkdir -p dist
out="dist/${1:-2023EE10656}.zip"
rm -f "$out"
git archive --format=zip -o "$out" HEAD
echo "wrote $out ($(du -h "$out" | cut -f1)) from $(git rev-parse --short HEAD)"
