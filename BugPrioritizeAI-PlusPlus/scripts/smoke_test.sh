#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
python -m bugprioritize make-demo --out data/demo_raw.jsonl --projects 15 --bugs-per-project 3 --tests-per-bug 8
python -m bugprioritize build --input data/demo_raw.jsonl --out data/demo_processed.jsonl --filter-log logs/demo_filters.csv
python -m bugprioritize make-splits --data data/demo_processed.jsonl --out splits/demo --protocol projectwise
python -m bugprioritize train --data data/demo_processed.jsonl --split splits/demo/fold_1.json --out results/smoke --encoder hash --hidden 24 --vocab 256 --epochs 2 --batch-size 12
python -m bugprioritize evaluate --rankings results/smoke/rankings --out results/smoke/recalculated
python -m bugprioritize baseline --data data/demo_processed.jsonl --split splits/demo/fold_1.json --out results/smoke_bm25 --method bm25
pytest -q
