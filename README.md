[![DOI](https://zenodo.org/badge/1467459684.svg)](https://doi.org/10.5281/zenodo.24439779)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/mamathat7/BugPrioritizeAI-PlusPlus)

# BugPrioritizeAI++

### Multimodal Test-Case Prioritisation via Retrieval-Augmented Fusion of Bug Reports, Code Diffs, and Historical Failure Signals

## Overview

**BugPrioritizeAI++** is a retrieval-augmented multimodal framework for intelligent regression test-case prioritisation. It integrates bug reports, source-code changes, test-case metadata, historical failure information, and execution costs to predict test failure risks and prioritise test execution.

The framework combines natural-language processing, code representation learning, historical information retrieval, patch-aware attention, and availability-aware gated fusion to produce cost-aware test-case rankings.

By identifying high-risk test cases and prioritising their execution, BugPrioritizeAI++ aims to improve early fault detection and regression testing efficiency.

## Key Features

- **Multimodal Information Processing:** Integrates bug reports, source-code diffs, test metadata, and historical failure signals.
- **Shared Representation Learning:** Uses natural-language and code encoders to extract meaningful representations.
- **Patch-Aware Diff Attention:** Captures relevant information from source-code modifications.
- **Temporal Failure Analysis:** Incorporates historical test execution and failure patterns.
- **Historical Retrieval:** Retrieves relevant historical information to support test prioritisation.
- **Availability-Aware Gated Fusion:** Dynamically combines available modalities while handling missing information.
- **Failure-Risk Prediction:** Estimates the likelihood of individual test cases revealing failures.
- **Cost-Aware Ranking:** Considers test execution costs when prioritising test cases.
- **Leakage-Controlled Evaluation:** Supports project-wise, time-wise, and cross-benchmark data splitting.
- **Comprehensive Evaluation:** Includes baseline comparisons, ablation studies, statistical analysis, and per-test audit trails.

## Methodology

The BugPrioritizeAI++ framework consists of the following major components:

### 1. Multimodal Feature Extraction

Extracts information from different sources:

- Bug reports and issue descriptions.
- Source-code changes and repair patches.
- Test-case names, identifiers, and metadata.
- Historical test execution results.
- Test execution time and cost information.

### 2. Shared Natural-Language and Code Encoding

Processes textual and code-related information using shared representation-learning components. CodeBERT is supported as the pretrained backbone.

### 3. Patch-Aware Diff Attention

Analyses code modifications to identify relevant changes associated with potential software defects.

### 4. Temporal Failure-History Modelling

Captures historical test outcomes and temporal failure patterns to identify potentially fault-revealing test cases.

### 5. Historical Retrieval

Retrieves relevant historical information from previously observed test executions and failure records.

### 6. Availability-Aware Gated Fusion

Combines the available multimodal representations using gated fusion while accounting for unavailable modalities.

### 7. Failure-Risk Prediction

Predicts the failure risk associated with individual candidate test cases.

### 8. Cost-Aware Test Prioritisation

Combines predicted failure risk and execution cost to generate an ordered list of test cases for execution.

## System Workflow

1. Collect benchmark data and test execution information.
2. Normalise datasets and verify test labels.
3. Extract bug reports, code diffs, test metadata, and historical failure information.
4. Generate multimodal representations.
5. Apply patch-aware attention and temporal failure-history modelling.
6. Retrieve relevant historical information.
7. Perform availability-aware gated fusion.
8. Predict test-case failure risks.
9. Apply cost-aware test ranking.
10. Evaluate prioritisation performance using established metrics.

## Datasets and Benchmarks

BugPrioritizeAI++ supports three public software-testing benchmark families.

| Benchmark | Application |
|---|---|
| Defects4J v2.0.1 | Main Java defect benchmark |
| BugsInPy | Python cross-language evaluation |
| Bears | Java CI/build-pair evaluation |

### Official Repositories

- [Defects4J](https://github.com/rjust/defects4j)
- [BugsInPy](https://github.com/soarsmu/BugsInPy)
- [Bears](https://github.com/bears-bugs/bears-benchmark)

Each benchmark requires its respective installation, execution environment, and data preparation.

## Installation

### Requirements

- Python 3.10 or later
- PyTorch
- Git
- Java 8 for the relevant Defects4J v2.x and Bears execution environments
- Maven for Bears
- Additional dependencies for optional model and evaluation components

### 1. Clone the Repository

```bash
git clone https://github.com/mamathat7/BugPrioritizeAI-PlusPlus.git
cd BugPrioritizeAI-PlusPlus
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

For the additional model, retrieval, tree-ranking, and plotting features:

```bash
python -m pip install -e ".[all]"
```

## Benchmark Acquisition

Inspect the benchmark acquisition configuration:

```bash
python scripts/acquire_benchmarks.py --dest data/benchmarks
```

Execute benchmark acquisition:

```bash
python scripts/acquire_benchmarks.py --dest data/benchmarks --execute
```

For historical benchmark snapshots, provide verified Git revisions using the supported `--bugsinpy-ref` and `--bears-ref` arguments.

## Data Preparation

The framework provides dataset normalisation, label verification, and benchmark-specific collection utilities.

### Defects4J

```bash
python -m bugprioritize collect --dataset defects4j --project Lang --bug 1 --workspace data/raw/defects4j --out data/raw/lang_1.jsonl --repetitions 3
```

### BugsInPy

```bash
python -m bugprioritize collect --dataset bugsinpy --project youtube-dl --bug 2 --workspace data/raw/bugsinpy --out data/raw/youtubedl_2.jsonl --repetitions 3
```

### Bears

```bash
python -m bugprioritize collect --dataset bears --bug Bears-150 --benchmark-repo data/benchmarks/bears --workspace data/raw/bears --out data/raw/bears_150.jsonl --repetitions 3
```

Combine the collected JSONL records and prepare the processed dataset using:

```bash
python -m bugprioritize build --input data/raw/combined.jsonl --out data/processed/real_rkd.jsonl --filter-log logs/filters.csv --scenario rkd
```

The framework supports two prioritisation scenarios:

- **RKD:** Retrospective known-defect prioritisation using repair-patch and fixed-version candidate-suite information.
- **PRO:** Prioritisation using information available before the repair, with repair-related information excluded according to the benchmark-specific protocol.

## Dataset Splitting

The framework supports project-wise, time-wise, and cross-benchmark evaluation protocols.

Generate project-wise splits:

```bash
python -m bugprioritize make-splits --data data/processed/real_rkd.jsonl --out splits/real --protocol projectwise
```

The splitting procedure maintains separation between training, validation, and testing data and supports leakage-controlled evaluation.

## Model Training

BugPrioritizeAI++ supports CodeBERT-based representation learning, multimodal fusion, historical retrieval, and failure-risk prediction.

### Full Model Configuration

The main configuration includes:

| Parameter | Value |
|---|---:|
| Encoder | CodeBERT |
| Hidden dimension | 256 |
| Retrieval top-k | 10 |
| Training epochs | 50 |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Dropout | 0.20 |
| Positive BCE weight | 5.0 |
| Ranking margin | 1.0 |
| Ranking loss weight | 1.0 |
| Cost loss weight | 0.5 |
| L2 regularisation | 0.00001 |
| Random seeds | 5 |

### Training Command

```bash
python -m bugprioritize train ^
  --data data/processed/real_rkd.jsonl ^
  --split splits/real/fold_1.json ^
  --out results/bugprioritize/seed11/fold1 ^
  --encoder codebert ^
  --checkpoint microsoft/codebert-base ^
  --hidden 256 ^
  --topk 10 ^
  --epochs 50 ^
  --patience 8 ^
  --batch-size 32 ^
  --lr 0.0001 ^
  --dropout 0.20 ^
  --ranking-weight 1.0 ^
  --cost-weight 0.5 ^
  --l2-weight 0.00001 ^
  --positive-weight 5.0 ^
  --margin 1.0 ^
  --seed 11 ^
  --device auto ^
  --faiss
```

For Linux/macOS, replace the Windows `^` line continuation characters with `\`.

CodeBERT requires the Transformers package and pretrained model weights. The alternative hash encoder is also available for offline pipeline execution.

## Test-Case Ranking

The trained model can rank a new candidate test suite.

```bash
python -m bugprioritize rank ^
  --checkpoint results/bugprioritize/model.pt ^
  --input data/processed/real_rkd.jsonl ^
  --uid "Defects4J::project_00::1" ^
  --out rankings/example.csv ^
  --train-data data/processed/real_rkd.jsonl ^
  --split splits/real/fold_1.json
```

The training data and split must match the checksums stored in the checkpoint to reconstruct the historical retrieval memory.

The output contains prioritised test cases and their corresponding prediction scores.

## Baseline Comparisons

The framework includes multiple baseline and comparison methods.

| Method | Description |
|---|---|
| Random | Random test-case ordering |
| Original | Original test-suite ordering |
| History | Historical failure-based prioritisation |
| Cost | Execution-cost-based prioritisation |
| BM25 | Information retrieval-based ranking |
| LambdaMART | Learning-to-rank approach |
| GRU | Recurrent historical failure modelling |
| RETECS-style | Policy-gradient-based prioritisation |
| MM-LambdaMART | Multimodal learning-to-rank approach |
| MM-Concat-MLP | Multimodal concatenation-based model |
| BugPrioritizeAI | Attention-based fusion comparison |
| BugPrioritizeAI++ | Complete proposed framework |

Example baseline execution:

```bash
python -m bugprioritize baseline --data data/processed/real_rkd.jsonl --split splits/real/fold_1.json --out results/bm25 --method bm25

python -m bugprioritize baseline --data data/processed/real_rkd.jsonl --split splits/real/fold_1.json --out results/lambdamart --method lambdamart

python -m bugprioritize baseline --data data/processed/real_rkd.jsonl --split splits/real/fold_1.json --out results/gru --method gru --epochs 10

python -m bugprioritize baseline --data data/processed/real_rkd.jsonl --split splits/real/fold_1.json --out results/retecs --method retecs --epochs 10
```

## Ablation Studies

Ablation studies evaluate the contribution of individual framework components.

The implemented configurations include:

- Bug-report representations.
- Direct and attention-based bug features.
- Code-diff representations.
- Direct and attention-based diff features.
- Test metadata representations.
- Historical failure features.
- Historical retrieval.
- Gated fusion.
- Cost-aware scoring.

Execute the ablation pipeline:

```bash
python -m bugprioritize ablate --data data/processed/real_rkd.jsonl --split splits/real/fold_1.json --out results/ablations --epochs 2 --hidden 24 --vocab 256
```

## Evaluation Metrics

The framework evaluates test-case prioritisation using the following metrics:

| Metric | Purpose |
|---|---|
| APFD | Measures the rate of fault detection |
| APFDc | Measures cost-aware fault detection |
| MRR | Measures the position of the first relevant result |
| Recall@5 | Measures relevant test coverage in the top five |
| Recall@10 | Measures relevant test coverage in the top ten |
| NDCG@5 | Measures ranking quality in the top five |
| NDCG@10 | Measures ranking quality in the top ten |
| Top@5 | Evaluates top-five prioritisation |
| Top@10 | Evaluates top-ten prioritisation |

The evaluation module also supports:

- Per-bug performance calculations.
- Macro-averaged evaluation.
- Paired Wilcoxon statistical testing.
- Holm multiple-comparison correction.
- Rank-biserial effect sizes.
- Project-level bootstrap analysis.
- Execution latency measurement.
- Result visualisation.

### Statistical Analysis

```bash
python -m bugprioritize stats --input results --reference full --out results/stats
```

### Execution Latency

```bash
python scripts/benchmark_latency.py --checkpoint results/bugprioritize/model.pt --data data/processed/real_rkd.jsonl --out results/latency --sizes 10,100,1000
```

## Repository Structure

```text
BugPrioritizeAI-PlusPlus/
│
├── configs/
│   ├── offline_smoke.yaml
│   └── paper.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmarks/
│
├── scripts/
│   ├── acquire_benchmarks.py
│   ├── benchmark_latency.py
│   ├── reproduce.py
│   ├── smoke_test.sh
│   └── smoke_test.ps1
│
├── splits/
│
├── src/
│   └── bugprioritize/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── demo.py
│       ├── train.py
│       ├── utils.py
│       │
│       ├── datasets/
│       │   ├── collectors.py
│       │   ├── schema.py
│       │   ├── splits.py
│       │   └── features.py
│       │
│       ├── models/
│       │   ├── network.py
│       │   ├── memory.py
│       │   └── loss.py
│       │
│       ├── baselines/
│       │   └── run.py
│       │
│       └── evaluation/
│           ├── metrics.py
│           ├── summary.py
│           ├── stats.py
│           └── plots.py
│
├── tests/
│   ├── test_core.py
│   └── test_pipeline.py
│
└── requirements.txt
```

## Testing

Run the automated test suite:

```bash
python -m pytest -q
```

For the complete offline pipeline check:

**Windows PowerShell:**

```powershell
.\scripts\smoke_test.ps1
```

**Linux/macOS:**

```bash
bash scripts/smoke_test.sh
```

## Reproducibility

For consistent experimental evaluation, retain:

- Benchmark versions and full Git commit identifiers.
- Candidate test inventories.
- Dataset preprocessing and exclusion logs.
- Bug-report and repair timestamps.
- Training, validation, and testing split files.
- Model configurations and checkpoints.
- Random seeds and hardware specifications.
- Per-bug ranking outputs.
- Evaluation metrics and statistical results.


## Dependencies

The project uses the following technologies and libraries:

- Python
- PyTorch
- CodeBERT
- Hugging Face Transformers
- FAISS (optional)
- LightGBM
- Optuna
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Pytest

Refer to `requirements.txt` for dependency specifications.


## Repository

**BugPrioritizeAI++**

Developed as a multimodal framework for intelligent, retrieval-augmented, and cost-aware regression test-case prioritisation.

