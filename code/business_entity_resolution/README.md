# Business Entity Resolution Pipeline

## Overview
High-performance entity resolution pipeline matching query entities from Source 1 against target candidates from Source 2 and Source 3.

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── eda.py             # Rapid data profiling (answers 6 key pre-modeling questions)
│   ├── normalize.py       # NFKC, legal suffix stripping, address abbreviations
│   ├── extractors.py      # Regex postal code, street numbers, script detection
│   ├── blocking.py        # High-recall multi-key candidate blocking engine
│   ├── features.py        # Pairwise string similarity, Jaccard, script metrics
│   ├── train_lgbm.py      # GroupKFold GBDT classifier with Macro F0.5 threshold sweep
│   ├── train_deberta.py   # Multilingual mDeBERTa-v3 cross-encoder (local RTX 4060)
│   ├── ensemble.py        # Blending engine & optimal decision threshold search
│   ├── evaluate.py        # Exact Macro F0.5 evaluator with breakdown reporting
│   ├── validate_local.py  # Local format & consistency validator
│   └── run_all.py         # End-to-end execution orchestrator
└── README.md
```

## Quick Start (Run Locally)

### 1. Run Rapid EDA on New Data
```bash
python code/business_entity_resolution/src/eda.py --data_dir tests/data
```

### 2. Run Metric Unit Tests
```bash
pytest tests/test_metric.py -v
```

### 3. Run End-to-End Baseline Pipeline
```bash
# Run on synthetic test split
python code/business_entity_resolution/src/run_all.py \
    --data_dir tests/data \
    --output_dir output \
    --split test

# Run and evaluate on train split
python code/business_entity_resolution/src/run_all.py \
    --data_dir tests/data \
    --output_dir output \
    --split train
```

### 4. Train LightGBM Model
```bash
python code/business_entity_resolution/src/train_lgbm.py \
    --data_dir tests/data \
    --weights_dir weights
```

### 5. Validate Submission TSVs
```bash
python code/business_entity_resolution/src/validate_local.py \
    --matching output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv \
    --test_s1 tests/data/test_source1.tsv \
    --test_s2 tests/data/test_source2.tsv \
    --test_s3 tests/data/test_source3.tsv
```
