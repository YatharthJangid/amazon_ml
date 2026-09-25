# Business Entity Resolution Pipeline

## Overview
High-performance entity resolution pipeline matching query entities from Source 1 against target candidates from Source 2 and Source 3.

## Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── normalize.py       # NFKC, legal suffix stripping, address abbreviations
│   ├── extractors.py      # Regex postal code, street numbers, script detection
│   ├── blocking.py        # High-recall multi-key candidate blocking engine
│   ├── features.py        # Pairwise string similarity, Jaccard, script metrics
│   ├── evaluate.py        # Exact Macro F0.5 evaluator with breakdown reporting
│   ├── validate_local.py  # Local format & consistency validator
│   └── run_all.py         # End-to-end execution script
└── README.md
```

## Running End-to-End Pipeline
```bash
# Run on synthetic test data
python code/business_entity_resolution/src/run_all.py \
    --data_dir tests/data \
    --output_dir output \
    --split test

# Run and evaluate on train data
python code/business_entity_resolution/src/run_all.py \
    --data_dir tests/data \
    --output_dir output \
    --split train
```

## Submission Integrity Checks
```bash
python code/business_entity_resolution/src/validate_local.py \
    --matching output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv \
    --test_s1 tests/data/test_source1.tsv \
    --test_s2 tests/data/test_source2.tsv \
    --test_s3 tests/data/test_source3.tsv
```
