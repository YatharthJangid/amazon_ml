# Amazon ML Challenge 2026 - Business Entity Resolution Pipeline

High-performance, modular, and competition-compliant pipeline for resolving business entities across heterogeneous sources (Source 1 reference queries matched against Source 2 and Source 3 candidate catalogs).

**Target Metric:** Macro $F_{0.5}$ (Precision weighted $2\times$ over recall). Singletons score 1.0 if empty predicted, 0.0 if any false match is predicted.

---

## 🚀 Quickstart for Teammates

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verified dependencies: `polars`, `pyarrow`, `rapidfuzz`, `scikit-learn`, `lightgbm`, `torch` (CUDA supported).

### 2. Offline Dataset Preprocessing & Parquet Caching
To eliminate the $O(N \times C)$ lag from normalizing strings 66 million times in nested loops, we precompute canonical features once per record and cache them into compressed Parquet files (`dataset/cache/*.parquet`):

```bash
.venv/bin/python code/business_entity_resolution/src/preprocess.py
```
* **Performance:** Normalizes ~24.4 million records across 12 CPU threads and streams directly to Parquet via PyArrow with peak RAM strictly bounded under 3 GB.
* **Precomputed Fields:** `norm_name`, `norm_addr`, `sort_name`, `sort_addr`, `norm_name_tokens`, `norm_addr_tokens`, `postal`, `postal_3`, `st_num`, `first_tok`, `acronym`, `script_indic`, `script_ascii`, `script_type`.
* **Current Cache State:**
  * ✅ `train_source1.parquet` (2.2M rows, 443.5 MB)
  * ✅ `train_source2.parquet` (5.0M rows, 1054.0 MB)
  * ✅ `train_source3.parquet` (5.3M rows, 1074.9 MB)
  * ✅ `test_source1.parquet` (1.7M rows, 360.8 MB)
  * ⏳ `test_source2/3.parquet` (run `preprocess.py` to complete the test split cache).

### 3. Run Candidate Blocking & Feature Extraction
```bash
.venv/bin/python code/business_entity_resolution/src/run_all.py \
    --data_dir dataset/train \
    --output_dir output/train_prod \
    --split train \
    --threshold 0.50 \
    --max_candidates 30 \
    --max_workers 8
```

### 4. Train LightGBM Model (Teammate B)
```bash
.venv/bin/python code/business_entity_resolution/src/train_lgbm.py \
    --data_dir dataset/train \
    --features_path output/train_prod/features.parquet \
    --output_dir output/lgbm_model
```

### 5. Validate Submission Files
```bash
.venv/bin/python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidates output/candidate_pairs.tsv \
    --test_s1 dataset/test/test_source1.tsv \
    --test_s2 dataset/test/test_source2.tsv \
    --test_s3 dataset/test/test_source3.tsv
```

---

## 🏛️ Core Architecture & Key Decisions

1. **Zero Cross-Country Matches:** 
   Our EDA over 26.4M records proved mathematically that 100% of matches occur within the same country (`US`, `IN`, `FR`). The pipeline strictly partitions by country, reducing candidate space by ~65% and preventing cross-border false positives.
2. **Memory-Safe Sparse Blocking:**
   In `blocking.py`, `CharNgramIndex` uses sparse row iteration (`sim_batch.getrow(row_idx)`) instead of dense `.toarray()`, eliminating the 24.6 GB RAM explosion on 6M candidate records.
3. **Precision-First Thresholding for $F_{0.5}$:**
   Singletons (empty ground truth) represent 5.58% of entities. Predicting a false match drops singleton score from 1.0 directly to 0.0. All decision thresholds are tuned aggressively to favor precision over recall.
4. **Leakage-Free Validation:**
   All cross-validation strictly enforces `GroupKFold(groups='entity_id')` on S1 query IDs so that pairs for a given entity are never split across train and test folds.

---

## 📁 Repository Structure
```
amazon_ml/
├── README.md                           # Quickstart & architectural documentation
├── TEAMMATES.md                        # Team workflow, ownership, and decision log
├── COMPETITION_README.md               # Full competition technical specification
├── requirements.txt                    # Python environment requirements
├── dataset/
│   ├── train/                          # Raw train TSVs & ground truth
│   ├── test/                           # Raw test TSVs
│   └── cache/                          # Preprocessed Parquet cache files
├── weights/                            # Offline pre-cached transformer weights
├── utils/
│   └── validate_submission.py          # Official submission validator
└── code/business_entity_resolution/src/
    ├── normalize.py                    # NFKC, legal suffix, & address cleaning
    ├── extractors.py                   # Postal & street regexes, script profile
    ├── blocking.py                     # Multi-key inverted indices & sparse CharNgramIndex
    ├── preprocess.py                   # High-speed offline Parquet preprocessor
    ├── features.py                     # RapidFuzz C++ pairwise similarity features
    ├── scoring.py                      # Multi-signal scoring & threshold tuning
    ├── evaluate.py                     # Spec-exact Macro F0.5 evaluator
    ├── run_all.py                      # Multithreaded orchestrator & parquet exporter
    ├── train_lgbm.py                   # GroupKFold LightGBM baseline
    └── validate_local.py               # Pre-submission integrity validator
```
