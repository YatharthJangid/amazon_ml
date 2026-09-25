# 🤝 Teammate Onboarding & Technical Decision Log

Welcome to the **Business Entity Resolution** pipeline. This document is the single source of truth for all 4 team members. Read this before touching any code.

---

## 1. Quick Onboarding (Get Running in 3 Minutes)

### Local Environment Setup
```bash
# 1. Clone & enter repository
git clone https://github.com/YatharthJangid/amazon_ml.git
cd amazon_ml

# 2. Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify GPU (if local NVIDIA card available)
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 5. Run sanity check tests (must pass 5/5)
pytest tests/test_metric.py -v

# 6. Run end-to-end smoke test on synthetic data
python code/business_entity_resolution/src/run_all.py --split train
```

### Kaggle Environment Setup (For Parallel GPU Compute)
If you are running experiments on Kaggle (e.g., Teammate C training mDeBERTa):
1. Create a Kaggle Notebook with GPU enabled (T4 × 2).
2. Go to **Add-ons** $\to$ **Secrets** $\to$ Add `GH_PAT` (read-only GitHub Personal Access Token).
3. First cell:
   ```python
   from kaggle_secrets import UserSecretsClient
   import sys
   token = UserSecretsClient().get_secret("GH_PAT")
   !rm -rf amazon_ml && git clone https://{token}@github.com/YatharthJangid/amazon_ml.git
   %cd amazon_ml
   !pip install -q polars rapidfuzz jellyfish datasketch unidecode wordfreq sentencepiece protobuf tiktoken
   sys.path.insert(0, "/kaggle/working/amazon_ml/code/business_entity_resolution/src")
   ```

---

## 2. Division of Labor & Team Ownership

| Teammate | Focus Area | Primary Files | Key Deliverable |
|---|---|---|---|
| **Yatharth (A)** | Candidate Blocking & Pipeline Integration | `blocking.py`, `run_all.py` | High-recall ($\ge 99\%$) candidate generator (`candidate_pairs.tsv`) |
| **Teammate B** | Features & GBDT Modeling | `features.py`, `train_lgbm.py`, `scoring.py` | LightGBM model + hard negative mining + feature engineering |
| **Teammate C** | Deep Multilingual Models & Ensembling | `train_deberta.py`, `ensemble.py` | Fine-tuned mDeBERTa-v3 cross-encoder + MiniLM embeddings |
| **Teammate D** | EDA, Local Validation & Submission Discipline | `eda.py`, `validate_local.py`, docs | Submission TSV validation + 5/day submission management |

---

## 3. Architecture & File Walkthrough

All source code lives in [`code/business_entity_resolution/src/`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/):

| File | Purpose | Key Nuances to Know |
|---|---|---|
| [`normalize.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/normalize.py) | Text cleaning & canonicalization | Uses NFKC, removes legal suffixes (`pvt ltd`, `corp`, `llc`), expands street abbreviations (`rd` $\to$ `road`), and sorts tokens for word-order invariance. |
| [`extractors.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/extractors.py) | Regex-only extractors | Strict rule compliance: **NO libpostal, NO usaddress**. Regex for US ZIP (5-digit), India PIN (6-digit), and French fallback. Script profiler detects Devanagari vs Latin extended/accented vs ASCII. |
| [`blocking.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/blocking.py) | Candidate pair generator | Combines exact normalized names, sorted token signatures, address grouping, composite postal+street keys, and a `CharNgramIndex` (char_wb TF-IDF cosine top-$K$) for typos. Uses cheap similarity-ranked truncation to prevent dropping true matches. |
| [`features.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/features.py) | Pairwise feature extraction | RapidFuzz metrics (Levenshtein norm, Jaro-Winkler, token sort/set), character bigram cosine, token containment, exact normalized country match, and postal prefix match. |
| [`scoring.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/scoring.py) | Signal weighting & threshold tuner | Baseline weighted scoring rule across name, address, postal, and country signals + grid search tuner for Macro $F_{0.5}$. |
| [`evaluate.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/evaluate.py) | Exact competition metric | Macro $F_{0.5}$ evaluator with singleton handling ($1.0$ if empty match predicted correctly, $0.0$ if missed). Verified against official PDF worked example ($0.714$). |
| [`validate_local.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/validate_local.py) | Submission format gatekeeper | Verifies 2-column TSVs, correct headers, all test S1 IDs present with 0 duplicates, correct `S2-`/`S3-` prefixes, and subset integrity ($matching \subseteq candidates$). |
| [`eda.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/eda.py) | 15-second data profiler | Analyzes raw datasets immediately: row counts, singleton %, match histograms, many-to-many conflicts, and script distributions. |
| [`train_lgbm.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/train_lgbm.py) | GBDT training pipeline | Uses **`GroupKFold` grouped by S1 entity ID** (prevents entity data leakage across folds), hard negative mining from blocking candidates, and out-of-fold threshold optimization. |
| [`train_deberta.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/train_deberta.py) | Multilingual Cross-Encoder | Sequence-pair classification head on `microsoft/mdeberta-v3-base` with mixed precision (`fp16`). |
| [`ensemble.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/ensemble.py) | Model blender | Blends LightGBM + mDeBERTa-v3 + heuristic similarities, then sweeps optimal decision thresholds. |
| [`download_weights.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/download_weights.py) | Offline weight manager | Caches models locally in `weights/` for 100% offline compliance. |
| [`run_all.py`](file:///home/jangi/projects/amazon/code/business_entity_resolution/src/run_all.py) | Full pipeline orchestrator | Loads data, runs blocking, extracts features, scores candidates, writes TSVs, runs format validation, and evaluates against ground truth. |

---

## 4. Key Architectural & Strategic Decisions Log

### Decision 1: Metric Optimization — Precision Over Recall
- **Why:** The evaluation metric is **Macro $F_{0.5}$** ($\beta = 0.5$). Precision is mathematically weighted **twice as heavily** as recall:
  $$F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
- **Impact:** False positives hurt our score much more than false negatives. Never output low-confidence matches. Our decision thresholds $\tau$ are tuned higher (typically $0.55 - 0.75$).

### Decision 2: Singleton Rule Handling
- **Why:** In ground truth, many S1 entities are singletons (0 matches in S2/S3).
- **Metric rule:** If true match is empty, predicting empty gives **1.0**; predicting even a single false match gives **0.0**.
- **Impact:** Correctly predicting empty for singletons is a massive leaderboard booster. A dedicated singleton filter / high-threshold barrier is essential.

### Decision 3: No Winner-Takes-All (Keep All Valid Matches)
- **Why:** Challenge rules explicitly state: *"Include every S2/S3 ID believed to be the same business — do not truncate."*
- **Impact:** If S1 matches both S2-00047 and S3-00812, include **both** in the comma-separated list. Do not greedily select only the single highest-scoring match.

### Decision 4: Rule Compliance on External Data & Libraries
- **Banned:** `libpostal`, `usaddress`, external geocoders, and gazetteer databases are strictly banned.
- **Allowed:** Hand-written dictionaries (`LEGAL_SUFFIXES`, `ADDR_ABBREV`), regex-based extractors, `fastText`, `wordfreq`, transductive statistics over train+test, and self-training.
- **Impact:** All parsing logic is 100% regex-based and hand-crafted to guarantee zero audit disqualification.

### Decision 5: Leakage Prevention via GroupKFold on S1
- **Why:** Standard random K-Fold splits candidate pairs randomly, meaning an S1 entity would appear in both train and validation sets, causing catastrophic data leakage.
- **Decision:** All cross-validation **must** use `GroupKFold(groups=s1_id)` so all pairs for a given S1 entity are strictly confined to either train or validation.

### Decision 6: Local-First Hardware Split
- **Local Machine (RTX 4060 8GB VRAM + 50GB NVMe):** Serves as the primary hub for data processing, candidate blocking, feature extraction, and LightGBM sweeps. No session timeouts, fast I/O.
- **Kaggle (2×T4 / P100):** Used as an auxiliary GPU worker for Teammate C to train mDeBERTa cross-encoders in parallel without tying up the local machine.

### Decision 7: Offline Execution & Pre-Cached Weights
- **Why:** Competition submission environments are offline.
- **Decision:** Both pre-approved models (`microsoft/mdeberta-v3-base` [MIT] and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` [Apache-2.0]) are downloaded to `weights/`. Code loads via `local_files_only=True`.

### Decision 8: Official Submission TSV Headers
- **Format:**
  - `candidate_pairs.tsv`: `source1_entity_id\tcandidate_entity_ids`
  - `matching_results.tsv`: `source1_entity_id\tmatched_entity_ids`
- Both files must have exactly 2 tab-separated columns, all test S1 IDs must appear, and `matching_results ⊆ candidate_pairs`.

### Decision 9: Offline Preprocessing & Parquet Caching (`preprocess.py`)
- **Why:** In raw TSVs, normalizing strings and evaluating regexes on-the-fly inside the 66M candidate pair loop caused severe lag and thrashing.
- **Decision:** Shift compute from $O(N \times C)$ to $O(N)$. Precompute canonical features (`norm_name`, `norm_addr`, `sort_name`, `sort_addr`, `norm_name_tokens`, `norm_addr_tokens`, `postal`, `postal_3`, `st_num`, `acronym`, `script_indic`, `script_ascii`) once per record via Polars + `multiprocessing` across CPU threads, and stream directly to compressed Parquet files (`dataset/cache/*.parquet`) with PyArrow.
- **Impact:** Peak RAM during preprocessing stays strictly under 3 GB, and Parquet loading takes < 2 seconds.

### Decision 10: Sparse Row Iteration in Blocking (Zero-OOM)
- **Why:** Running `.toarray()` on a $(1024 \times 6,000,000)$ TF-IDF similarity matrix created a 24.6 GB dense NumPy array per batch, causing kernel memory crashes on 10 GB RAM machines.
- **Decision:** In `blocking.py`, `CharNgramIndex.query()` iterates sparse rows directly using `sim_batch.getrow(row_idx)` and sparse array slicing. No dense `.toarray()` allocations ever occur.

---

## 5. Daily Submission Discipline (Max 5/Day per Team)

We have a team limit of **5 submissions per calendar day**. Teammate D owns the submission button. 

Every single submission must be logged in our shared tracking sheet before hitting submit:

| Column | Description |
|---|---|
| **Sub #** | Sequential ID (`01`, `02`, ...) |
| **Date & Time** | Timestamp of submission |
| **Author** | Teammate (A / B / C / D) |
| **Git Commit** | Exact commit hash on `main` (e.g., `7cdbbc1`) |
| **Model / Pipeline** | Brief description of changes |
| **Blocking Recall (CV)** | Candidate blocking recall on local validation |
| **Local CV $F_{0.5}$** | Out-of-fold Macro $F_{0.5}$ score |
| **Public LB $F_{0.5}$** | Score returned by leaderboard |
| **$\Delta$ (LB - CV)** | Difference to detect overfitting or format issues |
| **Decision / Notes** | Keep, discard, or tune threshold |

---

## 6. What to Do When the Dataset Drops

1. **Place raw files into `dataset/`:**
   ```
   dataset/
   ├── train_source1.tsv
   ├── train_source2.tsv
   ├── train_source3.tsv
   ├── train_ground_truth.tsv
   ├── test_source1.tsv
   ├── test_source2.tsv
   └── test_source3.tsv
   ```
2. **Run the 15-second EDA profiler:**
   ```bash
   python code/business_entity_resolution/src/eda.py --data_dir dataset
   ```
3. **Run Candidate Blocking & LightGBM Baseline:**
   ```bash
   python code/business_entity_resolution/src/train_lgbm.py --data_dir dataset --weights_dir weights
   ```
4. **Generate and Validate First Submission:**
   ```bash
   python code/business_entity_resolution/src/run_all.py --data_dir dataset --output_dir output --split test
   ```
5. **Verify Submission Files:**
   ```bash
   python code/business_entity_resolution/src/validate_local.py \
       --matching output/matching_results.tsv \
       --candidates output/candidate_pairs.tsv \
       --test_s1 dataset/test_source1.tsv \
       --test_s2 dataset/test_source2.tsv \
       --test_s3 dataset/test_source3.tsv
   ```
6. **Submit Sub #1 (Baseline) to confirm zero format errors.**
