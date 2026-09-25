# Team Entity Resolution — Solution Documentation & Audit

## 1. Executive Summary & Architecture Overview
- **Task:** Business Entity Resolution across three disparate sources (S1 query against S2/S3 candidate catalog).
- **Core Pipeline:**
  1. **Preprocessing & Normalization:** Unicode NFKC, legal suffix stripping, address expansion, regex postal code and script detection.
  2. **Candidate Blocking:** Inverted token index + TF-IDF char n-gram candidate generation (Target ≥99% recall).
  3. **Feature Engineering:** String similarity metrics (Levenshtein, Jaro-Winkler, Token Sort/Set Jaccard), script consistency, postal code matching, fastText / multilingual embedding cosine similarity.
  4. **Scoring & Classification:** LightGBM binary classifier + mDeBERTa-v3 cross-encoder fine-tuning.
  5. **Post-Processing & Thresholding:** Macro F0.5 optimization with full inclusion of valid duplicate/multi-matches without premature truncation.

## 2. Team Division of Labor
| Team Member | Stream / Ownership | Deliverable |
|---|---|---|
| **Yatharth (A)** | Candidate Generation & Blocking Engine, Pipeline Integration | `blocking.py`, `candidate_pairs.tsv` generator (≥99% recall), `run_all.py` |
| **Teammate B** | Feature Suite, Hard Negative Sampling, GBDT Modeling | `features.py`, `train_lgbm.py`, threshold optimization |
| **Teammate C** | Deep Multilingual Models & Ensembling | `train_deberta.py`, `ensemble.py`, embedding extraction |
| **Teammate D** | Data Profiling, Local Validation, Ops & Submission Discipline | `validate_local.py`, EDA report, submission logging |

## 3. License Audit & Model Cards
| Asset | Source / Identifier | License | Purpose |
|---|---|---|---|
| `mDeBERTa-v3-base` | `microsoft/mdeberta-v3-base` | MIT | Multilingual cross-encoder re-ranker |
| `paraphrase-multilingual-MiniLM-L12-v2` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Apache-2.0 | Dense multilingual sentence embeddings |
| `rapidfuzz` | `maxbachmann/rapidfuzz` | MIT | High-performance C++ string matching |
| `jellyfish` | `jamesturk/jellyfish` | BSD-2-Clause | Phonetic and phonetic-adjacent distance |
| `wordfreq` | `rspeer/wordfreq` | MIT | Corpus-based unigram frequency filtering |

## 4. Offline Execution & Reproducibility Guarantee
- The entire inference pipeline (`run_all.py`) is designed to run in an isolated environment with `HF_HUB_OFFLINE=1`.
- All model weights are cached locally inside the `weights/` directory.
- No network requests, geocoders, or banned external gazetteers are utilized.
