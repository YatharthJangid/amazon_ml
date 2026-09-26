# Amazon ML Challenge 2026: status report and training runbook (v3)

*Team working notes, 26 Sep 2026, 20:30 IST. About 27.5 h to the deadline (27 Sep, 23:59 IST). The 48 h / Top-500 credit cut-off is 27 Sep 00:00 IST.*

---

## 1. Where we stand in one paragraph

We replaced the original blocking, which could not finish at this data size, with a scalable pipeline:
- per-state sparse-token blocking (forward top-10 plus reverse top-1)
- 53 vectorised pair features
- a LightGBM matcher
- a "one S1 per target" assignment step

On the training data (3-fold GroupKFold by S1) it scores **macro F0.5 ≈ 0.9745** (US 0.979, India 0.970). It keeps **98.4 % (US) / 95.9 % (India)** of true matches in the candidate set at **~10.3 candidates per S1**. Two validated submissions exist (`sub01`, `sub02`); both pass the official validator with `--check-ids`.

The public-leaderboard top 5 are at **0.989–0.991**. On CV we are ~1.5 points behind, but we have **not yet seen our own leaderboard score**, and that decides how big the real gap is.

---

## 2. What the problem asks and what we have

| Requirement (problem statement / emails) | Status |
|---|---|
| `matching_results.tsv`: one row per test S1, matched S2/S3 IDs | Done (sub01, sub02), validator PASS incl. `--check-ids` |
| `candidate_pairs.tsv`: the exact set the model scores | Done, written by the same pipeline run (matches ⊆ candidates) |
| Blocking must scale and use a **small candidate set per S1** (ranked higher in final evaluation) | ~10.3 / 10.7 / 11.1 cands per S1 (US / India / France). Linear-time sparse-token index, no N×M comparisons |
| France (unseen in train) must be handled; treat country as an open set | Pipeline loops over whatever countries appear. No country feature in the model, so France uses the same language-agnostic signals |
| No external data / APIs / geocoders | Only hand-written dictionaries plus a token dictionary **learned from the training pairs** (Indic script → Latin). No libpostal, no lookups |
| Model licence MIT/Apache, ≤ 8B params | LightGBM (MIT). No pretrained model used |
| Runnable code + README + requirements + methodology doc in the final zip | Code and README in `src/v2/`. **The methodology doc (`Documentation_template.md`) and final zip layout are still to do** (section 7) |

---

## 3. How the pipeline works

```
TSV --prep--> normalised + hashed tokens --block--> ~10 candidates / S1 --features--> 53 features
    --train--> LightGBM (GroupKFold by S1) --predict--> assignment (each target -> ≤ 1 S1, p ≥ 0.70) --> 2 TSVs
```

1. **Normalisation (`norm.py`, Polars-vectorised).**
   - Accents, `&`, ordinals (33nd → 33) and leading zeros (0048 → 48).
   - Street abbreviations (Rd/Dr/Ct …).
   - A **state is detected only when a whole comma-separated part of the address is a state name, a state code, or a native-script state name**. So "306 Massachusetts Avenue, …, OK" is Oklahoma, not Massachusetts.
   - Indic-script tokens are mapped to Latin with a 1,328-entry dictionary learned from aligned training pairs (रियल → real, प्राइवेट → private).
2. **Blocking (`block.py`).** Each record becomes hashed tokens in two channels:
   - **name**: legal-stripped words, the glued name, and char 4-grams (for typos, `greenimpex.com`, `#servicesblue`)
   - **address**: tokens and adjacent bigrams (`1795_westchester`)

   Blocking runs **per (country, state) partition**, so token rarity is judged locally. Targets with no detectable state (~3.5 %) join every partition.
   - Score = `cos_name + cos_addr + 0.5·min(cos_name, cos_addr)`, computed as chunked sparse matmul.
   - Candidates = **forward top-10 per S1 ∪ reverse top-1 S1 per target**. The reverse direction exploits a data fact: every S2/S3 record belongs to at most one S1.
3. **Features (`pair_features.py`).** 53 features via `rapidfuzz.process.cpdist` (C++), about 4 min for 18M test pairs:
   - name/address fuzzy ratios
   - digit-run overlap: decoys often change one house number
   - legal-form categories and conflicts: decoys often swap Private↔Public or LLP↔Ltd
   - name-ambiguity counts
   - blocking ranks and gaps
4. **Model and decision.** LightGBM with GroupKFold by S1. Each target is assigned to its single best S1 if p ≥ 0.70 (the OOF-optimal threshold).

---

## 4. Results so far

| Version | Blocking recall US / India | Cands / S1 | OOF macro F0.5 US / India / overall | Public LB |
|---|---|---|---|---|
| Yatharth's original (char-TF-IDF) | cannot finish (≈75 GB/batch) | – | – | – |
| sub01: v2 blocking | 96.6 % / 95.4 % | ~10.5 | 0.973 / 0.970 / **0.9717** | *please fill in* |
| sub02: v3 per-state blocking | **98.4 % / 95.9 %** | ~10.3 | **0.979 / 0.970 / 0.9745** | *please fill in* |

Notes:
- The OOF numbers come from a model trained on only **150k S1 per country (~14 % of train)**, because of a 7 GB sandbox. The full-data run (section 5) should be a bit better and more stable.
- The **ceiling** (a perfect matcher on our v3 candidates) is **0.990**. The matcher currently recovers 0.9745 of it.

---

## 5. Runbook: full training on the 32 GB machine

Goal: rerun the whole pipeline with **all 2.2M training S1s** (`--train_s1 0`). Expect about +0.1 to +0.3 points, and a more stable model for the **private** leaderboard.

### 5.1 What to upload

| Item | Where from | Size |
|---|---|---|
| Dataset: `6ab10eb3b23ba_student_resource.zip` (or just `student_resource/dataset/`) | `D:\amazon ML challeneg\` | 1.09 GB zip |
| Code: this PR branch `v3-scalable-pipeline` (clone), **or** `code_v3.zip` | GitHub / `D:\amazon ML challeneg\code_v3.zip` | < 1 MB |

Put the dataset zip in **Google Drive** (e.g. `MyDrive/amlc/`) so it survives runtime resets.

### 5.2 Steps (Colab / any Linux box with 32 GB RAM)

A ready notebook is in the PR: `code/business_entity_resolution/src/v2/colab_full_training.ipynb`. The equivalent commands:

```bash
# 0) (Colab) mount Drive, pick the High-RAM CPU runtime. No GPU is needed.
from google.colab import drive; drive.mount('/content/drive')

# 1) get the code
git clone -b v3-scalable-pipeline https://github.com/YatharthJangid/amazon_ml.git /content/amazon_ml
pip install -q -r /content/amazon_ml/code/business_entity_resolution/requirements_v2.txt

# 2) unpack the data onto the fast local disk
unzip -q /content/drive/MyDrive/amlc/6ab10eb3b23ba_student_resource.zip -d /content/
#    -> /content/6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/*.tsv

# 3) run everything (each stage caches to --work; rerunning the same command resumes)
cd /content/amazon_ml/code/business_entity_resolution/src/v2
python pipeline.py --data /content/6ab10eb3b23ba_student_resource/student_resource/dataset \
       --work /content/work --out /content/output --train_s1 0 --folds 0 all

# 4) validate, then copy results to Drive
python /content/amazon_ml/utils/validate_submission.py --matching /content/output/matching_results.tsv \
       --candidate /content/output/candidate_pairs.tsv \
       --test-dir /content/6ab10eb3b23ba_student_resource/student_resource/dataset/test --check-ids
cp -r /content/output /content/work/lgbm.txt /content/work/model_meta.json /content/drive/MyDrive/amlc/
```

Flags:
- `--train_s1 0` trains on **all** S1s. Use `600000` if memory gets tight.
- `--folds 0` skips cross-validation and uses the known-good settings (threshold 0.70, 800 rounds). That saves 1–2 h.
- If there is time left, `--folds 3` also prints the OOF F0.5 and re-tunes the threshold. That number goes into the documentation.

**Expected time** on an 8-vCPU High-RAM runtime:

| Stage | Time |
|---|---|
| prep | ~10 min |
| block (train + test) | ~60–90 min, mostly single-threaded |
| features | ~15 min |
| train (`--folds 0`) | ~30–60 min |
| predict | ~10 min |
| **Total** | **~2.5–3 h** |

Colab disconnects idle sessions, so keep the tab active. After a disconnect, rerun the same command: finished stages are skipped automatically, as long as `/content/work` survived. If the VM was recycled, restart from step 2.

**Memory:** peak is in `train` with `--train_s1 0`, about 23M pairs × 53 features ≈ 5 GB for the matrix plus LightGBM bins. That fits in 32 GB.

### 5.3 What to submit afterwards
The new `output/matching_results.tsv` becomes **sub03**. Log it in TEAMMATES.md: commit, CV (if `--folds 3`), public LB.

---

## 6. Blockers and risks

1. **Unknown leaderboard calibration.** We don't yet know sub01/sub02's public score, so we can't tell whether our CV (0.9745) matches the leaderboard or is pessimistic. **Action: submit sub02 before 00:00 IST** (the Top-500 credit cut-off) and record the score.
2. **India recall (95.9 %)** is the largest remaining loss. It comes from Indic-script names with sparse or garbled addresses, and null-address records whose name is shared by several S1s. Per-state partitioning helped the US much more than India.
3. **Ambiguous cases the matcher rejects.** Ceiling 0.990 vs achieved 0.9745. Most remaining misses are null-address targets with generic names, and planted decoys that differ only in legal form or house number. Two stacking experiments (context features from a first-stage model; same-name cluster support) gave < +0.001.
4. **France has no labels.** We can't measure it. The model has no country feature and France's output looks statistically normal (5.5 % empty, 3.3 matches/S1), but it's a blind spot. **Private-LB risk:** don't tune anything specifically to France.
5. **Compute.** All development so far ran in a 7 GB / 2-core sandbox, so training used 14 % of the data. The 32 GB machine removes this limit.
6. **Submission file size.** `matching_results.tsv` is ~92 MB. Upload it as-is to the portal (the split zips + `merge.py` were only for transferring it).

---

## 7. What's still to do (priority order for the remaining ~27 h)

| # | Task | Owner (suggested) | Why |
|---|---|---|---|
| 1 | **Submit sub02 before 00:00 IST** and note the score | Shaivi | Top-500 credit cut-off; calibrates CV vs LB |
| 2 | **Full-data training run** (section 5) → sub03 | Whoever has the 32 GB machine | +stability for private LB |
| 3 | India blocking: more candidates only for records with **no address or no state** (reverse top-2 for those), and more Indic transliteration coverage | Claude / Shaivi | India is the biggest loss |
| 4 | Candidate-set size: test **fwd top-8 + rev1** (US recall 98.2 % at 8.3 cands vs 98.4 % at 10.3). Adopt it if F0.5 is unchanged | Claude | Smaller sets rank higher in the final evaluation |
| 5 | Fill in `Documentation_template.md` (methodology, blocking, features, results). Draft from this report + TEAMMATES decisions 11–16 | Teammate D | Required in the final zip |
| 6 | Assemble the final zip: `output/` (both TSVs), `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md`. Update `code/business_entity_resolution/README.md` to point to `src/v2/pipeline.py` | Teammate D | Required submission package |
| 7 | Final pick: choose the submission by **CV and public LB together**. Never pick on a small public-LB gain that CV doesn't support | Team | Protects the private leaderboard |

### Guardrails for public + private leaderboards
- Tune thresholds and settings on **train CV**, not by trial-and-error on the public leaderboard.
- Keep the model **country-agnostic** (no country feature, no France-specific rules).
- Keep the edge-case handling already in place:
  - null or very short addresses
  - Indic scripts
  - domains and hashtags used as names
  - reordered address parts
  - leading zeros and ordinals
  - countries with no candidates
  - exactly one output row per S1, even with zero candidates

---

## 8. Files in this PR

- `code/business_entity_resolution/src/v2/`
  - `pipeline.py`: end-to-end driver (`prep`, `block`, `features`, `train`, `predict`, `all`)
  - `norm.py`, `block.py`, `pair_features.py`, `postproc.py`: the modules
  - `cv_train.py`: experiment script used for per-country CV
  - `README_v2.md`: how to run
  - `colab_full_training.ipynb`: the 32 GB runbook as a notebook
- `code/business_entity_resolution/requirements_v2.txt`
- `REPORT_v3.md`: this report
- `TEAMMATES.md`: decisions 11–16 and the submission log, appended
