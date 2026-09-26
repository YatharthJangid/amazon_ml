# v2 pipeline — scalable blocking + LightGBM matcher

Runs end-to-end on a normal laptop CPU (tested in a 2-core / 7 GB sandbox; 16 GB RAM recommended).
No GPU is needed.

## Run

```bash
pip install -r ../../requirements_v2.txt
cd code/business_entity_resolution/src/v2
python pipeline.py --data "<path>/student_resource/dataset" --work work --out ../../../../output all
python ../../../../utils/validate_submission.py --matching ../../../../output/matching_results.tsv \
       --candidate ../../../../output/candidate_pairs.tsv --test-dir "<path>/student_resource/dataset/test"
```

**Full-data run on a 32 GB machine** (recommended for the final model; see `colab_full_training.ipynb` and `REPORT_v3.md` section 5):
```bash
python pipeline.py --data "<path>/dataset" --work work --out output --train_s1 0 --folds 0 all
```
`--folds 0` skips CV and uses threshold 0.70 and 800 rounds. `--folds 3` runs GroupKFold CV, prints the OOF macro F0.5 and re-tunes the threshold.

The stages can be run one at a time (`prep`, `block`, `features`, `train`, `predict`). Each stage caches its output in `work/`, so a crash
resumes where it stopped. Delete `work/cand_*` / `work/feat_*` to force a rerun.
`--train_s1 N` sets how many S1 entities per country are used to train (default 300k; use 0 for all of them on a big machine).

## Stages and timings (2-core sandbox)

| stage | what | time |
|---|---|---|
| prep | TSV -> normalised tokens -> hashed token lists + state bitmasks | ~6 min |
| block | per (country, state) partition: sparse token index, forward top-10 + reverse top-1 | India ~13 min, US ~9 min, France ~5 min (test) |
| features | 53 pair features via rapidfuzz `cpdist` (C++) | ~4 min for 18.6M test pairs |
| train | LightGBM, GroupKFold(S1), threshold search on OOF macro F0.5 | ~10 min |
| predict | score + one-S1-per-target assignment + write TSVs | ~15 min |

## Files

- `norm.py`: Polars-vectorised normalisation. Handles accents, `&`, ordinals, leading zeros, US/India state names ↔ codes ↔ native script, and street abbreviations. The Indic→Latin token dictionary is **learned from the training pairs** (`translit.json`), not looked up externally.
- `block.py`: `block_country` / `merge_parts` (per-state partitions) and `MultiBlocker`, the candidate generator (see decisions 11–16).
- `pair_features.py`: pair features (`FEATURES` list).
- `postproc.py`: decision rules (threshold + one-S1-per-target assignment; an expected-F0.5 variant that gave no gain).
- `pipeline.py`: the reproducible end-to-end driver.
- `cv_train.py`: the experiment script used for per-country CV numbers.
- `colab_full_training.ipynb`: the full-data training runbook as a notebook.
