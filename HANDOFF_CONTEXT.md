# HANDOFF: Amazon ML Challenge 2026 (Business Entity Resolution)

*Written 27 Sep 2026, ~01:15 IST, at the end of a long working session, for teammates (and any AI assistant) taking over in a new chat. Paste this whole file into the new chat as context.*

**Deadline: 27 Sep 2026, 23:59 IST.** Max **5 leaderboard submissions per day** (the counter reset at midnight). The final ranking uses the **private** leaderboard. The final zip package is also reviewed, including candidate-set size and the code.

---

## 0. TL;DR

- **Best public leaderboard score: 0.967 (sub04).** History: sub02 0.960 → sub03 0.965 → sub04 0.967. Top 5 are at 0.989–0.991; the Top-500 cut-off at the 48 h mark was ~0.975.
- The code is **our own scalable pipeline** (`src/v2/pipeline.py`):
  - per-state sparse-token blocking (~10.5 candidates/S1)
  - a LightGBM matcher on 56 features
  - "each S2/S3 record goes to at most one S1" assignment at p ≥ 0.70
- **US/India are fine; France is the weak spot.** On training CV, US ≈ 0.979 and India ≈ 0.970 (macro F0.5). Backing out France from the public score gives only ≈ 0.92. France has **no training labels**, so every France fix is judged only by the leaderboard.
- **Next highest-value steps:**
  - (a) keep hunting France-specific error patterns
  - (b) full-data training on the 32 GB Colab machine
  - (c) assemble the final zip + methodology doc
  - (d) pick the final submission by CV plus public LB, not LB alone

---

## 1. The problem (short)

- Three sources of business records (`entity_id`, `business_name`, `business_address`, `country`). Source 1 (S1) is the deduplicated reference.
- For **every test S1**, output all S2/S3 records that are the same business. The list may be empty ("singleton").
- **Metric:** macro F0.5, computed per S1 then averaged, precision-weighted:
  - A singleton scores 1.0 if the list is empty, 0.0 if anything is predicted.
  - Missing 1 of 4 true matches with perfect precision gives 0.94 for that S1.
- **Outputs** (tab-separated):
  - `matching_results.tsv` with columns `source1_entity_id`, `matched_entity_ids` (comma-separated). This is the only scored file.
  - `candidate_pairs.tsv` with columns `source1_entity_id`, `candidate_entity_ids`: the exact candidate set fed to the model. Matches must be a subset. **A smaller candidate set per S1 ranks higher in the final evaluation.**
- **Rules:**
  - no external data, APIs or geocoders
  - model licence MIT/Apache, ≤ 8B params (LightGBM is MIT)
  - test has a 3rd country, **France**, not in train
- **Final zip:**
  ```
  <team>_submission.zip
    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/{src/, README.md, requirements.txt}
    Documentation_template.md   (filled-in methodology)
  ```
- Validator: `python utils/validate_submission.py --matching ... --candidate ... --test-dir dataset/test --check-ids`

## 2. Data facts (from our EDA; they drive the design)

- Train:
  - 2,206,821 S1 (US 1.32M, India 0.88M)
  - S2 5.03M, S3 5.29M records
  - 7,638,365 true pairs
- Test:
  - 1,732,544 S1 (US 663k, India 810k, **France 259k**)
  - S2 4.89M, S3 5.08M
- **5.6 % singletons.** The average S1 has **3.5 matches** (up to 10+), so recall matters a lot.
- **Every S2/S3 record belongs to at most one S1.** Our reverse blocking and assignment step use this.
- Noise types:
  - typos
  - legal-suffix changes
  - Indic-script names (13–23 % of India target names)
  - junk or domain names ("hartmanmcevoy.com")
  - null addresses (~4 %)
  - reordered address parts
  - state name ↔ code ↔ native script
  - leading zeros
- **Planted decoys** are unmatched records that copy a real business but change:
  - the legal form (Private↔Public, LLP↔Ltd, SARL↔SAS), or
  - one house number (82/2C5 → 82/2C16), or
  - one name "type" word (France: "Leducation **Club**" → "Leducation **Ecole**").
- S1 addresses rarely have ZIP/PIN codes (7 %), so postal-code blocking is useless.
- France: 3 regions only. Hauts-de-France (Nord, Pas-de-Calais), Nouvelle-Aquitaine (Gironde), Pays de la Loire (Loire-Atlantique). S1 writes the region; S2/S3 often write the department.

## 3. Where everything is

### On Shaivi's PC: `D:\amazon ML challeneg\`
| Path | What |
|---|---|
| `6ab10eb3b23ba_student_resource.zip` / `…\student_resource\dataset\{train,test}\*.tsv` | Official data (+ `.tsv.gz` copies) |
| `submissions\sub02\`, `sub03\`, `sub04\` | Each has `matching_part1.zip` + `matching_part2.zip` + `merge.py` → run `python merge.py` to get `matching_results.tsv` |
| `submissions\sub04\candidate_part1..6.zip` + `merge_candidates.py` | → `candidate_pairs.tsv` **matching sub04**, needed for the final zip |
| `model_sub04\lgbm.txt`, `model_meta.json`, `translit.json` | The exact trained model behind sub04 (threshold 0.70) |
| `pr\v3-scalable-pipeline.bundle` + `pr\PR_DESCRIPTION.md` | The git branch with all code (2 commits), ready to push |
| `code_v3.zip` | Older code snapshot. **Use the bundle instead** |
| `HANDOFF_CONTEXT.md` | This file |

Files are split into zips only because the transfer tool had a 20 MB limit. `merge.py` reassembles them byte-exact.

### GitHub
- Upstream repo: `https://github.com/YatharthJangid/amazon_ml` (owner Yatharth). His original pipeline is in `code/business_entity_resolution/src/`. Its char-TF-IDF blocking **cannot finish at full scale** (measured ~75 GB per batch), so don't use it.
- Our work is branch **`v3-scalable-pipeline`**. It adds `src/v2/`, `requirements_v2.txt`, `REPORT_v3.md` and TEAMMATES.md decisions 11–17, and doesn't touch existing files.
- To push it (Shaivi = GitHub `shaeve`; if you get a 403, fork and push to the fork):
  ```powershell
  git clone https://github.com/YatharthJangid/amazon_ml.git amazon_ml_git; cd amazon_ml_git
  git fetch "D:\amazon ML challeneg\pr\v3-scalable-pipeline.bundle" v3-scalable-pipeline:v3-scalable-pipeline
  git push origin v3-scalable-pipeline      # then open the PR on GitHub, paste PR_DESCRIPTION.md
  ```

## 4. Pipeline (branch `v3-scalable-pipeline`, folder `code/business_entity_resolution/src/v2/`)

```
python pipeline.py --data <dataset dir with train/ test/> --work work --out output [--train_s1 N] [--folds F] all
stages: prep → block → features → train → predict   (each cached in work/, rerunning resumes)
```

| File | Role |
|---|---|
| `norm.py` | Polars-vectorised normalisation (details below) |
| `block.py` | `block_country()` per (country, state) partition + `merge_parts()` (numpy); `MultiBlocker` = two sparse-token channels |
| `pair_features.py` | 56 features via `rapidfuzz.process.cpdist` (C++) |
| `pipeline.py` | End-to-end driver. Final model is LightGBM (lr 0.1, 127 leaves, ~780 rounds), threshold 0.70 |
| `cv_train.py` | Experiment script for per-country GroupKFold CV |
| `colab_full_training.ipynb` | Runbook for the 32 GB machine |

**`norm.py` does:**
- accents, `&`, ordinals and leading zeros
- street abbreviations
- **state = only a whole comma-separated address part** (so "306 Massachusetts Avenue, OK" is Oklahoma)
- the Indic→Latin token dictionary **learned from training pairs** (`translit.json`, 1,328 tokens)
- France only: region/department → region code, St→Saint, Ets→Etablissements

**Blocking:**
- Two channels:
  - name: legal-stripped words, the glued name, char 4-grams
  - address: tokens and adjacent bigrams
- IDF per partition; tokens with doc-frequency above 2000 (name) / 3000 (address) are dropped.
- Score = `cos_name + cos_addr + 0.5·min(cos_name, cos_addr)`.
- Candidates = **forward top-10 per S1 ∪ reverse top-1 S1 per target**.
- Targets without a detected state join every partition.

**Features:** fuzzy name/address ratios, then:
- **digit-run overlap** (`dig_*`, catches house-number decoys)
- **legal-form categories + `lg_conflict`**. French forms each get their own slot: SARL→llc, SAS→inc, SASU→corp, EURL→pc, SA→ltd, SCI→llp, EI→co, SNC→pub.
- **name-word swap** (`nk_unm_s1`, `nk_unm_t`, `nk_swap`)
- name-ambiguity counts (`nk*_cnt_*`)
- blocking ranks and gaps
- **No country feature**, deliberately, so the model transfers to France.

**Decision:** each target → its highest-probability S1, only if p ≥ 0.70 (the OOF optimum).

**Memory notes:**
- My sandbox was 7 GB. Blocking partitions are written to disk and merged in the parent process, and tokenisation streams chunks to disk. A 16 GB machine is comfortable; 32 GB handles `--train_s1 0`.
- If you call pipeline functions from your own script, wrap the calls in `if __name__ == "__main__":`. Blocking spawns worker processes.

## 5. Results log

| Sub | What changed | Train CV F0.5 (US / India / overall) | Public LB |
|---|---|---|---|
| 01 | v2 blocking (country-wide IDF) | 0.973 / 0.970 / 0.9717 | not submitted |
| 02 | v3 per-state blocking. Recall US 96.6→98.4 %, India 95.4→95.9 % | 0.979 / 0.970 / 0.9745 | **0.960** |
| 03 | French legal forms as separate categories (France-only change) | same | **0.965** |
| 04 | + name-swap features, France regions, St/Saint, Ets; model retrained | 0.9747 overall | **0.967** |

- CV = 3-fold GroupKFold by S1 on a **150k-S1-per-country sample** (7 GB sandbox limit).
- Ceiling = a perfect matcher on our candidates ≈ 0.990.
- **Implied France score ≈ 0.92** vs ~0.975 for US/India. This is where the points are.

### Things tried that did NOT help (don't repeat)
- A second-stage stacking model on first-model probabilities (S1/target context): +0.0006.
- "Cluster support" features (other same-name matches of the S1): +0.0007.
- An expected-F0.5 per-S1 decision rule instead of a fixed threshold: no gain.
- Reverse top-2 instead of top-1: +0.001 CV for +30 % candidates, which isn't worth it given the candidate-size ranking.

## 6. What to do next (priority order, ~22 h left)

1. **France error hunting.** It gives the biggest gain per hour, but can only be validated on the leaderboard. Method: score the France test pairs with the model, look at accepted pairs (p ≥ 0.7) and borderline ones (0.5–0.9), and find systematic decoy patterns the features can't see. Patterns already fixed: legal-form swaps, name-word swaps, region vs department, St/Saint. Candidates to check:
   - French house numbers with "bis/ter" (15 bis vs 15)
   - "(France)" and "& Fils/& Cie/et Associés" as noise vs decoy
   - rue/avenue/boulevard swaps
   - accents in names ("Frànce", "Sàint" are typos, fine)
   - France's threshold: it can't be tuned offline, but one leaderboard submission at 0.75 vs 0.70 **for France rows only** would tell whether France has a precision problem

   Each France-only change takes ~15 min to rebuild: re-block France (if norm changed) → features → predict.
2. **Full-data training on the 32 GB Colab machine.** More stable for the private leaderboard; expected +0.1–0.3 points on US/India.
   - Upload the dataset zip to Drive `MyDrive/amlc/`.
   - Open `src/v2/colab_full_training.ipynb` (it clones the branch, so push the PR first).
   - Run with `--train_s1 0 --folds 0` (~2.5–3 h). With `--folds 3`, you also get a CV number for the documentation.
3. **India recall (95.9 %)** is the second-biggest loss: Indic-script names with sparse addresses, and null-address records with shared names. One idea: give records with no address or state an extra reverse candidate (rev top-2 only for those).
4. **Candidate-set size:** fwd top-8 gives US recall 98.2 % at 8.3 cands/S1 vs 98.4 % at 10.3. It might rank better in the final review at almost no F0.5 cost. Test it.
5. **Final package (don't leave it to the last hour):**
   - Fill in `Documentation_template.md` from `REPORT_v3.md` and TEAMMATES.md decisions 11–17.
   - Point `code/business_entity_resolution/README.md` to `src/v2/pipeline.py`.
   - Use pinned `requirements_v2.txt`.
   - `output/` = the chosen submission's `matching_results.tsv` + **its own** `candidate_pairs.tsv`. They must come from the same run; the sub04 pair is on D: already.
6. **Final pick:** choose by CV and public LB together. Don't over-fit to public-LB noise, since the private split decides. Keep the model country-agnostic.

## 7. Gotchas we hit (save yourself the time)

- Official files are TSV: read them with `sep="\t"` / `separator="\t", quote_char=None`.
- Don't merge the split zips with PowerShell `Get-Content | Set-Content`: it can change the encoding. Use `merge.py` or the byte-copy snippet.
- Streets named after states ("Massachusetts Avenue", "Washington St") must not be read as the state. That's handled by whole-address-part detection.
- Measuring token rarity country-wide kills recall; per-state IDF is what fixed US recall.
- Pure-Python pair features are too slow (tens of millions of pairs). Use `rapidfuzz.process.cpdist`.
- Polars 1.44.2 was used. Some older-version polars APIs differ.
- The model was trained without France. For France, rely only on generic signals, and map French forms/regions onto the existing categories rather than adding France-only features (those would have no training signal).
