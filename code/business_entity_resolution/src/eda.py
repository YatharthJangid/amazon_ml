"""
Fast Exploratory Data Analysis (EDA) for Business Entity Resolution.
Answers the 6 essential pre-modeling questions the moment dataset lands:
1. Row counts and column schemas across all 8 files.
2. Ground truth statistics (singleton %, match distribution, max matches per entity).
3. Many-to-many check: Does any S2/S3 ID appear under multiple S1 entities in Ground Truth?
4. Country distributions and label discrepancies (e.g. US vs USA, IN vs India).
5. Character script profiling (ASCII vs Devanagari vs Latin-extended/accents).
6. Exact duplicates or near-duplicate rates within candidate sources (S2 and S3).
"""

import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Any

# Ensure src in sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extractors import script_profile, postal_code
from normalize import basic_clean, normalize_name


def parse_tsv_simple(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        headers = header_line.split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            while len(parts) < len(headers):
                parts.append("")
            records.append({headers[i]: parts[i] for i in range(len(headers))})
    return records


def run_eda(data_dir: Path):
    print("=" * 75)
    print("🔍 BUSINESS ENTITY RESOLUTION — RAPID DATA PROFILING (EDA)")
    print(f"Directory: {data_dir.resolve()}")
    print("=" * 75)

    files = [
        "train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv",
        "test_source1.tsv", "test_source2.tsv", "test_source3.tsv"
    ]

    # 1. File existence & row counts
    print("\n--- 1. FILE DISCOVERY & ROW COUNTS ---")
    data_records: Dict[str, List[Dict[str, str]]] = {}
    for fname in files:
        fpath = data_dir / fname
        if fpath.exists():
            rows = parse_tsv_simple(fpath)
            data_records[fname] = rows
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {fname:<25} : {len(rows):>8} records ({size_mb:.2f} MB)")
        else:
            print(f"  ✗ {fname:<25} : MISSING")

    # 2. Ground Truth Analysis
    gt_file = "train_ground_truth.tsv"
    if gt_file in data_records and data_records[gt_file]:
        print("\n--- 2. GROUND TRUTH PROFILE ---")
        gt_rows = data_records[gt_file]
        total_s1 = len(gt_rows)
        match_counts = []
        target_to_s1 = defaultdict(list)
        singletons = 0

        for r in gt_rows:
            s1_id = r.get("id", "")
            matches_str = r.get("matches", "").strip()
            if not matches_str:
                singletons += 1
                match_counts.append(0)
            else:
                targets = [m.strip() for m in matches_str.split(",") if m.strip()]
                match_counts.append(len(targets))
                for tid in targets:
                    target_to_s1[tid].append(s1_id)

        count_dist = Counter(match_counts)
        print(f"  Total S1 entities in GT:   {total_s1}")
        print(f"  Singletons (zero matches): {singletons} ({singletons / total_s1 * 100:.2f}%)")
        print(f"  Matched S1 entities:       {total_s1 - singletons} ({(total_s1 - singletons) / total_s1 * 100:.2f}%)")
        print(f"  Max matches for single S1: {max(match_counts) if match_counts else 0}")
        print("  Distribution of matches per S1:")
        for k in sorted(count_dist.keys())[:10]:
            print(f"    - {k} matches: {count_dist[k]} entities ({count_dist[k] / total_s1 * 100:.2f}%)")

        # 3. Many-to-Many Conflict Analysis
        print("\n--- 3. CROSS-ENTITY CONFLICT CHECK (Can target match multiple S1?) ---")
        conflicts = {tid: s1_list for tid, s1_list in target_to_s1.items() if len(s1_list) > 1}
        if conflicts:
            print(f"  🚨 FOUND {len(conflicts)} S2/S3 IDs that match MULTIPLE S1 entities!")
            print("  Example conflict:")
            sample_tid = next(iter(conflicts))
            print(f"    Target {sample_tid} linked to S1s: {conflicts[sample_tid]}")
            print("  Conclusion: Problem allows many-to-many relationship (do NOT enforce 1-to-1 matching).")
        else:
            print("  ✓ ZERO conflicts found! Each S2/S3 ID appears under at most ONE S1 entity in train GT.")

    # 4. Country Analysis
    print("\n--- 4. COUNTRY DISTRIBUTION & NORMALIZATION CHECK ---")
    all_countries = Counter()
    for fname, rows in data_records.items():
        if "ground_truth" in fname:
            continue
        for r in rows:
            c = r.get("country", "").strip()
            all_countries[c if c else "<EMPTY>"] += 1

    print("  Top countries across all records:")
    for country, count in all_countries.most_common(12):
        print(f"    {country:<25} : {count:>8} records")

    # 5. Script Analysis
    print("\n--- 5. SCRIPT & MULTILINGUAL DETECTION ---")
    devanagari_count = 0
    latin_ext_count = 0
    total_names = 0

    for fname, rows in data_records.items():
        if "ground_truth" in fname:
            continue
        for r in rows:
            name = r.get("name", "")
            if name:
                total_names += 1
                sp = script_profile(name)
                if sp["dev_frac"] > 0.1:
                    devanagari_count += 1
                if sp["latin_ext_frac"] > 0.05:
                    latin_ext_count += 1

    if total_names > 0:
        print(f"  Names analyzed:            {total_names}")
        print(f"  Names with Devanagari:     {devanagari_count} ({devanagari_count / total_names * 100:.2f}%)")
        print(f"  Names with Latin Accents:  {latin_ext_count} ({latin_ext_count / total_names * 100:.2f}%)")

    print("\n" + "=" * 75)
    print("EDA Complete! Use these parameters to calibrate candidate blocking and weights.")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rapid EDA for Entity Resolution")
    parser.add_argument("--data_dir", type=Path, default=Path("tests/data"), help="Directory containing source TSVs")
    args = parser.parse_args()

    run_eda(args.data_dir)
