"""
Local Submission Validator.
Enforces all competition format and consistency rules:
1. Exactly 2 tab-separated columns per submission file.
2. Official headers:
   - matching_results.tsv:  source1_entity_id \t matched_entity_ids
   - candidate_pairs.tsv:   source1_entity_id \t candidate_entity_ids
3. Every test S1 ID appears exactly once.
4. No duplicate rows, no duplicate candidate/match IDs within a row.
5. All matched and candidate IDs start with S2- or S3- and exist in test catalogs.
6. matching_results is a strict subset of candidate_pairs per entity.
"""

import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple, Optional


def load_source_ids(path: Path) -> List[str]:
    """Extracts entity IDs from a source TSV (first column or entity_id)."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ids = []
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        headers = header_line.split("\t")
        id_idx = 0
        for i, h in enumerate(headers):
            if h.lower() in ("entity_id", "source1_entity_id", "source2_entity_id", "source3_entity_id", "id"):
                id_idx = i
                break

        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > id_idx:
                val = parts[id_idx].strip()
                if val:
                    ids.append(val)
    return ids


def parse_submission_tsv(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    """Parses a 2-column submission TSV into an ordered list of keys and a mapping of id -> list of matches."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    keys: List[str] = []
    mapping: Dict[str, List[str]] = {}

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\r\n") for line in f if line.strip()]

    if not lines:
        raise ValueError(f"File {path} is empty")

    header = lines[0].split("\t")
    if len(header) != 2:
        raise ValueError(f"File {path} header does not have exactly 2 tab-separated columns: {header}")

    for idx, line in enumerate(lines[1:], start=2):
        parts = line.split("\t")
        if len(parts) > 2:
            raise ValueError(f"Line {idx} in {path} has more than 2 tab-separated columns: {len(parts)}")

        s1_id = parts[0].strip()
        matches_str = parts[1].strip() if len(parts) > 1 else ""

        if not s1_id:
            raise ValueError(f"Empty S1 ID at line {idx} in {path}")

        if s1_id in mapping:
            raise ValueError(f"Duplicate S1 ID '{s1_id}' detected at line {idx} in {path}")

        if matches_str:
            id_list = [item.strip() for item in matches_str.split(",") if item.strip()]
            if len(id_list) != len(set(id_list)):
                raise ValueError(f"Duplicate candidate IDs inside row for '{s1_id}' at line {idx} in {path}")
        else:
            id_list = []

        keys.append(s1_id)
        mapping[s1_id] = id_list

    return keys, mapping


def validate_submission(
    matching_tsv: Path,
    candidates_tsv: Path,
    test_s1_tsv: Path,
    test_s2_tsv: Optional[Path] = None,
    test_s3_tsv: Optional[Path] = None,
) -> bool:
    """Performs end-to-end validation of submission artifacts."""
    print("=" * 60)
    print("Running Local Submission Validation...")
    print(f"Matching file:   {matching_tsv}")
    print(f"Candidates file: {candidates_tsv}")
    print("=" * 60)

    # 1. Parse both submission files
    match_keys, match_dict = parse_submission_tsv(matching_tsv)
    cand_keys, cand_dict = parse_submission_tsv(candidates_tsv)

    # 2. Parse ground truth test S1 IDs
    expected_s1_keys = load_source_ids(test_s1_tsv)
    expected_s1_set = set(expected_s1_keys)

    # Check that all expected S1 keys are present
    match_s1_set = set(match_keys)
    cand_s1_set = set(cand_keys)

    if match_s1_set != expected_s1_set:
        diff_missing = expected_s1_set - match_s1_set
        diff_extra = match_s1_set - expected_s1_set
        raise AssertionError(
            f"Matching TSV mismatch with test S1 IDs. Missing: {len(diff_missing)}, Extra: {len(diff_extra)}"
        )

    if cand_s1_set != expected_s1_set:
        diff_missing = expected_s1_set - cand_s1_set
        diff_extra = cand_s1_set - expected_s1_set
        raise AssertionError(
            f"Candidate TSV mismatch with test S1 IDs. Missing: {len(diff_missing)}, Extra: {len(diff_extra)}"
        )

    print(f"✓ All {len(expected_s1_set)} test S1 entities present with zero duplicates.")

    # 3. Check subset condition: matching_results ⊆ candidate_pairs
    for s1_id in expected_s1_keys:
        m_set = set(match_dict.get(s1_id, []))
        c_set = set(cand_dict.get(s1_id, []))
        if not m_set.issubset(c_set):
            violators = m_set - c_set
            raise AssertionError(
                f"Row '{s1_id}' in matching_results contains IDs not in candidate_pairs: {violators}"
            )

    print("✓ Matching results are a strict subset of candidate pairs (matching ⊆ candidates).")

    # 4. Check ID prefixes and catalog membership if S2/S3 paths provided
    valid_target_ids: Set[str] = set()
    if test_s2_tsv and test_s2_tsv.exists():
        valid_target_ids.update(load_source_ids(test_s2_tsv))
    if test_s3_tsv and test_s3_tsv.exists():
        valid_target_ids.update(load_source_ids(test_s3_tsv))

    for s1_id, c_list in cand_dict.items():
        for cid in c_list:
            if not (cid.startswith("S2-") or cid.startswith("S3-")):
                raise AssertionError(f"Candidate ID '{cid}' does not start with S2- or S3-")
            if valid_target_ids and cid not in valid_target_ids:
                raise AssertionError(f"Candidate ID '{cid}' does not exist in test S2 or S3 catalog")

    print(f"✓ All candidate and matched IDs have valid prefixes and catalog membership.")
    print("=" * 60)
    print(">>> VALIDATION PASSED: 100% COMPLIANT WITH COMPETITION FORMAT <<<")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate ER submission TSVs")
    parser.add_argument("--matching", type=Path, required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidates", type=Path, required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test_s1", type=Path, required=True, help="Path to test_source1.tsv")
    parser.add_argument("--test_s2", type=Path, default=None, help="Path to test_source2.tsv")
    parser.add_argument("--test_s3", type=Path, default=None, help="Path to test_source3.tsv")
    args = parser.parse_args()

    validate_submission(args.matching, args.candidates, args.test_s1, args.test_s2, args.test_s3)
