"""
End-to-End Orchestrator for Business Entity Resolution.
Fully compliant with official competition specification:
- Handles schema column aliases (entity_id, business_name, business_address)
- Writes official submission headers:
  candidate_pairs.tsv:  source1_entity_id \t candidate_entity_ids
  matching_results.tsv: source1_entity_id \t matched_entity_ids
- Candidate Blocking with CountryPartitionedBlocker (US, India, France)
- Multithreaded feature extraction (ThreadPoolExecutor, releases GIL)
- Memory-safe streaming export to features.parquet (PyArrow ParquetWriter)
- Evaluates candidate blocking recall ceiling (target >= 0.98)
- Comprehensive multi-signal scoring via scoring.py
- Validates format compliance via validate_local.py
"""

import os
import sys
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

# Ensure local imports work cleanly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from normalize import normalize_name, normalize_addr
from blocking import CountryPartitionedBlocker, blocking_recall
from features import extract_pair_features
from scoring import score_pair, tune_threshold
from evaluate import macro_f05, report_f05
from validate_local import validate_submission

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

SPEC_ALIASES: Dict[str, str] = {
    "entity_id": "id",
    "source1_entity_id": "id",
    "source2_entity_id": "id",
    "source3_entity_id": "id",
    "business_name": "name",
    "business_address": "address",
    "country": "country",
}


def canonicalize(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Maps competition column names to standard internal keys."""
    return [{SPEC_ALIASES.get(k, k): v for k, v in r.items()} for r in records]


def load_source_tsv(path: Path) -> List[Dict[str, str]]:
    """Loads source TSV into list of record dictionaries."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        headers = header_line.split("\t")
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str.strip():
                continue
            parts = line_str.split("\t")
            while len(parts) < len(headers):
                parts.append("")
            row = {headers[i]: parts[i] for i in range(len(headers))}
            records.append(row)
    return records


def load_ground_truth(path: Path) -> Dict[str, List[str]]:
    """Loads ground truth mapping: s1_id -> [matched_s2_s3_ids]."""
    gt = {}
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()  # skip header
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str.strip():
                continue
            parts = line_str.split("\t")
            s1_id = parts[0].strip()
            matches_str = parts[1].strip() if len(parts) > 1 else ""
            gt[s1_id] = [m.strip() for m in matches_str.split(",") if m.strip()]
    return gt


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    split: str = "test",
    threshold: float = 0.50,
    max_candidates: int = 30,
    max_workers: int = 8,
):
    print("=" * 75)
    print(f"🚀 RUNNING BUSINESS ENTITY RESOLUTION PIPELINE ({split.upper()} MODE)")
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Threshold:        {threshold}")
    print(f"Workers:          {max_workers}")
    print(f"Max candidates:   {max_candidates}")
    print("=" * 75)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve source paths (handles flat or train/test subfolders)
    def resolve_file(d: Path, fname: str) -> Path:
        if (d / fname).exists():
            return d / fname
        if (d / split / fname).exists():
            return d / split / fname
        return d / fname

    s1_path = resolve_file(data_dir, f"{split}_source1.tsv")
    s2_path = resolve_file(data_dir, f"{split}_source2.tsv")
    s3_path = resolve_file(data_dir, f"{split}_source3.tsv")
    gt_path = resolve_file(data_dir, f"{split}_ground_truth.tsv")

    print(f"Loading source TSVs from: {s1_path.parent}...")
    s1_records = canonicalize(load_source_tsv(s1_path))
    s2_records = canonicalize(load_source_tsv(s2_path))
    s3_records = canonicalize(load_source_tsv(s3_path))
    print(f"Loaded: S1={len(s1_records):,}, S2={len(s2_records):,}, S3={len(s3_records):,} records")

    # Combine S2 + S3 into unified candidate target catalog
    target_catalog = s2_records + s3_records
    target_map = {r["id"]: r for r in target_catalog}

    # 2. Candidate Blocking using CountryPartitionedBlocker
    print("\nIndexing candidate catalog by country and generating candidate pairs...")
    engine = CountryPartitionedBlocker()
    engine.index_catalog(target_catalog)
    candidate_pairs_map = engine.generate_candidates(s1_records, max_candidates_per_query=max_candidates)

    # In train mode, compute and print blocking recall ceiling
    gt_map = None
    if gt_path.exists():
        raw_gt = load_ground_truth(gt_path)
        gt_map = {k: set(v) for k, v in raw_gt.items()}
        recall_ceiling = blocking_recall(candidate_pairs_map, raw_gt)
        print("=" * 60)
        print(f"🎯 CANDIDATE BLOCKING RECALL CEILING: {recall_ceiling * 100:.2f}%")
        print("=" * 60)

    # Write candidate_pairs.tsv with exact spec headers
    cand_out_path = output_dir / "candidate_pairs.tsv"
    with open(cand_out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for r in s1_records:
            qid = r["id"]
            cands = candidate_pairs_map.get(qid, [])
            f.write(f"{qid}\t{','.join(cands)}\n")
    print(f"✓ Wrote candidates to: {cand_out_path}")

    # 3. Multithreaded Feature Extraction + Streaming Parquet Export
    print(f"\nExtracting pairwise features (Multithreaded: {max_workers} workers)...")
    feat_path = output_dir / "features.parquet"
    parquet_writer = None
    batch_buffer: List[Dict[str, Any]] = []
    BATCH_FLUSH = 100_000

    def process_query(r: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Tuple[str, List[str]]]:
        qid = r["id"]
        cands = candidate_pairs_map.get(qid, [])
        local_pairs = []
        matched_cands = []

        for cid in cands:
            target_rec = target_map.get(cid)
            if not target_rec:
                continue

            feats = extract_pair_features(r, target_rec)
            score = score_pair(feats)

            if score >= threshold:
                matched_cands.append(cid)

            feats["s1_id"] = qid
            feats["target_id"] = cid
            feats["score"] = float(score)

            if gt_map is not None:
                feats["label"] = 1.0 if cid in gt_map.get(qid, set()) else 0.0
            else:
                feats["label"] = -1.0

            local_pairs.append(feats)

        return local_pairs, (qid, matched_cands)

    matching_map: Dict[str, List[str]] = {}
    total_pairs = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(process_query, s1_records)
        if HAS_TQDM:
            iterator = tqdm(iterator, total=len(s1_records), desc="Extracting Features")

        for local_pairs, (qid, matched) in iterator:
            matching_map[qid] = matched
            if local_pairs:
                batch_buffer.extend(local_pairs)
                total_pairs += len(local_pairs)

            if len(batch_buffer) >= BATCH_FLUSH:
                if HAS_PYARROW:
                    table = pa.Table.from_pylist(batch_buffer)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(feat_path, table.schema, compression="snappy")
                    parquet_writer.write_table(table)
                batch_buffer.clear()

    # Flush remaining batch buffer
    if batch_buffer and HAS_PYARROW:
        table = pa.Table.from_pylist(batch_buffer)
        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(feat_path, table.schema, compression="snappy")
        parquet_writer.write_table(table)
        batch_buffer.clear()

    if parquet_writer is not None:
        parquet_writer.close()
        print(f"✓ Saved {total_pairs:,} total feature rows to: {feat_path} (Snappy compressed)")
    else:
        print(f"✓ Processed {total_pairs:,} total candidate pairs.")

    # Write matching_results.tsv with exact spec headers
    match_out_path = output_dir / "matching_results.tsv"
    with open(match_out_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for r in s1_records:
            qid = r["id"]
            matches = matching_map.get(qid, [])
            f.write(f"{qid}\t{','.join(matches)}\n")
    print(f"✓ Wrote matching results to: {match_out_path}")

    # 4. Enforce submission validation
    print("\nRunning submission integrity check...")
    validate_submission(
        matching_tsv=match_out_path,
        candidates_tsv=cand_out_path,
        test_s1_tsv=s1_path,
        test_s2_tsv=s2_path,
        test_s3_tsv=s3_path,
    )

    # 5. Optional Evaluation against Ground Truth (if available)
    if gt_map is not None:
        print("\nEvaluating against Ground Truth...")
        all_s1 = [r["id"] for r in s1_records]
        raw_gt = load_ground_truth(gt_path)
        macro_score = macro_f05(raw_gt, matching_map, all_s1_ids=all_s1)
        rep = report_f05(raw_gt, matching_map, all_s1_ids=all_s1)
        print("=" * 60)
        print(f"🏆 VALIDATION MACRO F0.5 SCORE (at threshold {threshold:.2f}): {macro_score:.4f}")
        print(f"   Singleton Accuracy:         {rep['singleton_accuracy']:.4f} ({rep['singleton_count']} entities)")
        print(f"   Matched Entities F0.5:       {rep['matched_macro_f05']:.4f} ({rep['matched_count']} entities)")
        print("=" * 60)

    print("\n Pipeline execution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full entity resolution pipeline")
    parser.add_argument("--data_dir", type=Path, default=Path("dataset/train"), help="Directory containing source TSVs")
    parser.add_argument("--output_dir", type=Path, default=Path("output"), help="Output directory for predictions")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split to evaluate")
    parser.add_argument("--threshold", type=float, default=0.50, help="Decision threshold")
    parser.add_argument("--max_candidates", type=int, default=30, help="Max candidates per entity")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of worker threads")
    args = parser.parse_args()

    run_pipeline(
        args.data_dir,
        args.output_dir,
        args.split,
        args.threshold,
        args.max_candidates,
        args.max_workers,
    )
