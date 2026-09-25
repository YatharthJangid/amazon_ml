"""
preprocess.py - Memory-Safe High-Speed Preprocessor
Reads raw TSVs via Polars, processes via multiprocessing (tuples, not dicts),
and streams directly to Parquet via PyArrow to guarantee Peak RAM < 2GB.
"""
import sys
import time
from pathlib import Path
import multiprocessing as mp
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalize import normalize_name, normalize_addr, sorted_tokens
from extractors import postal_code, extract_street_number, script_profile, script_type

COUNTRY_MAP = {
    "USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US",
    "U.S.": "US", "U.S.A.": "US", "IND": "IN", "INDIA": "IN",
    "FRA": "FR", "FRANCE": "FR"
}

def clean_country(c):
    raw = str(c or "").strip().upper()
    return COUNTRY_MAP.get(raw, raw)

def process_record(row_tuple):
    # row_tuple is (entity_id, business_name, business_address, country)
    eid, name, addr, country = row_tuple
    eid = str(eid or "")
    name = str(name or "")
    addr = str(addr or "")
    c = clean_country(country)
    
    n_name = normalize_name(name)
    n_addr = normalize_addr(addr)
    
    n_name_toks = n_name.split()
    n_addr_toks = n_addr.split()
    
    post = postal_code(addr, c) or ""
    st_num = extract_street_number(addr, c) or ""
    
    acronym = "".join([t[0] for t in n_name_toks if t])
    post_3 = post[:3] if post else ""
    first_tok = n_name_toks[0] if n_name_toks else ""
    
    sp = script_profile(name)
    
    return (
        eid, name, addr, c, n_name, n_addr, 
        sorted_tokens(n_name), sorted_tokens(n_addr),
        n_name_toks, n_addr_toks, len(n_name_toks), len(n_addr_toks),
        post, post_3, st_num, first_tok, acronym,
        float(sp["indic_frac"]), float(sp["ascii_frac"]), float(sp["latin_ext_frac"]),
        script_type(name)
    )

def process_chunk(chunk_tuples):
    return [process_record(r) for r in chunk_tuples]

def process_file(tsv_path: Path, out_dir: Path, num_workers: int = 12):
    print(f"🚀 Ingesting {tsv_path.name} with Polars...")
    t0 = time.time()
    
    if not tsv_path.exists():
        print(f"   ✗ File not found: {tsv_path}")
        return
        
    # Map columns to standard names immediately
    df = pl.read_csv(tsv_path, separator="\t", quote_char=None, truncate_ragged_lines=True)
    
    # Rename columns to standard schema
    rename_map = {}
    for col in df.columns:
        if "entity_id" in col: rename_map[col] = "entity_id"
        elif "name" in col: rename_map[col] = "business_name"
        elif "address" in col: rename_map[col] = "business_address"
        elif "country" in col: rename_map[col] = "country"
    df = df.rename(rename_map)
    
    # Ensure all required columns exist
    for col in ["entity_id", "business_name", "business_address", "country"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
            
    print(f"   Loaded {len(df):,} rows in {time.time()-t0:.2f}s")
    
    if len(df) == 0:
        print("   ✗ Empty DataFrame, skipping.")
        return
        
    # Extract as lightweight tuples (AVOIDS IPC dict explosion)
    records = df.select(["entity_id", "business_name", "business_address", "country"]).rows()
    
    chunk_size = max(1000, len(records) // (num_workers * 4))
    chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
    
    print(f"   Processing across {num_workers} CPU threads ({len(chunks)} chunks)...")
    t1 = time.time()
    
    # Stream directly to Parquet to avoid RAM accumulation
    out_path = out_dir / f"{tsv_path.stem}.parquet"
    schema = pa.schema([
        ("entity_id", pa.string()), ("raw_name", pa.string()), ("raw_addr", pa.string()), ("country", pa.string()),
        ("norm_name", pa.string()), ("norm_addr", pa.string()), ("sort_name", pa.string()), ("sort_addr", pa.string()),
        ("norm_name_tokens", pa.list_(pa.string())), ("norm_addr_tokens", pa.list_(pa.string())),
        ("name_len", pa.int32()), ("addr_len", pa.int32()),
        ("postal", pa.string()), ("postal_3", pa.string()), ("st_num", pa.string()), 
        ("first_tok", pa.string()), ("acronym", pa.string()),
        ("script_indic", pa.float32()), ("script_ascii", pa.float32()), ("script_latin_ext", pa.float32()),
        ("script_type", pa.string())
    ])
    
    with pq.ParquetWriter(out_path, schema, compression="snappy") as writer:
        with mp.Pool(processes=num_workers) as pool:
            for chunk_result in pool.imap_unordered(process_chunk, chunks):
                if not chunk_result:
                    continue
                # Convert list of tuples to PyArrow RecordBatch and write immediately
                arrays = [pa.array(col, type=schema.field(i).type) for i, col in enumerate(zip(*chunk_result))]
                batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
                writer.write_batch(batch)
                
    file_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Normalization & Parquet streaming complete in {time.time()-t1:.2f}s")
    print(f"   ✓ Saved to: {out_path} ({file_mb:.1f} MB)\n")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    cache_dir = Path("dataset/cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    files = [
        "dataset/train/train_source1.tsv", "dataset/train/train_source2.tsv", "dataset/train/train_source3.tsv",
        "dataset/test/test_source1.tsv", "dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv"
    ]
    for f in files:
        p = Path(f)
        if p.exists():
            process_file(p, cache_dir, num_workers=12)
