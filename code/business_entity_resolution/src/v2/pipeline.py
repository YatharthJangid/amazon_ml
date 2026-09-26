"""
pipeline.py - End-to-end Business Entity Resolution pipeline.

Stages (each cached to --work so any stage can be re-run on its own):
  1. prep      TSV -> parquet -> normalised tokens + hashed token lists     (all splits)
  2. block     per (country, state) partition: sparse token index (name + address channels),
               forward top-K targets per S1  UNION  reverse top-R S1 per target
  3. features  vectorised pair features (rapidfuzz cpdist) on the candidate pairs
  4. train     LightGBM, GroupKFold by S1, OOF threshold search for macro F0.5  (train split)
  5. predict   score test candidates, one-S1-per-target assignment, write
               output/matching_results.tsv + output/candidate_pairs.tsv

Example (16 GB RAM for the default 300k-S1 training sample; 32 GB for --train_s1 0):
  python pipeline.py --data "D:/.../student_resource/dataset" --work work --out output all
  python pipeline.py --data dataset --work work --out output --train_s1 0 --folds 0 all   # full data, no CV
"""
from __future__ import annotations
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from norm import add_tokens, learn_translit_dict          # noqa: E402
from block import add_hashes, block_country, merge_parts   # noqa: E402
from pair_features import record_view, pair_features, FEATURES, name_counts  # noqa: E402

CFG = dict(name_cap=2000, addr_cap=3000, fwd_k=10, rev_r=1, block_k=10, floor=0.25,
           both_bonus=0.5, q_chunk=4000)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------ 1. prep
def read_tsv(p: Path) -> pl.DataFrame:
    return pl.read_csv(p, separator="\t", quote_char=None, infer_schema=False)


def stage_prep(data: Path, work: Path):
    work.mkdir(parents=True, exist_ok=True)
    raw = {}
    for split in ("train", "test"):
        for i in (1, 2, 3):
            raw[f"{split}_source{i}"] = data / split / f"{split}_source{i}.tsv"
    gt = read_tsv(data / "train" / "train_ground_truth.tsv")
    gt.write_parquet(work / "train_ground_truth.parquet")
    # positive pairs (only used to learn the Indic->Latin token dictionary + labels)
    pos = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
             .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
             .rename({"source1_entity_id": "s1", "matched_entity_ids": "t"}))
    s1 = read_tsv(raw["train_source1"]).rename({"entity_id": "s1", "business_name": "n1", "business_address": "a1", "country": "c1"})
    tg = pl.concat([read_tsv(raw[f"train_source{i}"]) for i in (2, 3)]).rename(
        {"entity_id": "t", "business_name": "n2", "business_address": "a2", "country": "c2"})
    pos = pos.join(s1, on="s1").join(tg, on="t")
    pos.select("s1", "t", "c1").write_parquet(work / "pos_pairs.parquet")
    log("learning transliteration dictionary from training pairs")
    tr = learn_translit_dict(pos.filter(pl.col("n2").fill_null("").str.contains(r"[\u0900-\u0DFF]")
                                        | pl.col("a2").fill_null("").str.contains(r"[\u0900-\u0DFF]")))
    json.dump(tr, open(work / "translit.json", "w"), ensure_ascii=False)
    del pos, s1, tg; gc.collect()
    for name, p in raw.items():
        import pyarrow.parquet as pq
        raw = read_tsv(p)
        out, w = work / f"{name}.tok.parquet", None
        for s in range(0, raw.height, 300_000):     # stream chunks to disk: bounded memory
            ch = add_hashes(add_tokens(raw.slice(s, 300_000), tr)).to_arrow()
            if w is None:
                w = pq.ParquetWriter(out, ch.schema)
            w.write_table(ch.cast(w.schema)); del ch; gc.collect()
        w.close()
        log(f"prep {name}: {raw.height:,} rows")
        del raw; gc.collect()


# ------------------------------------------------------------------------------ 2. block
def _block_one(work: Path, split: str, country: str):
    """Runs in its own process so all index memory is returned to the OS afterwards."""
    cols = ["country", "h_name", "h_addr", "states"]
    load = lambda f: (pl.scan_parquet(work / f"{f}.tok.parquet").select(cols)
                        .filter(pl.col("country") == country).collect(engine="streaming"))
    tg = pl.concat([load(f"{split}_source{i}") for i in (2, 3)])
    s1 = load(f"{split}_source1")
    log(f"block {split}/{country}: {s1.height:,} S1 x {tg.height:,} targets")
    if tg.height == 0 or s1.height == 0:   # nothing to match against -> empty candidate set
        pl.DataFrame(schema={"q": pl.UInt32, "t": pl.UInt32, "sc": pl.Float32, "cn": pl.Float32, "ca": pl.Float32,
                             "fwd_rank": pl.Int8, "rev_rank": pl.Int8}).write_parquet(
            work / f"cand_raw_{split}_{country}.parquet")
        return
    pdir = work / f"parts_{split}_{country}"
    block_country(s1, tg, pdir, K=CFG["block_k"], R=CFG["rev_r"], both_bonus=CFG["both_bonus"],
                  name_cap=CFG["name_cap"], addr_cap=CFG["addr_cap"], floor=CFG["floor"],
                  q_chunk=CFG["q_chunk"], log=log)
    # the merge runs in the parent process (stage_block): memory held here is not returned to the OS


def stage_block(work: Path, split: str):
    import multiprocessing as mp
    s1_all = pl.read_parquet(work / f"{split}_source1.tok.parquet", columns=["country"])
    for country in s1_all["country"].unique().sort().to_list():   # open set: US, India, France, ...
        out = work / f"cand_{split}_{country}.parquet"
        if out.exists():
            log(f"block {split}/{country}: cached"); continue
        if not (work / f"cand_raw_{split}_{country}.parquet").exists():
            for attempt in range(3):   # partitions are cached on disk, so a retry resumes where it died
                pr = mp.get_context("spawn").Process(target=_block_one, args=(work, split, country))
                pr.start(); pr.join()
                if pr.exitcode == 0:
                    break
                log(f"block {split}/{country}: worker exited {pr.exitcode}, resuming (attempt {attempt + 2})")
            else:
                raise RuntimeError(f"blocking {split}/{country} failed - out of memory?")
            pdir = work / f"parts_{split}_{country}"
            if pdir.exists():
                merge_parts(pdir, K=CFG["block_k"], R=CFG["rev_r"]).write_parquet(work / f"cand_raw_{split}_{country}.parquet")
        c = pl.read_parquet(work / f"cand_raw_{split}_{country}.parquet")
        tg_ids = pl.concat([pl.read_parquet(work / f"{split}_source{i}.tok.parquet", columns=["entity_id", "country"])
                            for i in (2, 3)]).filter(pl.col("country") == country)["entity_id"]
        s1_ids = pl.read_parquet(work / f"{split}_source1.tok.parquet", columns=["entity_id", "country"]).filter(
            pl.col("country") == country)["entity_id"]
        c = c.with_columns(s1=s1_ids.gather(c["q"]), t=tg_ids.gather(c["t"])).drop("q")
        c = c.filter((pl.col("fwd_rank") < CFG["fwd_k"]) | (pl.col("rev_rank") < CFG["rev_r"]))
        c.write_parquet(out)
        log(f"block {split}/{country}: {c.height:,} candidate pairs ({c.height / max(1, s1_ids.len()):.1f}/S1)")
        del c; gc.collect()


# ------------------------------------------------------------------------------ 3. features
def stage_features(work: Path, split: str, max_s1: int | None = None, seed: int = 0):
    need = ["entity_id", "business_name", "business_address", "ntok_all", "ntok", "atok", "num", "country"]
    for cp in sorted(work.glob(f"cand_{split}_*.parquet")):
        country = cp.stem.split("_", 2)[2]
        out = work / f"feat_{split}_{country}.parquet"
        if out.exists():
            log(f"features {split}/{country}: cached"); continue
        c = pl.read_parquet(cp)
        # target-side context is computed on the FULL candidate table, before any subsampling
        c = c.with_columns(sc_gap_t=pl.col("sc").max().over("t") - pl.col("sc"), n_s1_for_t=pl.len().over("t"))
        if max_s1:   # subsample S1 entities for training speed (whole groups kept)
            keep = c.select("s1").unique().sample(min(max_s1, c["s1"].n_unique()), seed=seed)
            c = c.join(keep, on="s1")
        scan = lambda name, cols: pl.scan_parquet(work / f"{name}.tok.parquet").select(cols).filter(pl.col("country") == country)
        nkc = name_counts(scan(f"{split}_source1", ["ntok", "country"]).collect(),
                          pl.concat([scan(f"{split}_source{i}", ["ntok", "country"]).collect() for i in (2, 3)]))
        gc.collect()
        s1 = scan(f"{split}_source1", need).join(c.select(entity_id="s1").unique().lazy(), on="entity_id").collect()
        tset = c.select(entity_id="t").unique().lazy()
        tg = pl.concat([scan(f"{split}_source{i}", need).join(tset, on="entity_id").collect() for i in (2, 3)])
        s1v, tv = record_view(s1), record_view(tg); del s1, tg; gc.collect()
        step = 1_000_000
        c = c.sort("s1")
        pos = pl.read_parquet(work / "pos_pairs.parquet", columns=["s1", "t"]) if split == "train" else None
        for k, s in enumerate(range(0, c.height, step)):
            d = pair_features(c.slice(s, step), s1v, tv, nkc)
            if pos is not None:
                d = d.join(pos.with_columns(y=pl.lit(1, pl.Int8)), on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
            d.write_parquet(work / f"feat_{split}_{country}.part{k:03d}.parquet")
            log(f"features {split}/{country}: {min(s + step, c.height):,}/{c.height:,}")
            del d; gc.collect()
        out.touch()


# ------------------------------------------------------------------------------ 4/5. model
def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> float:
    tp = pred.join(gt, on=["s1", "t"]).group_by("s1").len("tp")
    np_ = pred.group_by("s1").len("np")
    ng = gt.group_by("s1").len("ng")
    d = (pl.DataFrame({"s1": s1_ids}).join(tp, on="s1", how="left").join(np_, on="s1", how="left")
         .join(ng, on="s1", how="left").fill_null(0))
    p, r = pl.col("tp") / pl.col("np").clip(1), pl.col("tp") / pl.col("ng").clip(1)
    f = (1.25 * p * r / (0.25 * p + r)).fill_nan(0)
    f = pl.when((pl.col("ng") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(f)
    return d.select(f.mean()).item()


def assign(c: pl.DataFrame, th: float) -> pl.DataFrame:
    """Each S2/S3 record is given to at most ONE S1 (its highest-probability S1), if p >= th."""
    return c.filter(pl.col("p") >= th).sort("p", descending=True).unique("t", keep="first").select("s1", "t")


def stage_train(work: Path, folds: int = 3, rounds: int = 800, threshold: float = 0.70):
    """LightGBM on all train feature parts.
    folds >= 2: GroupKFold(S1) CV -> OOF macro F0.5 + threshold search, then a final model.
    folds == 0: skip CV (much faster on the full data); use `rounds` and `threshold`
                (0.70 / ~750-800 rounds were the CV optimum on 300k S1)."""
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    files = sorted(work.glob("feat_train_*.part*.parquet"))
    d = pl.concat([pl.read_parquet(p, columns=FEATURES + ["s1", "t", "y"]) for p in files], how="diagonal_relaxed")
    log(f"train: {d.height:,} pairs, {d['y'].sum():,} positives, {d['s1'].n_unique():,} S1")
    X = d.select(FEATURES).to_numpy().astype(np.float32)
    y = d["y"].to_numpy()
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=0, seed=42)
    meta = {"cfg": CFG, "n_train_pairs": int(d.height)}
    if folds >= 2:
        groups = d["s1"].cast(pl.Categorical).to_physical().to_numpy()
        oof = np.zeros(len(y), np.float32)
        iters = []
        for k, (tr, va) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=FEATURES), 2000,
                          valid_sets=[lgb.Dataset(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
            iters.append(m.best_iteration)
            log(f"train fold {k}: best_iter {m.best_iteration}")
            del m; gc.collect()
        c = d.select("s1", "t").with_columns(p=pl.Series(oof))
        c.write_parquet(work / "oof_train.parquet")
        s1_ids = d["s1"].unique()
        gt = pl.read_parquet(work / "pos_pairs.parquet", columns=["s1", "t"]).join(pl.DataFrame({"s1": s1_ids}), on="s1")
        best = (threshold, -1.0)
        for th in np.arange(0.4, 0.91, 0.05):
            f = macro_f05(assign(c, th), gt, s1_ids)
            log(f"  th={th:.2f}  OOF macro F0.5 (assign) = {f:.4f}")
            if f > best[1]:
                best = (round(float(th), 2), f)
        log(f"best threshold {best[0]:.2f} -> OOF F0.5 {best[1]:.4f}  (blocking recall {d['y'].sum() / gt.height:.4f})")
        threshold, rounds = best[0], int(np.mean(iters) * 1.1)
        meta.update(oof_f05=best[1], cv_iters=iters)
        del oof, c; gc.collect()
    del d; gc.collect()
    final = lgb.train(params, lgb.Dataset(X, y, feature_name=FEATURES), rounds)
    final.save_model(str(work / "lgbm.txt"))
    meta.update(threshold=threshold, rounds=rounds)
    json.dump(meta, open(work / "model_meta.json", "w"), indent=2)
    log(f"saved final model ({rounds} rounds, threshold {threshold:.2f})")


def stage_predict(work: Path, out: Path):
    import lightgbm as lgb
    m = lgb.Booster(model_file=str(work / "lgbm.txt"))
    th = json.load(open(work / "model_meta.json"))["threshold"]
    preds, cands = [], []
    for fp in sorted(work.glob("feat_test_*.part*.parquet")):
        d = pl.read_parquet(fp)
        p = m.predict(d.select(FEATURES).to_numpy().astype(np.float32))
        preds.append(d.select("s1", "t").with_columns(p=pl.Series(p)))
        cands.append(d.select("s1", "t"))
    c = pl.concat(preds)
    matched = assign(c, th)
    s1_ids = pl.read_parquet(work / "test_source1.tok.parquet", columns=["entity_id"])["entity_id"]
    out.mkdir(parents=True, exist_ok=True)

    def write(long: pl.DataFrame, col: str, path: Path):
        agg = long.group_by("s1").agg(pl.col("t").unique().sort().str.join(",").alias(col))
        full = pl.DataFrame({"source1_entity_id": s1_ids}).join(
            agg.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left").with_columns(pl.col(col).fill_null(""))
        full.write_csv(path, separator="\t", quote_style="never")
    write(pl.concat(cands), "candidate_entity_ids", out / "candidate_pairs.tsv")
    write(matched, "matched_entity_ids", out / "matching_results.tsv")
    log(f"wrote {out}: {matched.height:,} matches for {s1_ids.len():,} S1 "
        f"({matched['s1'].n_unique() / s1_ids.len():.3f} non-empty); threshold {th:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="dataset/ folder with train/ and test/")
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--out", type=Path, default=Path("output"))
    ap.add_argument("--train_s1", type=int, default=300_000, help="S1 entities per country used to train (0 = all)")
    ap.add_argument("--folds", type=int, default=3, help="GroupKFold folds for CV; 0 = skip CV (fast)")
    ap.add_argument("--rounds", type=int, default=800, help="boosting rounds when --folds 0")
    ap.add_argument("--threshold", type=float, default=0.70, help="decision threshold when --folds 0")
    ap.add_argument("stages", nargs="+", choices=["prep", "block", "features", "train", "predict", "all"])
    a = ap.parse_args()
    st = set(a.stages)
    if "all" in st:
        st = {"prep", "block", "features", "train", "predict"}
    if "prep" in st:
        stage_prep(a.data, a.work)
    if "block" in st:
        stage_block(a.work, "train"); stage_block(a.work, "test")
    if "features" in st:
        stage_features(a.work, "train", a.train_s1); stage_features(a.work, "test")
    if "train" in st:
        stage_train(a.work, a.folds, a.rounds, a.threshold)
    if "predict" in st:
        stage_predict(a.work, a.out)
