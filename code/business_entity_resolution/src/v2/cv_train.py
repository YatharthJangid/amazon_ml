"""
train.py - Pair features -> LightGBM (GroupKFold by S1) -> assignment -> macro F0.5.

Usage: python train.py <country> <fwdK> <revR> [max_s1_for_training]
Reads  data/cand_<country>.parquet (from blocking) and the tokenised sources.
"""
import sys, time, gc
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, ".")
from pair_features import record_view, pair_features, FEATURES, name_counts

D = "work"  # folder produced by `pipeline.py prep`


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, s1_ids: pl.Series) -> float:
    """pred/gt: long (s1, t). Macro F0.5 over all s1_ids (singletons included)."""
    tp = pred.join(gt, on=["s1", "t"]).group_by("s1").len("tp")
    np_ = pred.group_by("s1").len("np")
    ng = gt.group_by("s1").len("ng")
    d = (pl.DataFrame({"s1": s1_ids}).join(tp, on="s1", how="left").join(np_, on="s1", how="left")
         .join(ng, on="s1", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("np").clip(1)
    r = pl.col("tp") / pl.col("ng").clip(1)
    f = (1.25 * p * r / (0.25 * p + r)).fill_nan(0)
    f = pl.when((pl.col("ng") == 0) & (pl.col("np") == 0)).then(1.0).otherwise(f)
    return d.select(f.mean()).item()


def assign(c: pl.DataFrame, th: float, rel: float = 0.0) -> pl.DataFrame:
    """Each target goes to at most one S1: its highest-probability S1, if prob >= th."""
    best = c.filter(pl.col("p") >= th).sort("p", descending=True).unique("t", keep="first")
    return best.select("s1", "t")


NS1 = 250_000


def build(country, K, R):
    t0 = time.time()
    c = pl.read_parquet(f"{D}/cand_train_{country}.parquet")
    c = c.filter((pl.col("fwd_rank") < K) | (pl.col("rev_rank") < R))
    # target-side context must be computed on the FULL candidate table (before subsampling S1)
    c = c.with_columns(sc_gap_t=pl.col("sc").max().over("t") - pl.col("sc"), n_s1_for_t=pl.len().over("t"))
    full_ids = c["s1"].unique()
    keep = pl.DataFrame({"s1": full_ids}).sample(min(NS1, full_ids.len()), seed=0)
    c = c.join(keep, on="s1")
    need = ["entity_id", "business_name", "business_address", "ntok_all", "ntok", "atok", "num"]
    nkc = name_counts(pl.read_parquet(f"{D}/train_source1.tok.parquet", columns=["ntok", "country"]).filter(pl.col("country") == country),
                      pl.concat([pl.read_parquet(f"{D}/train_source{i}.tok.parquet", columns=["ntok", "country"]).filter(pl.col("country") == country) for i in (2, 3)]))
    gc.collect()
    s1 = pl.read_parquet(f"{D}/train_source1.tok.parquet", columns=need + ["country"]).filter(pl.col("country") == country).join(keep.rename({"s1": "entity_id"}), on="entity_id")
    tset = c.select(entity_id="t").unique()
    tg = pl.concat([pl.read_parquet(f"{D}/train_source{i}.tok.parquet", columns=need + ["country"]).join(tset, on="entity_id") for i in (2, 3)])
    s1v, tv = record_view(s1), record_view(tg)
    del tg; gc.collect()
    outs = []
    step = 1_000_000
    c = c.sort("s1")
    for s in range(0, c.height, step):
        outs.append(pair_features(c.slice(s, step), s1v, tv, nkc))
        print(f"  features {min(s+step, c.height):,}/{c.height:,} ({time.time()-t0:.0f}s)", flush=True)
    d = pl.concat(outs)
    gt = pl.read_parquet(f"{D}/pos_pairs.parquet", columns=["s1", "t", "c1"]).filter(pl.col("c1") == country).drop("c1")
    d = d.join(gt.with_columns(y=pl.lit(1, pl.Int8)), on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    gt = gt.join(keep, on="s1")
    return d, gt, keep["s1"]


if __name__ == "__main__":
    country, K, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    NS1 = int(sys.argv[4]) if len(sys.argv) > 4 else 250_000
    ds, gts, ids = [], [], []
    for ctry in country.split(","):
        d_, g_, i_ = build(ctry, K, R)
        ds.append(d_); gts.append(g_); ids.append(i_)
        print(ctry, "pairs", d_.height, "blocking recall", d_["y"].sum() / g_.height, flush=True)
        gc.collect()
    d, gt, s1_ids = pl.concat(ds, how="diagonal_relaxed"), pl.concat(gts), pl.concat(ids)
    del ds, gts; gc.collect()
    d.write_parquet(f"{D}/feat_{country.replace(',', '_')}_K{K}R{R}.parquet")
    X = d.select(FEATURES).to_numpy().astype(np.float32)
    y = d["y"].to_numpy()
    groups = d["s1"].cast(pl.Categorical).to_physical().to_numpy()
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=0)
    oof = np.zeros(len(y), np.float32)
    for k, (tr, va) in enumerate(GroupKFold(n_splits=3).split(X, y, groups)):
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr], feature_name=FEATURES), 2000,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        print(f"  fold {k}: best_iter {m.best_iteration}", flush=True)
    imp = sorted(zip(m.feature_importance("gain"), FEATURES), reverse=True)[:15]
    print("top features:", [f for _, f in imp])
    c = d.select("s1", "t").with_columns(p=pl.Series(oof))
    c.write_parquet(f"{D}/oof_{country.replace(',', '_')}_K{K}R{R}.parquet")
    m.save_model(f"{D}/lgbm_fold.txt")
    print("upper bound (perfect matcher on these candidates):", macro_f05(d.filter(pl.col("y") == 1).select("s1", "t"), gt, s1_ids))
    for th in [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        print(f"  th={th:.2f}  assign F0.5={macro_f05(assign(c, th), gt, s1_ids):.4f}   "
              f"no-assign F0.5={macro_f05(c.filter(pl.col('p') >= th).select('s1', 't'), gt, s1_ids):.4f}")
