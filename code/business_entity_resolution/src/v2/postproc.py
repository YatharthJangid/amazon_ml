"""Decision rules turning pair probabilities into match lists."""
import numpy as np
import polars as pl


def assign(c: pl.DataFrame, th: float) -> pl.DataFrame:
    """Each target goes to at most one S1 (its highest-probability S1), if p >= th."""
    return c.filter(pl.col("p") >= th).sort("p", descending=True).unique("t", keep="first").select("s1", "t")


def expected_f05(c: pl.DataFrame, floor: float = 0.05, miss: float = 0.0) -> pl.DataFrame:
    """Per S1, choose the top-k (after one-S1-per-target) maximising expected F0.5:
       TP_k = sum_{i<=k} p_i ; FP_k = k - TP_k ; FN_k = G - TP_k ; G = sum_i p_i + miss
       F_k  = 1.25 TP / (1.25 TP + 0.25 FN + FP);   F_0 = prod(1 - p_i) (all-empty is right)."""
    c = c.sort("p", descending=True).unique("t", keep="first").filter(pl.col("p") >= floor)
    c = c.sort(["s1", "p"], descending=[False, True])
    c = c.with_columns(
        k=pl.int_range(1, pl.len() + 1).over("s1"),
        tp=pl.col("p").cum_sum().over("s1"),
        G=pl.col("p").sum().over("s1") + miss,
        f0=(1 - pl.col("p")).product().over("s1"),
    )
    c = c.with_columns(F=1.25 * pl.col("tp") / (1.25 * pl.col("tp") + 0.25 * (pl.col("G") - pl.col("tp"))
                                                  + (pl.col("k") - pl.col("tp"))))
    best = c.group_by("s1").agg(kbest=pl.col("k").get(pl.col("F").arg_max()), Fbest=pl.col("F").max(), f0=pl.col("f0").first())
    best = best.with_columns(kbest=pl.when(pl.col("f0") > pl.col("Fbest")).then(0).otherwise(pl.col("kbest")))
    return c.join(best.select("s1", "kbest"), on="s1").filter(pl.col("k") <= pl.col("kbest")).select("s1", "t")
