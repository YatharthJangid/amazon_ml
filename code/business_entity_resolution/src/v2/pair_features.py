"""
features.py - Vectorised pairwise features for candidate (S1, target) pairs.

All string similarities use rapidfuzz.process.cpdist (C++, multi-threaded, element-wise
over aligned lists), so tens of millions of pairs are feasible on a laptop CPU.
No country one-hot: every feature is language-agnostic so the model transfers to the
unseen test country (France).
"""
from __future__ import annotations
import numpy as np
import polars as pl
from rapidfuzz import process, fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

NON_LATIN = r"[ऀ-෿]"

# legal-form categories: distractor records often differ from the true entity ONLY here
LEGAL_CATS = {
    "priv": ["private", "pvt"], "pub": ["public"], "ltd": ["limited", "ltd"], "llp": ["llp"],
    "inc": ["inc", "incorporated"], "llc": ["llc"], "corp": ["corp", "corporation"],
    "co": ["co", "company", "cie"], "pc": ["pc", "pllc", "pa", "plc", "lp"],
    "fr": ["sarl", "sas", "sasu", "eurl", "sa", "sci", "ei", "snc"],
}


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def record_view(df: pl.DataFrame) -> pl.DataFrame:
    """Per-record derived strings used by the pair features."""
    return df.select(
        "entity_id",
        nm=pl.col("ntok_all").list.join(" "),            # full cleaned name (translit applied)
        nk=pl.col("ntok").list.join(" "),                # legal-stripped key
        ng=pl.col("ntok").list.join(""),                 # glued key (domains / hashtags)
        ad=pl.col("atok").list.join(" "),
        nums=pl.col("atok").list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$"))),
        st=pl.col("atok").list.eval(pl.element().filter(pl.element().str.starts_with("_st_"))).list.first(),
        num1=pl.col("num"),
        a_null=pl.col("business_address").is_null() | (pl.col("business_address").str.len_chars() < 3),
        n_nonlatin=pl.col("business_name").fill_null("").str.contains(NON_LATIN),
        n_domain=pl.col("business_name").fill_null("").str.contains(r"(?i)\.com|www\.|^#|^@"),
        n_len=pl.col("ntok").list.len(),
        # every digit run in the raw address (keeps digits glued to letters: 2C16 -> 2, 16)
        dig=pl.col("business_address").fill_null("").str.extract_all(r"\d+")
            .list.eval(pl.element().str.replace(r"^0+(\d)", "$1")),
        **{f"lg_{k}": pl.col("ntok_all").list.eval(pl.element().is_in(v)).list.any() for k, v in LEGAL_CATS.items()},
        a_len=pl.col("atok").list.len(),
    )


def name_counts(s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """How many S1 / S2+S3 records in the same country share each legal-stripped name key.
    High counts = generic / ambiguous names (e.g. 'Dream Investments' x20) -> needs address evidence."""
    k1 = s1.select(nk=pl.col("ntok").list.join(" ")).group_by("nk").len("nk_cnt_s1")
    k2 = tg.select(nk=pl.col("ntok").list.join(" ")).group_by("nk").len("nk_cnt_t")
    return k1.join(k2, on="nk", how="full", coalesce=True).fill_null(0)


def pair_features(pairs: pl.DataFrame, s1v: pl.DataFrame, tv: pl.DataFrame, nkc: pl.DataFrame | None = None) -> pl.DataFrame:
    """pairs: s1, t (+ blocking cols sc, cn, ca, fwd_rank, rev_rank). Returns pairs + features."""
    d = (pairs.join(s1v.rename(lambda c: c + "_1" if c != "entity_id" else "s1"), on="s1", how="left")
              .join(tv.rename(lambda c: c + "_2" if c != "entity_id" else "t"), on="t", how="left"))
    a = {c: d[c].fill_null("").to_list() for c in ["nm_1", "nm_2", "nk_1", "nk_2", "ng_1", "ng_2", "ad_1", "ad_2"]}
    f = {}
    f["nm_ratio"] = _cp(a["nm_1"], a["nm_2"], fuzz.ratio)
    f["nk_ratio"] = _cp(a["nk_1"], a["nk_2"], fuzz.ratio)
    f["nk_tset"] = _cp(a["nk_1"], a["nk_2"], fuzz.token_set_ratio)
    f["nk_tsort"] = _cp(a["nk_1"], a["nk_2"], fuzz.token_sort_ratio)
    f["nk_partial"] = _cp(a["nk_1"], a["nk_2"], fuzz.partial_ratio)
    f["ng_ratio"] = _cp(a["ng_1"], a["ng_2"], fuzz.ratio)
    f["ng_partial"] = _cp(a["ng_1"], a["ng_2"], fuzz.partial_ratio)
    f["nk_jw"] = _cp(a["nk_1"], a["nk_2"], JaroWinkler.normalized_similarity)
    f["ad_ratio"] = _cp(a["ad_1"], a["ad_2"], fuzz.ratio)
    f["ad_tset"] = _cp(a["ad_1"], a["ad_2"], fuzz.token_set_ratio)
    f["ad_tsort"] = _cp(a["ad_1"], a["ad_2"], fuzz.token_sort_ratio)
    f["ad_partial"] = _cp(a["ad_1"], a["ad_2"], fuzz.partial_ratio)
    del a
    ds1 = d["dig_1"].list.join(" ").fill_null("").to_list()
    ds2 = d["dig_2"].list.join(" ").fill_null("").to_list()
    f["dig_ratio"] = _cp(ds1, ds2, fuzz.ratio)
    del ds1, ds2
    d = d.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    d = d.with_columns(
        num1_eq=(pl.col("num1_1") == pl.col("num1_2")).cast(pl.Int8).fill_null(-1),
        nums_jacc=(pl.col("nums_1").list.set_intersection("nums_2").list.len()
                   / pl.col("nums_1").list.set_union("nums_2").list.len().clip(1)),
        nums_in=(pl.col("nums_2").list.set_difference("nums_1").list.len() == 0).cast(pl.Int8),
        st_eq=(pl.col("st_1") == pl.col("st_2")).cast(pl.Int8).fill_null(-1),
        a_null_2=pl.col("a_null_2").cast(pl.Int8),
        n_nonlatin_2=pl.col("n_nonlatin_2").cast(pl.Int8),
        n_domain_2=pl.col("n_domain_2").cast(pl.Int8),
        n_len_1=pl.col("n_len_1"), n_len_2=pl.col("n_len_2"),
        a_len_1=pl.col("a_len_1"), a_len_2=pl.col("a_len_2"),
        is_s3=pl.col("t").str.starts_with("S3").cast(pl.Int8),
        dig_jacc=(pl.col("dig_1").list.set_intersection("dig_2").list.len()
                  / pl.col("dig_1").list.set_union("dig_2").list.len().clip(1)),
        dig_t_extra=pl.col("dig_2").list.set_difference("dig_1").list.len(),
        dig_s_extra=pl.col("dig_1").list.set_difference("dig_2").list.len(),
        dig_n2=pl.col("dig_2").list.len(),
        **{f"lg_{k}": (pl.col(f"lg_{k}_1").cast(pl.Int8) * 2 + pl.col(f"lg_{k}_2").cast(pl.Int8)) for k in LEGAL_CATS},
        lg_conflict=pl.sum_horizontal([(pl.col(f"lg_{k}_1") != pl.col(f"lg_{k}_2")).cast(pl.Int8) for k in LEGAL_CATS]),
    )
    if nkc is not None:
        d = (d.join(nkc.rename({"nk": "nk_1", "nk_cnt_s1": "nk1_cnt_s1", "nk_cnt_t": "nk1_cnt_t"}), on="nk_1", how="left")
              .join(nkc.rename({"nk": "nk_2", "nk_cnt_s1": "nk2_cnt_s1", "nk_cnt_t": "nk2_cnt_t"}), on="nk_2", how="left")
              .with_columns(pl.col("nk1_cnt_s1", "nk1_cnt_t", "nk2_cnt_s1", "nk2_cnt_t").fill_null(0)))
    # context features: how does this pair compare to the alternatives?
    d = d.with_columns(
        n_cand=pl.len().over("s1"),
        sc_gap_s1=pl.col("sc").max().over("s1") - pl.col("sc"),
    )
    drop = [c for c in d.columns if c.endswith("_1") or c.endswith("_2")]
    keep_int = ["a_null_2", "n_nonlatin_2", "n_domain_2", "n_len_1", "n_len_2", "a_len_1", "a_len_2"]
    return d.drop([c for c in drop if c not in keep_int])


FEATURES = ["sc", "cn", "ca", "fwd_rank", "rev_rank", "nm_ratio", "nk_ratio", "nk_tset", "nk_tsort",
            "nk_partial", "ng_ratio", "ng_partial", "nk_jw", "ad_ratio", "ad_tset", "ad_tsort",
            "ad_partial", "num1_eq", "nums_jacc", "nums_in", "st_eq", "a_null_2", "n_nonlatin_2",
            "n_domain_2", "n_len_1", "n_len_2", "a_len_1", "a_len_2", "is_s3", "n_cand",
            "sc_gap_s1", "sc_gap_t", "n_s1_for_t", "dig_jacc", "dig_t_extra", "dig_s_extra", "dig_n2",
            "dig_ratio", "lg_conflict", "nk1_cnt_s1", "nk1_cnt_t", "nk2_cnt_s1", "nk2_cnt_t"] + [f"lg_{k}" for k in LEGAL_CATS]
