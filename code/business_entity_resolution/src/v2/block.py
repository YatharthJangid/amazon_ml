"""
block.py - Scalable candidate generation (blocking).

Idea: each record -> a bag of hashed tokens
    n:<name token>   (legal suffixes removed)
    a:<address token>
    b:<address bigram>  (very selective: "1795_westchester", "westchester_drive")
Tokens are IDF-weighted over the (country-specific) S2+S3 catalog. Tokens whose document
frequency exceeds `df_cap` are dropped from the index (they carry little information
and dominate cost). Candidate score = sum of IDF of shared tokens, computed as a chunked
sparse matrix product Q @ T^T (C-speed, bounded memory). The top-K targets per S1
record are kept.

Cost per query = sum of posting-list lengths of its tokens <= n_tokens * df_cap, so the
whole thing is O(N) and scales linearly (no N x M comparisons).
"""
from __future__ import annotations
import time
import numpy as np
import polars as pl
import scipy.sparse as sp


def _grams(e: pl.Expr, n: int = 4) -> pl.Expr:
    """All overlapping char n-grams of a string expression (as a list)."""
    parts = [e.str.slice(o).str.extract_all("." * n) for o in range(n)]
    return pl.concat_list(parts).list.unique()


CHANNELS = {
    # name channel: word tokens + glued name + char 4-grams of glued name (typos, domains, #tags)
    "name": lambda: pl.concat_list(
        pl.col("ntok").list.eval(pl.lit("n:") + pl.element()),
        pl.concat_list(pl.lit("n:") + pl.col("ntok").list.join("")),
        _grams(pl.col("ntok").list.join("")).list.eval(pl.lit("c:") + pl.element()),
    ),
    # address channel: tokens + adjacent bigrams
    "addr": lambda: pl.concat_list(
        pl.col("atok").list.eval(pl.lit("a:") + pl.element()),
        pl.col("abig").list.eval(pl.lit("b:") + pl.element()),
    ),
    "both": lambda: pl.concat_list(CHANNELS["name"](), CHANNELS["addr"]()),
}


from norm import US_STATES, IN_STATES
_ST_CODES = sorted({f"_st_{v}" for v in list(US_STATES.values()) + list(IN_STATES.values())})
_ST_BIT = {c: i for i, c in enumerate(_ST_CODES)}  # < 128 codes -> two uint64 words


def state_masks(df: pl.DataFrame) -> np.ndarray:
    """(n, 2) uint64 bitmask of state tokens found in the address token list 'atok'."""
    ex = (df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("i"), pl.col("atok"))
            .explode("atok").filter(pl.col("atok").str.starts_with("_st_"))
            .with_columns(b=pl.col("atok").replace_strict(_ST_BIT, default=None, return_dtype=pl.Int64))
            .drop_nulls("b").unique(["i", "b"]))
    m = np.zeros((df.height, 2), np.uint64)
    i, b = ex["i"].to_numpy(), ex["b"].to_numpy()
    lo = b < 64
    np.bitwise_or.at(m[:, 0], i[lo], np.left_shift(np.uint64(1), b[lo].astype(np.uint64)))
    np.bitwise_or.at(m[:, 1], i[~lo], np.left_shift(np.uint64(1), (b[~lo] - 64).astype(np.uint64)))
    return m


def add_hashes(df: pl.DataFrame, chunk: int = 500_000) -> pl.DataFrame:
    """Adds hashed token lists h_name / h_addr (list[u64]) and drops nothing."""
    outs = []
    for s in range(0, df.height, chunk):
        d = df.slice(s, chunk)
        m = state_masks(d)
        outs.append(d.with_columns(
            st0=pl.Series(m[:, 0]), st1=pl.Series(m[:, 1]),
            states=pl.col("atok").list.eval(pl.element().filter(pl.element().str.starts_with("_st_"))).list.unique().list.sort(),
            h_name=CHANNELS["name"]().list.eval(
                pl.element().filter(pl.element().str.len_chars() > 2).hash(seed=7)).list.unique(),
            h_addr=CHANNELS["addr"]().list.eval(
                pl.element().filter(pl.element().str.len_chars() > 2).hash(seed=7)).list.unique(),
        ))
    return pl.concat(outs)


def token_long(df: pl.DataFrame, channel: str = "name", chunk: int = 1_000_000) -> pl.DataFrame:
    """(ri, h) long table from the pre-hashed list column h_<channel>."""
    return df.select(pl.col("ri"), h=pl.col(f"h_{channel}")).explode("h").drop_nulls("h")


class TokenBlocker:
    def __init__(self, channel: str = "both", df_cap: int = 2000, topk: int = 15, q_chunk: int = 4000):
        self.channel = channel
        self.df_cap = df_cap
        self.topk = topk
        self.q_chunk = q_chunk

    def _long_chunks(self, df, chunk=500_000):
        for s in range(0, df.height, chunk):
            yield s, token_long(df.slice(s, chunk).with_columns(pl.col("ri") - s), self.channel, chunk=chunk)

    def _csr_from(self, df, chunk=500_000):
        """Build CSR (rows=df rows, cols=vocab) chunk by chunk to bound peak memory."""
        blocks = []
        for s, l in self._long_chunks(df, chunk):
            n = min(chunk, df.height - s)
            l = l.join(self.vocab, on="h", how="inner").sort("ri")
            ri = l["ri"].to_numpy().astype(np.int64)
            indptr = np.zeros(n + 1, dtype=np.int64)
            np.add.at(indptr, ri + 1, 1)
            indptr = np.cumsum(indptr)
            w = l["idf"].to_numpy()
            rl = np.diff(indptr)
            norms = np.sqrt(np.bincount(np.repeat(np.arange(n), rl), weights=w * w, minlength=n))
            norms = np.where(rl > 0, norms, 1.0)
            w = w / np.repeat(norms, rl)
            blocks.append(sp.csr_matrix(
                (w.astype(np.float32), l["vid"].to_numpy().astype(np.int32), indptr),
                shape=(n, self.V)))
            del l
        if not blocks:
            return sp.csr_matrix((0, self.V), dtype=np.float32)
        return sp.vstack(blocks, format="csr")

    def fit(self, targets: pl.DataFrame):
        """targets must have columns ri (0..n-1) + token lists."""
        t0 = time.time()
        n = targets.height
        dfc = targets.select(h=pl.col(f"h_{self.channel}")).explode("h").drop_nulls().group_by("h").len("df")
        vocab = dfc.filter(pl.col("df") <= self.df_cap).with_columns(
            vid=pl.int_range(pl.len(), dtype=pl.UInt32),
            idf=(np.log(n) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32),
        )
        del dfc
        self.vocab = vocab.select("h", "vid", "idf")
        self.V = vocab.height
        T = self._csr_from(targets)
        self.TT = T.T.tocsr()  # V x n
        nnz = T.nnz
        del T
        print(f"    index: {n:,} targets, vocab {self.V:,}, nnz {nnz:,} "
              f"({time.time()-t0:.1f}s)")
        return self

    def query(self, queries: pl.DataFrame):
        """Returns (q_ri, t_ri, score) arrays of the top-K per query."""
        t0 = time.time()
        nq = queries.height
        Q = self._csr_from(queries)
        K = self.topk
        qs, ts, ss = [], [], []
        for s in range(0, nq, self.q_chunk):
            S = (Q[s:s + self.q_chunk] @ self.TT).tocsr()
            S.sort_indices()
            indptr, idx, dat = S.indptr, S.indices, S.data
            lens = np.diff(indptr)
            # vectorised top-K per row: sort by (row, -score)
            rows = np.repeat(np.arange(S.shape[0]), lens)
            order = np.lexsort((-dat, rows))
            rows_o = rows[order]
            # rank within row
            start = indptr[:-1]
            rank = np.arange(len(order)) - np.repeat(start, lens)
            keep = rank < K
            sel = order[keep]
            qs.append(rows_o[keep] + s)
            ts.append(idx[sel])
            ss.append(dat[sel])
        out = (np.concatenate(qs), np.concatenate(ts), np.concatenate(ss))
        print(f"    query: {nq:,} queries -> {len(out[0]):,} pairs ({time.time()-t0:.1f}s)")
        return out


def _lookup(M, r, c):
    """Vectorised M[r, c] for a CSR matrix with sorted indices (0 where absent)."""
    M.sort_indices()
    lens = np.diff(M.indptr)
    keys = np.repeat(np.arange(M.shape[0], dtype=np.int64), lens) * M.shape[1] + M.indices
    want = r.astype(np.int64) * M.shape[1] + c
    pos = np.searchsorted(keys, want)
    pos = np.clip(pos, 0, max(len(keys) - 1, 0))
    hit = (len(keys) > 0) & (keys[pos] == want) if len(keys) else np.zeros(len(want), bool)
    return np.where(hit, M.data[pos] if len(keys) else 0, 0).astype(np.float32)


def _topk_per_group(g, v, k):
    """indices of the top-k values of v within each group g (any order)."""
    order = np.lexsort((-v, g))
    gs = g[order]
    first = np.r_[True, gs[1:] != gs[:-1]]
    start_idx = np.maximum.accumulate(np.where(first, np.arange(len(gs)), 0))
    rank = np.arange(len(gs)) - start_idx
    return order[rank < k], rank[rank < k]


class MultiBlocker:
    """Candidate generation = forward top-K targets per S1  UNION  reverse top-R S1 per target.

    Pair score = w_name * cos_name + w_addr * cos_addr, computed with chunked sparse products
    (only pairs sharing at least one informative token are ever touched). Pairs below `floor`
    are discarded before ranking. Reverse top-R is tracked incrementally across query chunks,
    which enforces the data property that each S2/S3 record belongs to at most one S1.
    """

    def __init__(self, specs=(("name", 2000, 1.0), ("addr", 3000, 1.0)), topk: int = 20,
                 rev_k: int = 2, floor: float = 0.25, q_chunk: int = 4000):
        self.blockers = [(TokenBlocker(channel=c, df_cap=cap, q_chunk=q_chunk), w)
                         for c, cap, w in specs]
        self.topk, self.rev_k, self.floor, self.q_chunk = topk, rev_k, floor, q_chunk

    def fit(self, targets):
        self.nt = targets.height
        for b, _ in self.blockers:
            b.fit(targets)
        return self

    def query(self, queries, q_state=None, t_state=None, conflict_penalty: float = 0.6, both_bonus: float = 0.0):
        """q_state/t_state: optional (n,2) uint64 state bitmasks; pairs whose known states
        do not intersect get `conflict_penalty` subtracted before ranking."""
        t0 = time.time()
        Qs = [(b._csr_from(queries), b, w) for b, w in self.blockers]
        nq, nt, K, R = queries.height, self.nt, self.topk, self.rev_k
        del queries
        # global reverse top-R state: per target -> R best (score, q, cn, ca)
        rs = np.full((nt, R), -1.0, np.float32); rq = np.full((nt, R), -1, np.int64)
        rn = np.zeros((nt, R), np.float32); ra = np.zeros((nt, R), np.float32)
        fwd = []
        for s in range(0, nq, self.q_chunk):
            if (s // self.q_chunk) % 25 == 0:
                print(f'      chunk {s:,}/{nq:,} {time.time()-t0:.0f}s', flush=True)
            mats = [((Q[s:s + self.q_chunk] @ b.TT) * w).tocsr() for Q, b, w in Qs]
            S = mats[0]
            for m in mats[1:]:
                S = S + m
            S = S.tocoo()
            keep = S.data >= self.floor
            r, c, v = S.row[keep], S.col[keep], S.data[keep]
            if q_state is not None:
                qs_, ts_ = q_state[r + s], t_state[c]
                known = (qs_.any(1)) & (ts_.any(1))
                conflict = known & ~((qs_ & ts_).any(1))
                v = v - conflict.astype(np.float32) * conflict_penalty
            keep = v >= self.floor
            r, c, v = r[keep], c[keep], v[keep]
            cn = _lookup(mats[0], r, c); ca = _lookup(mats[1], r, c) if len(mats) > 1 else cn * 0
            if both_bonus:
                v = v + both_bonus * np.minimum(cn, ca)   # reward name AND address evidence
            # forward top-K
            sel, rk = _topk_per_group(r, v, K)
            fwd.append(((r[sel] + s).astype(np.int32), c[sel].astype(np.int32), v[sel].astype(np.float32),
                        cn[sel], ca[sel], rk.astype(np.int8)))
            # reverse: top-R per target within chunk, merged into global state
            sel2, _ = _topk_per_group(c, v, R)
            c2, v2, r2, cn2, ca2 = c[sel2], v[sel2], r[sel2], cn[sel2], ca[sel2]
            del S, mats, r, c, v, cn, ca
            cc = c2
            u, inv = np.unique(cc, return_inverse=True)
            m_s = np.concatenate([rs[u], np.full((len(u), R), -1.0, np.float32)], 1)
            m_q = np.concatenate([rq[u], np.full((len(u), R), -1, np.int64)], 1)
            m_n = np.concatenate([rn[u], np.zeros((len(u), R), np.float32)], 1)
            m_a = np.concatenate([ra[u], np.zeros((len(u), R), np.float32)], 1)
            order = np.lexsort((-v2, cc))
            inv_o = inv[order]
            first = np.r_[True, inv_o[1:] != inv_o[:-1]]
            slot = np.arange(len(order)) - np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
            m_s[inv_o, R + slot] = v2[order]; m_q[inv_o, R + slot] = r2[order] + s
            m_n[inv_o, R + slot] = cn2[order]; m_a[inv_o, R + slot] = ca2[order]
            best = np.argsort(-m_s, 1)[:, :R]
            ar = np.arange(len(u))[:, None]
            rs[u], rq[u], rn[u], ra[u] = m_s[ar, best], m_q[ar, best], m_n[ar, best], m_a[ar, best]
        del Qs
        for b, _ in self.blockers:      # the index is no longer needed: free it before joining
            b.TT = None
        import gc; gc.collect()
        cols = {}
        for i, name in enumerate(["q", "t", "sc", "cn", "ca", "fwd_rank"]):
            cols[name] = np.concatenate([f[i] for f in fwd])
            for f in fwd:
                pass
        del fwd; gc.collect()
        fwd_df = pl.DataFrame(cols); del cols
        tt, kk = np.nonzero(rq >= 0)
        rev_df = pl.DataFrame({"q": rq[tt, kk].astype(np.int32), "t": tt.astype(np.int32), "sc": rs[tt, kk],
                               "cn": rn[tt, kk], "ca": ra[tt, kk], "rev_rank": kk.astype(np.int8)})
        del rs, rq, rn, ra
        out = fwd_df.join(rev_df, on=["q", "t"], how="full", coalesce=True)
        out = out.with_columns(
            sc=pl.coalesce("sc", "sc_right"), cn=pl.coalesce("cn", "cn_right"),
            ca=pl.coalesce("ca", "ca_right"),
            fwd_rank=pl.col("fwd_rank").fill_null(99).cast(pl.Int8), rev_rank=pl.col("rev_rank").fill_null(99).cast(pl.Int8),
        ).drop("sc_right", "cn_right", "ca_right")
        print(f"    multi-query: {nq:,} queries -> fwd {fwd_df.height:,} + rev {rev_df.height:,} "
              f"= {out.height:,} pairs ({time.time()-t0:.1f}s)")
        return out


def block_country(s1: pl.DataFrame, tg: pl.DataFrame, out_dir, K: int = 10, R: int = 1, both_bonus: float = 0.5,
                  name_cap: int = 2000, addr_cap: int = 3000, floor: float = 0.25, q_chunk: int = 4000,
                  log=print, resume: bool = True) -> None:
    """Blocking for ONE country, partitioned by state; writes one parquet per partition to out_dir.
    s1/tg need: h_name, h_addr, states. Targets with no detected state join every partition;
    S1 records with no state are matched against all targets. IDF / df-caps are therefore
    local to a state, so street names that are common nationally still count locally.
    Row indices q (into s1) and t (into tg) are global. Call merge_parts(out_dir) afterwards,
    once the big input frames have been freed."""
    import gc, contextlib, io
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for f in out_dir.glob("part_*.parquet"):
            f.unlink()
    s_st = s1.select(gq=pl.int_range(pl.len(), dtype=pl.UInt32), st=pl.col("states")).explode("st")
    t_st = tg.select(gt=pl.int_range(pl.len(), dtype=pl.UInt32), st=pl.col("states")).explode("st")
    t_none = t_st.filter(pl.col("st").is_null())["gt"]
    parts = s_st["st"].unique().to_list()
    for n_p, p in enumerate(sorted(parts, key=lambda x: (x is None, x or ""))):
        if p is None:
            qi = s_st.filter(pl.col("st").is_null())["gq"]
            ti = pl.Series("gt", range(tg.height), dtype=pl.UInt32)
        else:
            qi = s_st.filter(pl.col("st") == p)["gq"]
            ti = pl.concat([t_st.filter(pl.col("st") == p)["gt"], t_none]).unique().sort()
        if qi.len() == 0 or ti.len() == 0:
            continue
        if (out_dir / f"part_{n_p:04d}.parquet").exists():
            continue          # resume: this partition was finished by an earlier run
        tq = tg.select("h_name", "h_addr")[ti.to_numpy()].with_row_index("ri")
        sq = s1.select("h_name", "h_addr")[qi.to_numpy()].with_row_index("ri")
        b = MultiBlocker(specs=(("name", name_cap, 1.0), ("addr", addr_cap, 1.0)), topk=K, rev_k=R,
                         floor=floor, q_chunk=q_chunk)
        with contextlib.redirect_stdout(io.StringIO()):
            b.fit(tq)
            c = b.query(sq, both_bonus=both_bonus)
        c = c.with_columns(q=qi.gather(c["q"]), t=ti.gather(c["t"]))
        c.write_parquet(out_dir / f"part_{n_p:04d}.parquet")
        log(f"      part {p}: {qi.len():,} S1 x {ti.len():,} targets -> {c.height:,} pairs")
        del b, tq, sq, c; gc.collect()


def merge_parts(out_dir, K: int = 10, R: int = 1) -> pl.DataFrame:
    """Merge per-partition candidates (numpy, low memory): best score per (q, t) pair,
    then global ranks per direction among each direction's survivors."""
    from pathlib import Path
    files = sorted(Path(out_dir).glob("part_*.parquet"))
    cols = {k: [] for k in ["q", "t", "sc", "cn", "ca", "fwd_rank", "rev_rank"]}
    for f in files:
        d = pl.read_parquet(f)
        for k in cols:
            cols[k].append(d[k].to_numpy())
    a = {k: np.concatenate(v) for k, v in cols.items()}; del cols
    key = (a["q"].astype(np.uint64) << np.uint64(32)) | a["t"].astype(np.uint64)
    order = np.lexsort((-a["sc"], key))
    key = key[order]
    first = np.r_[True, key[1:] != key[:-1]]
    starts = np.flatnonzero(first)
    fr = np.minimum.reduceat(a["fwd_rank"][order].astype(np.int16), starts)
    rr = np.minimum.reduceat(a["rev_rank"][order].astype(np.int16), starts)
    best = order[starts]
    q, t, sc = a["q"][best], a["t"][best], a["sc"][best]
    cn, ca = a["cn"][best], a["ca"][best]
    del a, key, order

    def group_rank(g, v, mask):
        """rank (0 = best) of v within group g, among rows where mask; 99 elsewhere."""
        out = np.full(len(g), 99, np.int16)
        idx = np.flatnonzero(mask)
        o = idx[np.lexsort((-v[idx], g[idx]))]
        gs = g[o]
        first = np.r_[True, gs[1:] != gs[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(o)), 0))
        out[o] = np.minimum(np.arange(len(o)) - start, 99)
        return out

    fwd = group_rank(q, sc, fr < K)
    rev = group_rank(t, sc, rr < R)
    return pl.DataFrame({"q": q, "t": t, "sc": sc, "cn": cn, "ca": ca,
                         "fwd_rank": fwd.astype(np.int8), "rev_rank": rev.astype(np.int8)})
