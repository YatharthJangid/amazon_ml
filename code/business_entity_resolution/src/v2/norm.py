"""
norm.py - Vectorised (Polars) normalisation + tokenisation for entity resolution.

Everything here is expressed as Polars expressions so it runs multi-threaded in Rust
over millions of rows in seconds. No external data: all dictionaries are either
hand-written below or learned from the *training* pairs (see learn_translit_dict).
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict
import polars as pl

# ----------------------------------------------------------------------------------
# Hand-written dictionaries (allowed: no external lookup)
# ----------------------------------------------------------------------------------
# Legal / generic suffixes dropped from the *name* key (kept in raw name for features)
LEGAL = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "l", "c", "ltd",
    "limited", "pvt", "private", "plc", "llp", "lp", "pc", "pllc", "pa", "the", "and", "of",
    "dba", "aka", "a", "k", "public", "sa", "sarl", "sas", "sasu", "eurl", "sci", "ei", "cie",
    "et", "de", "du", "des", "la", "le", "les", "www", "com", "mr", "mrs", "dr", "sri", "shri",
    "shree", "m", "s", "md",
}

ADDR_ABBR = {
    # US street types
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "hwy": "highway", "ct": "court",
    "pl": "place", "pkwy": "parkway", "sq": "square", "ctr": "center", "cir": "circle",
    "ter": "terrace", "trl": "trail", "cr": "county", "cty": "county", "fwy": "freeway",
    "expy": "expressway", "tpke": "turnpike", "aly": "alley", "crk": "creek", "mt": "mount",
    "ft": "fort", "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    # India
    "no": "number", "nr": "near", "opp": "opposite", "hno": "house", "h": "house",
    "extn": "extension", "ext": "extension", "soc": "society", "mg": "marg", "clny": "colony",
    # France
    "bd": "boulevard", "r": "rue", "imp": "impasse", "ch": "chemin", "pl": "place",
    "fbg": "faubourg", "rte": "route", "all": "allee", "qu": "quai", "st": "saint",
}
# NB: "st" is ambiguous (street / saint) - we map both sides identically so it cancels.
ADDR_ABBR["st"] = "street"

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "keralam": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts",
    "tripura": "tr", "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
    "pondicherry": "py",
}

NON_LATIN_RE = r"[ऀ-෿]"  # Devanagari .. Malayalam / Sinhala block range


def _state_pairs():
    pairs = []
    for full, code in list(US_STATES.items()) + list(IN_STATES.items()):
        pairs.append((full, f"_st_{code}"))
    # longest first so "west virginia" beats "virginia"
    pairs.sort(key=lambda x: -len(x[0]))
    return pairs


def base_clean(e: pl.Expr) -> pl.Expr:
    """lower, strip accents (Latin only), punctuation->space, normalise numbers."""
    e = (e.fill_null("")
         .str.replace_all("&", " and ", literal=True)
         .str.normalize("NFKD")
         .str.replace_all(r"[̀-ͯ]", "")   # combining accents (Latin)
         .str.to_lowercase()
         .str.replace_all(r"'s\b", "s")
         .str.replace_all(r"[^\p{L}\p{N}\p{M}]+", " ")
         # split letter<->digit glue like "h no123" / "123b" -> keep number separate
         .str.replace_all(r"(\d)(st|nd|rd|th)\b", "$1")      # 1st / 33nd -> 1 / 33
         .str.replace_all(r"\b0+(\d)", "$1")                   # 05861 -> 5861
         .str.replace_all(r"\s+", " ")
         .str.strip_chars())
    return e


def addr_clean(e: pl.Expr) -> pl.Expr:
    e = base_clean(e)
    # multi-word state names -> single code token (both US and India)
    for full, tok in _state_pairs():
        e = e.str.replace_all(r"\b" + full + r"\b", tok)
    return e


def split_tokens(e: pl.Expr) -> pl.Expr:
    return e.str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0))


def map_tokens(e: pl.Expr, mapping: dict) -> pl.Expr:
    if not mapping:
        return e
    return e.list.eval(pl.element().replace(mapping))


def learn_translit_dict(pairs: pl.DataFrame, min_support: int = 3, min_ratio: float = 0.5,
                        max_rows: int = 3_000_000) -> dict:
    """Learn a non-Latin-token -> Latin-token dictionary from TRAINING positive pairs.

    For name pairs with the same token count where the target is in an Indic script,
    tokens are aligned by position. For addresses, every non-Latin token is associated
    with the Latin tokens of the S1 address (state names are the dominant case) and the
    most frequent partner wins. Only derived from the provided training data.
    """
    cnt = defaultdict(Counter)
    tot = Counter()
    nm = pairs.filter(pl.col("n2").fill_null("").str.contains(NON_LATIN_RE)).head(max_rows)
    nm = nm.select(t1=split_tokens(base_clean(pl.col("n1"))),
                   t2=split_tokens(base_clean(pl.col("n2"))))
    for t1, t2 in nm.iter_rows():
        if len(t1) == len(t2):
            for a, b in zip(t1, t2):
                if re.search(NON_LATIN_RE, b) and not re.search(NON_LATIN_RE, a):
                    cnt[b][a] += 1
                    tot[b] += 1
    am = pairs.filter(pl.col("a2").fill_null("").str.contains(NON_LATIN_RE)).head(max_rows // 3)
    am = am.select(t1=split_tokens(addr_clean(pl.col("a1"))),
                   t2=split_tokens(addr_clean(pl.col("a2"))))
    for t1, t2 in am.iter_rows():
        s1 = set(t1)
        for b in t2:
            if re.search(NON_LATIN_RE, b):
                tot[b] += 1
                for a in s1:
                    if a.startswith("_st_"):
                        cnt[b][a] += 1
    out = {}
    for b, c in cnt.items():
        a, n = c.most_common(1)[0]
        if n >= min_support and n / max(1, tot[b]) >= min_ratio:
            out[b] = a
    return out


US_CODES = {v: f"_st_{v}" for v in US_STATES.values()}
IN_CODES = {v: f"_st_{v}" for v in IN_STATES.values()}
IN_CODES.update({"tg": "_st_ts", "or": "_st_od", "uk": "_st_uk", "cg": "_st_cg", "ct": "_st_cg"})


US_COMP = {**{k: f"_st_{v}" for k, v in US_STATES.items()}, **US_CODES}
IN_COMP = {**{k: f"_st_{v}" for k, v in IN_STATES.items()}, **IN_CODES,
           "new delhi": "_st_dl", "nct of delhi": "_st_dl", "orissa": "_st_od"}


# France: region names AND their departments (sources 2/3 often give the department) -> one region code,
# so France gets per-region blocking like US states / Indian states.
FR_COMP = {"hauts de france": "_st_fr_hdf", "nord": "_st_fr_hdf", "pas de calais": "_st_fr_hdf",
           "nouvelle aquitaine": "_st_fr_naq", "gironde": "_st_fr_naq",
           "pays de la loire": "_st_fr_pdl", "loire atlantique": "_st_fr_pdl"}
# France-specific abbreviations: "St-Nazaire" must meet "Saint-Nazaire" (not "street")
FR_ABBR = {**ADDR_ABBR, "st": "saint", "ste": "sainte", "sts": "saints"}
FR_NAME = {"ets": "etablissements", "st": "saint", "ste": "sainte", "sté": "societe", "ste.": "societe"}


def address_tokens(df: pl.DataFrame, translit: dict) -> pl.DataFrame:
    """Address -> token list, with STATE detected only as a whole comma-separated component
    (so '306 Massachusetts Avenue' is a street, while ', Massachusetts' / ', MA' is a state).
    Native-script state names are mapped through the learned dictionary.
    Returns columns: i (row index), atok (list[str])."""
    ex = (df.select(i=pl.int_range(pl.len(), dtype=pl.UInt32), country=pl.col("country"),
                    comp=pl.col("business_address").fill_null("").str.split(","))
            .explode("comp")
            .with_columns(j=pl.int_range(pl.len(), dtype=pl.UInt32).over("i"))
            .with_columns(toks=split_tokens(base_clean(pl.col("comp")))))
    ex = ex.with_columns(toks=pl.when(pl.col("country") == "India")
                         .then(map_tokens(pl.col("toks"), translit)).otherwise(pl.col("toks")))
    joined = pl.col("toks").list.join(" ")
    st_whole = (pl.when(pl.col("country") == "US").then(joined.replace_strict(US_COMP, default=None))
                  .when(pl.col("country") == "India").then(joined.replace_strict(IN_COMP, default=None))
                  .when(pl.col("country") == "France").then(joined.replace_strict(FR_COMP, default=None))
                  .otherwise(pl.lit(None, pl.String)))
    uq = pl.col("toks").list.unique()
    st_native = pl.when((uq.list.len() == 1) & uq.list.first().str.starts_with("_st_")).then(uq.list.first())
    ex = ex.with_columns(st=pl.coalesce(st_whole, st_native))
    ex = ex.with_columns(toks=pl.when(pl.col("st").is_not_null()).then(pl.concat_list(pl.col("st")))
                         # a state-like token inside a longer component is NOT a state: drop the marker
                         .otherwise(pl.col("toks").list.eval(pl.element().str.replace(r"^_st_", ""))))
    out = (ex.sort(["i", "j"]).group_by("i", maintain_order=True)
             .agg(atok=pl.col("toks").flatten()))
    return out


def add_tokens(df: pl.DataFrame, translit: dict) -> pl.DataFrame:
    """Adds: ntok_all/ntok (name tokens), atok (addr tokens), abig (addr bigrams), num."""
    at = address_tokens(df, translit)
    df = df.with_columns(nm_raw=split_tokens(base_clean(pl.col("business_name"))))
    df = df.with_columns(
        ntok_all=map_tokens(pl.col("nm_raw"), translit),
        atok=at["atok"],
    ).drop("nm_raw")
    df = df.with_columns(atok=pl.when(pl.col("country") == "France")
                         .then(map_tokens(pl.col("atok"), FR_ABBR)).otherwise(map_tokens(pl.col("atok"), ADDR_ABBR)))
    df = df.with_columns(ntok_all=pl.when(pl.col("country") == "France")
                         .then(map_tokens(pl.col("ntok_all"), FR_NAME)).otherwise(pl.col("ntok_all")))
    df = df.with_columns(
        ntok=pl.col("ntok_all").list.eval(pl.element().filter(~pl.element().is_in(list(LEGAL)))),
        nm=pl.col("ntok_all").list.join(" "),
        ad=pl.col("atok").list.join(" "),
        num=pl.col("business_address").fill_null("").str.extract(r"(\d+)").str.replace(r"^0+(\d)", "$1"),
    )
    df = df.with_columns(
        abig=pl.col("atok").list.eval(
            pl.concat_str([pl.element(), pl.element().shift(-1)], separator="_")
        ).list.drop_nulls(),
    )
    return df
