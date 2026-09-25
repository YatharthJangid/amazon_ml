"""
Pairwise Feature Extraction for Entity Resolution.
Computes string similarity, token overlap, postal code, and script metrics.
Includes bug fixes and additions from Qwen audit:
- Safe normalized country matching (eliminates US in AUSTRIA false positives)
- Normalized Levenshtein distance (Levenshtein.distance)
- Character bigram cosine similarity (char_wb TF-IDF proxy)
- Token containment (unigram intersection / min token count)
- Address token sort/set and Jaro-Winkler
"""

import math
from collections import Counter
from typing import Dict, Any, List
from normalize import basic_clean, normalize_name, normalize_addr
from extractors import postal_code, extract_street_number, script_profile

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

COUNTRY_ALIAS: Dict[str, str] = {
    "USA": "US", "UNITED STATES": "US", "UNITED STATES OF AMERICA": "US",
    "U.S.": "US", "U.S.A.": "US", "IND": "IN", "INDIA": "IN",
    "FRA": "FR", "FRANCE": "FR", "DEU": "DE", "GERMANY": "DE",
    "GBR": "GB", "UK": "GB", "UNITED KINGDOM": "GB"
}


def norm_country(c: Any) -> str:
    """Normalizes country names/codes to standard 2-letter ISO code."""
    val = (str(c) if c is not None else "").strip().upper()
    return COUNTRY_ALIAS.get(val, val)


def jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def simple_ratio(s1: str, s2: str) -> float:
    """Fallback character bigram Jaccard if rapidfuzz is not yet installed."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    bg1 = set(s1[i:i+2] for i in range(len(s1)-1))
    bg2 = set(s2[i:i+2] for i in range(len(s2)-1))
    if not bg1 or not bg2:
        return 0.0
    return 2.0 * len(bg1 & bg2) / (len(bg1) + len(bg2))


def char_bigram_cos(s1: str, s2: str) -> float:
    """Computes cosine similarity between character bigram bags."""
    if not s1 or not s2:
        return 0.0
    a = Counter(s1[i:i+2] for i in range(len(s1)-1))
    b = Counter(s2[i:i+2] for i in range(len(s2)-1))
    dot = sum((a & b).values())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def extract_pair_features(rec1: Dict[str, Any], rec2: Dict[str, Any]) -> Dict[str, float]:
    """Computes comprehensive pairwise similarity features between an S1 entity and S2/S3 entity."""
    name1 = rec1.get("name", "")
    name2 = rec2.get("name", "")
    addr1 = rec1.get("address", "")
    addr2 = rec2.get("address", "")

    # Normalize countries using alias dictionary
    c1 = norm_country(rec1.get("country", ""))
    c2 = norm_country(rec2.get("country", ""))

    norm_name1 = normalize_name(name1)
    norm_name2 = normalize_name(name2)
    norm_addr1 = normalize_addr(addr1)
    norm_addr2 = normalize_addr(addr2)

    toks_name1 = norm_name1.split()
    toks_name2 = norm_name2.split()
    toks_addr1 = norm_addr1.split()
    toks_addr2 = norm_addr2.split()

    feats: Dict[str, float] = {}

    # 1. Exact match and Jaccard
    feats["exact_name"] = 1.0 if norm_name1 == norm_name2 and norm_name1 else 0.0
    feats["name_jaccard"] = jaccard_similarity(toks_name1, toks_name2)
    feats["addr_jaccard"] = jaccard_similarity(toks_addr1, toks_addr2)

    # 2. String distance / fuzzy metrics
    if HAS_RAPIDFUZZ:
        feats["name_ratio"] = fuzz.ratio(norm_name1, norm_name2) / 100.0
        feats["name_token_sort"] = fuzz.token_sort_ratio(norm_name1, norm_name2) / 100.0
        feats["name_token_set"] = fuzz.token_set_ratio(norm_name1, norm_name2) / 100.0
        feats["name_jaro_winkler"] = JaroWinkler.similarity(norm_name1, norm_name2)
        feats["name_lev_norm"] = 1.0 - Levenshtein.distance(norm_name1, norm_name2) / max(1, max(len(norm_name1), len(norm_name2)))

        feats["addr_ratio"] = fuzz.ratio(norm_addr1, norm_addr2) / 100.0
        feats["addr_token_sort"] = fuzz.token_sort_ratio(norm_addr1, norm_addr2) / 100.0
        feats["addr_token_set"] = fuzz.token_set_ratio(norm_addr1, norm_addr2) / 100.0
        feats["addr_jaro_winkler"] = JaroWinkler.similarity(norm_addr1, norm_addr2)
    else:
        r_name = simple_ratio(norm_name1, norm_name2)
        r_addr = simple_ratio(norm_addr1, norm_addr2)
        feats["name_ratio"] = r_name
        feats["name_token_sort"] = r_name
        feats["name_token_set"] = r_name
        feats["name_jaro_winkler"] = r_name
        feats["name_lev_norm"] = r_name

        feats["addr_ratio"] = r_addr
        feats["addr_token_sort"] = r_addr
        feats["addr_token_set"] = r_addr
        feats["addr_jaro_winkler"] = r_addr

    # 3. TF-IDF character bigram cosine proxies
    feats["name_bg_cos"] = char_bigram_cos(norm_name1, norm_name2)
    feats["addr_bg_cos"] = char_bigram_cos(norm_addr1, norm_addr2)

    # 4. Token containment (handles subset/superset abbreviations)
    s_n1, s_n2 = set(toks_name1), set(toks_name2)
    min_n = min(len(s_n1), len(s_n2))
    feats["name_containment"] = len(s_n1 & s_n2) / max(1, min_n) if min_n > 0 else 0.0

    s_a1, s_a2 = set(toks_addr1), set(toks_addr2)
    min_a = min(len(s_a1), len(s_a2))
    feats["addr_containment"] = len(s_a1 & s_a2) / max(1, min_a) if min_a > 0 else 0.0

    # 5. Name length ratio
    feats["name_len_ratio"] = min(len(norm_name1), len(norm_name2)) / max(1, max(len(norm_name1), len(norm_name2)))

    # 6. Postal code match & prefix match
    post1 = postal_code(addr1, c1)
    post2 = postal_code(addr2, c2)
    if post1 and post2:
        feats["postal_match"] = 1.0 if post1 == post2 else 0.0
        feats["postal_prefix_match"] = 1.0 if (len(post1) >= 3 and len(post2) >= 3 and post1[:3] == post2[:3]) else 0.0
        feats["postal_missing"] = 0.0
    else:
        feats["postal_match"] = 0.0
        feats["postal_prefix_match"] = 0.0
        feats["postal_missing"] = 1.0

    # 7. Street number match
    st1 = extract_street_number(addr1)
    st2 = extract_street_number(addr2)
    if st1 and st2:
        feats["street_num_match"] = 1.0 if st1 == st2 else 0.0
    else:
        feats["street_num_match"] = 0.5  # neutral

    # 8. Country match (safe exact matching on normalized ISO codes)
    if c1 and c2:
        feats["country_match"] = 1.0 if c1 == c2 else 0.0
    else:
        feats["country_match"] = 0.5

    # 9. Script consistency
    sp1 = script_profile(name1)
    sp2 = script_profile(name2)
    feats["script_dev_diff"] = abs(sp1["dev_frac"] - sp2["dev_frac"])
    feats["script_ascii_diff"] = abs(sp1["ascii_frac"] - sp2["ascii_frac"])

    return feats
