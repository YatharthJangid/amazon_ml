"""
Pairwise Feature Extraction for Entity Resolution.
Computes string similarity, token overlap, postal code, and script metrics.
"""

from typing import Dict, Any, List
from normalize import basic_clean, normalize_name, normalize_addr
from extractors import postal_code, extract_street_number, script_profile

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


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


def extract_pair_features(rec1: Dict[str, Any], rec2: Dict[str, Any]) -> Dict[str, float]:
    """Computes comprehensive pairwise similarity features between an S1 entity and S2/S3 entity."""
    name1 = rec1.get("name", "")
    name2 = rec2.get("name", "")
    addr1 = rec1.get("address", "")
    addr2 = rec2.get("address", "")
    c1 = str(rec1.get("country", "")).strip().upper()
    c2 = str(rec2.get("country", "")).strip().upper()

    norm_name1 = normalize_name(name1)
    norm_name2 = normalize_name(name2)
    norm_addr1 = normalize_addr(addr1)
    norm_addr2 = normalize_addr(addr2)

    toks_name1 = norm_name1.split()
    toks_name2 = norm_name2.split()
    toks_addr1 = norm_addr1.split()
    toks_addr2 = norm_addr2.split()

    feats: Dict[str, float] = {}

    # 1. Exact match flags
    feats["exact_name"] = 1.0 if norm_name1 == norm_name2 and norm_name1 else 0.0
    feats["name_jaccard"] = jaccard_similarity(toks_name1, toks_name2)
    feats["addr_jaccard"] = jaccard_similarity(toks_addr1, toks_addr2)

    # 2. String distance / fuzzy metrics
    if HAS_RAPIDFUZZ:
        feats["name_ratio"] = fuzz.ratio(norm_name1, norm_name2) / 100.0
        feats["name_token_sort"] = fuzz.token_sort_ratio(norm_name1, norm_name2) / 100.0
        feats["name_token_set"] = fuzz.token_set_ratio(norm_name1, norm_name2) / 100.0
        feats["name_jaro_winkler"] = JaroWinkler.similarity(norm_name1, norm_name2)
        feats["addr_ratio"] = fuzz.ratio(norm_addr1, norm_addr2) / 100.0
    else:
        r = simple_ratio(norm_name1, norm_name2)
        feats["name_ratio"] = r
        feats["name_token_sort"] = r
        feats["name_token_set"] = r
        feats["name_jaro_winkler"] = r
        feats["addr_ratio"] = simple_ratio(norm_addr1, norm_addr2)

    # 3. Postal code match
    post1 = postal_code(addr1, c1)
    post2 = postal_code(addr2, c2)
    if post1 and post2:
        feats["postal_match"] = 1.0 if post1 == post2 else 0.0
        feats["postal_missing"] = 0.0
    else:
        feats["postal_match"] = 0.0
        feats["postal_missing"] = 1.0

    # 4. Street number match
    st1 = extract_street_number(addr1)
    st2 = extract_street_number(addr2)
    if st1 and st2:
        feats["street_num_match"] = 1.0 if st1 == st2 else 0.0
    else:
        feats["street_num_match"] = 0.5  # neutral

    # 5. Country match
    if c1 and c2:
        feats["country_match"] = 1.0 if (c1 in c2 or c2 in c1 or c1 == c2) else 0.0
    else:
        feats["country_match"] = 0.5

    # 6. Script consistency
    sp1 = script_profile(name1)
    sp2 = script_profile(name2)
    feats["script_dev_diff"] = abs(sp1["dev_frac"] - sp2["dev_frac"])
    feats["script_ascii_diff"] = abs(sp1["ascii_frac"] - sp2["ascii_frac"])

    return feats
