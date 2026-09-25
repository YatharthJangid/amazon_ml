"""
Pairwise Scoring and Threshold Tuning Engine.
Combines multiple signals (name similarity, address similarity, postal match, street number, country)
and tunes decision threshold to maximize Macro F0.5.
"""

from typing import Dict, List, Tuple, Any
from evaluate import macro_f05


def score_pair(feats: Dict[str, float]) -> float:
    """Computes composite similarity score across name, address, postal, and country features."""
    return (
        0.28 * feats.get("exact_name", 0.0)
        + 0.18 * feats.get("name_jaro_winkler", 0.0)
        + 0.10 * feats.get("name_token_set", 0.0)
        + 0.08 * feats.get("name_lev_norm", feats.get("name_ratio", 0.0))
        + 0.12 * feats.get("addr_token_set", feats.get("addr_jaccard", 0.0))
        + 0.08 * feats.get("addr_ratio", 0.0)
        + 0.08 * feats.get("postal_match", 0.0)
        + 0.04 * feats.get("street_num_match", 0.5)
        + 0.04 * feats.get("country_match", 0.5)
        + 0.04 * feats.get("name_bg_cos", 0.0)
        + 0.04 * feats.get("addr_bg_cos", 0.0)
    )


def tune_threshold(
    scored_pairs: List[Tuple[str, str, float]],
    gt: Dict[str, List[str]],
    all_s1_ids: List[str],
    lo: float = 0.30,
    hi: float = 0.95,
    step: float = 0.05,
) -> Tuple[float, float]:
    """Sweeps threshold grid on scored pairs to find threshold maximizing Macro F0.5."""
    best_t = lo
    best_score = -1.0
    t = lo

    while t <= hi + 1e-9:
        pred_map: Dict[str, List[str]] = {}
        for qid, cid, s in scored_pairs:
            if s >= t:
                pred_map.setdefault(qid, []).append(cid)

        sc = macro_f05(gt, pred_map, all_s1_ids=all_s1_ids)
        print(f"  threshold={t:.2f}  |  Macro F0.5 = {sc:.4f}")
        if sc > best_score:
            best_score = sc
            best_t = round(t, 2)
        t += step

    return best_t, best_score
