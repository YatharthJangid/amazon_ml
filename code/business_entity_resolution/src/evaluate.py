"""
Macro F0.5 Evaluator for Entity Resolution.
Meets exact competition specifications:
- Precision is weighted higher than recall (beta = 0.5): F_0.5 = (1.25 * P * R) / (0.25 * P + R)
- Singleton Ground Truth: true_ids is empty -> score is 1.0 if pred_ids is empty, else 0.0
- Non-empty Ground Truth: missed match -> score is 0.0 if pred_ids is empty
- Macro average across all S1 entities
"""

from typing import Dict, List, Optional, Set, Any


def f05_entity(true_ids: Any, pred_ids: Any) -> float:
    """Computes F0.5 score for a single S1 entity."""
    t_set: Set[str] = set(true_ids) if true_ids else set()
    p_set: Set[str] = set(pred_ids) if pred_ids else set()

    # Singleton ground truth
    if not t_set:
        return 1.0 if not p_set else 0.0

    # Real match missed completely
    if not p_set:
        return 0.0

    tp = len(t_set & p_set)
    if tp == 0:
        return 0.0

    precision = tp / len(p_set)
    recall = tp / len(t_set)

    denominator = 0.25 * precision + recall
    if denominator == 0:
        return 0.0

    return (1.25 * precision * recall) / denominator


def macro_f05(
    gt: Dict[str, List[str]],
    pred: Dict[str, List[str]],
    all_s1_ids: Optional[List[str]] = None,
) -> float:
    """Computes Macro F0.5 across all S1 entities.
    
    gt: {s1_id: list of matched S2/S3 ids}
    pred: {s1_id: list of predicted S2/S3 ids}
    all_s1_ids: complete list of S1 ids required to be evaluated.
    """
    keys = all_s1_ids if all_s1_ids is not None else list(gt.keys())
    if not keys:
        return 0.0

    scores = [f05_entity(gt.get(k, []), pred.get(k, [])) for k in keys]
    return sum(scores) / len(scores)


def report_f05(
    gt: Dict[str, List[str]],
    pred: Dict[str, List[str]],
    all_s1_ids: Optional[List[str]] = None,
    metadata_by_s1: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generates detailed diagnostic breakdown of F0.5 performance:
    - Overall Macro F0.5
    - Singleton accuracy (ground truth empty)
    - Matched entities Macro F0.5 (ground truth non-empty)
    - Match count breakdown (1 match, 2 matches, 3+ matches)
    - Breakdown by country (if metadata_by_s1 provided)
    """
    keys = all_s1_ids if all_s1_ids is not None else list(gt.keys())
    
    singletons = []
    matched = []
    by_bucket: Dict[str, List[float]] = {"1_match": [], "2_matches": [], "3+_matches": []}
    by_country: Dict[str, List[float]] = {}

    for k in keys:
        t = gt.get(k, [])
        p = pred.get(k, [])
        score = f05_entity(t, p)

        if not t:
            singletons.append(score)
        else:
            matched.append(score)
            match_len = len(t)
            if match_len == 1:
                by_bucket["1_match"].append(score)
            elif match_len == 2:
                by_bucket["2_matches"].append(score)
            else:
                by_bucket["3+_matches"].append(score)

        if metadata_by_s1 and k in metadata_by_s1:
            country = str(metadata_by_s1[k].get("country", "UNKNOWN")).upper()
            by_country.setdefault(country, []).append(score)

    summary = {
        "macro_f05": sum(singletons + matched) / len(keys) if keys else 0.0,
        "num_entities": len(keys),
        "singleton_count": len(singletons),
        "singleton_accuracy": sum(singletons) / len(singletons) if singletons else 0.0,
        "matched_count": len(matched),
        "matched_macro_f05": sum(matched) / len(matched) if matched else 0.0,
        "bucket_f05": {
            b: (sum(vals) / len(vals) if vals else 0.0, len(vals))
            for b, vals in by_bucket.items()
        },
        "country_f05": {
            c: (sum(vals) / len(vals) if vals else 0.0, len(vals))
            for c, vals in by_country.items()
        },
    }
    return summary
