"""
Ensemble and Decision Blending Engine.
Combines:
1. GBDT pairwise probability (LightGBM)
2. Deep Cross-Encoder probability (mDeBERTa-v3)
3. High-precision string similarity heuristics

Optimizes Macro F0.5 decision threshold across validation sets.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure local imports work
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import macro_f05, report_f05
from validate_local import validate_submission


def blend_predictions(
    lgbm_scores: Dict[Tuple[str, str], float],
    deberta_scores: Dict[Tuple[str, str], float],
    heuristic_scores: Dict[Tuple[str, str], float],
    weights: Tuple[float, float, float] = (0.50, 0.40, 0.10),
) -> Dict[Tuple[str, str], float]:
    """Blends multi-model predictions using convex combination."""
    w_lgb, w_deb, w_heu = weights
    all_pairs = set(lgbm_scores.keys()) | set(deberta_scores.keys()) | set(heuristic_scores.keys())

    blended: Dict[Tuple[str, str], float] = {}
    for pair in all_pairs:
        s_lgb = lgbm_scores.get(pair, 0.0)
        s_deb = deberta_scores.get(pair, 0.0)
        s_heu = heuristic_scores.get(pair, 0.0)
        blended[pair] = w_lgb * s_lgb + w_deb * s_deb + w_heu * s_heu

    return blended


def find_optimal_threshold(
    blended_scores: Dict[Tuple[str, str], float],
    candidate_pairs_map: Dict[str, List[str]],
    gt: Dict[str, List[str]],
    all_s1_ids: List[str],
) -> Tuple[float, float]:
    """Sweeps threshold grid to find optimal Macro F0.5 score."""
    best_th = 0.50
    best_score = -1.0

    thresholds = [i / 100.0 for i in range(25, 80, 5)]

    for th in thresholds:
        pred_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}
        for s1_id in all_s1_ids:
            cands = candidate_pairs_map.get(s1_id, [])
            for cid in cands:
                score = blended_scores.get((s1_id, cid), 0.0)
                if score >= th:
                    pred_map[s1_id].append(cid)

        score = macro_f05(gt, pred_map, all_s1_ids=all_s1_ids)
        if score > best_score:
            best_score = score
            best_th = th

    return best_th, best_score
