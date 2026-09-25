"""
LightGBM Classifier for Business Entity Resolution.
Features:
- GroupKFold on S1 entity ID to prevent leakage across folds
- Hard negative sampling from Candidate Blocking
- Macro F0.5 threshold sweep on out-of-fold validation
- Model serialization to weights/
"""

import sys
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

# Ensure local imports work
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalize import normalize_name, normalize_addr
from blocking import BlockingEngine
from features import extract_pair_features
from evaluate import macro_f05, report_f05
from run_all import load_source_tsv, load_ground_truth

try:
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def build_training_dataset(
    s1_records: List[Dict[str, str]],
    target_catalog: List[Dict[str, str]],
    gt: Dict[str, List[str]],
    max_cands_per_query: int = 50,
    neg_to_pos_ratio: int = 5,
) -> Tuple[List[Dict[str, float]], List[int], List[str], List[str]]:
    """Builds pairwise dataset with positives and hard negatives from candidate blocking."""
    print("Generating candidate pairs for training...")
    target_map = {r["id"]: r for r in target_catalog}

    engine = BlockingEngine()
    engine.index_catalog(target_catalog)
    candidates_map = engine.generate_candidates(s1_records, max_candidates_per_query=max_cands_per_query)

    X_features: List[Dict[str, float]] = []
    y_labels: List[int] = []
    groups: List[str] = []
    pair_ids: List[str] = []

    pos_count = 0
    neg_count = 0

    for s1 in s1_records:
        s1_id = s1["id"]
        true_matches = set(gt.get(s1_id, []))
        cands = set(candidates_map.get(s1_id, []))

        # Always include true matches in candidates pool to ensure model sees positives
        all_pairs_to_consider = list(cands | true_matches)

        positives = [cid for cid in all_pairs_to_consider if cid in true_matches]
        negatives = [cid for cid in all_pairs_to_consider if cid not in true_matches]

        # Downsample negatives if necessary
        if negatives and len(negatives) > len(positives) * neg_to_pos_ratio:
            negatives = random.sample(negatives, len(positives) * neg_to_pos_ratio)

        for cid in positives:
            t_rec = target_map.get(cid)
            if not t_rec:
                continue
            feats = extract_pair_features(s1, t_rec)
            X_features.append(feats)
            y_labels.append(1)
            groups.append(s1_id)
            pair_ids.append(f"{s1_id}::{cid}")
            pos_count += 1

        for cid in negatives:
            t_rec = target_map.get(cid)
            if not t_rec:
                continue
            feats = extract_pair_features(s1, t_rec)
            X_features.append(feats)
            y_labels.append(0)
            groups.append(s1_id)
            pair_ids.append(f"{s1_id}::{cid}")
            neg_count += 1

    print(f"Dataset constructed: {len(y_labels)} pairs (Positives={pos_count}, Negatives={neg_count})")
    return X_features, y_labels, groups, pair_ids


def train_lgbm(
    data_dir: Path,
    weights_dir: Path,
    n_splits: int = 5,
    seed: int = 42,
):
    if not HAS_LGBM:
        print("Error: lightgbm or scikit-learn is not installed in current environment.")
        return

    random.seed(seed)
    np.random.seed(seed)
    weights_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🌲 TRAINING LIGHTGBM ENTITY RESOLUTION CLASSIFIER")
    print("=" * 65)

    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    s1_records = load_source_tsv(s1_path)
    s2_records = load_source_tsv(s2_path)
    s3_records = load_source_tsv(s3_path)
    gt = load_ground_truth(gt_path)
    target_catalog = s2_records + s3_records

    X_feats, y_labels, groups, pair_ids = build_training_dataset(s1_records, target_catalog, gt)

    if not X_feats:
        print("No training pairs found. Check input datasets.")
        return

    feature_names = sorted(list(X_feats[0].keys()))
    X_mat = np.array([[f[col] for col in feature_names] for f in X_feats], dtype=np.float32)
    y_vec = np.array(y_labels, dtype=np.int32)
    groups_arr = np.array(groups)

    print(f"Feature matrix shape: {X_mat.shape} with features: {feature_names}")

    # Set up GroupKFold
    unique_groups = np.unique(groups_arr)
    n_splits = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    oof_preds = np.zeros(len(y_vec), dtype=np.float32)
    models = []

    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "random_state": seed,
    }

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_mat, y_vec, groups=groups_arr)):
        X_trn, y_trn = X_mat[trn_idx], y_vec[trn_idx]
        X_val, y_val = X_mat[val_idx], y_vec[val_idx]

        trn_data = lgb.Dataset(X_trn, label=y_trn, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=trn_data)

        clf = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=500,
            valid_sets=[trn_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        oof_preds[val_idx] = clf.predict(X_val, num_iteration=clf.best_iteration)
        models.append(clf)

    # Threshold optimization for Macro F0.5
    print("\n🔍 Optimizing Decision Threshold for Macro F0.5...")
    best_th = 0.5
    best_f05 = 0.0
    all_s1_ids = [r["id"] for r in s1_records]

    for th in np.arange(0.20, 0.80, 0.05):
        pred_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}
        for idx, prob in enumerate(oof_preds):
            if prob >= th:
                s1_id, cid = pair_ids[idx].split("::")
                pred_map[s1_id].append(cid)

        score = macro_f05(gt, pred_map, all_s1_ids=all_s1_ids)
        if score > best_f05:
            best_f05 = score
            best_th = th

    print(f"Optimal Threshold: {best_th:.2f} | Out-of-Fold Macro F0.5: {best_f05:.4f}")

    # Retrain on full dataset and save
    full_data = lgb.Dataset(X_mat, label=y_vec, feature_name=feature_names)
    final_model = lgb.train(lgb_params, full_data, num_boost_round=models[0].best_iteration or 200)

    model_path = weights_dir / "lgbm_model.txt"
    final_model.save_model(str(model_path))
    print(f"✓ Saved production model to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM for Entity Resolution")
    parser.add_argument("--data_dir", type=Path, default=Path("tests/data"), help="Data directory")
    parser.add_argument("--weights_dir", type=Path, default=Path("weights"), help="Weights directory")
    parser.add_argument("--n_splits", type=int, default=3, help="GroupKFold splits")
    args = parser.parse_args()

    train_lgbm(args.data_dir, args.weights_dir, args.n_splits)
