import sys
from pathlib import Path

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from evaluate import f05_entity, macro_f05, report_f05


def test_worked_example():
    """Worked example from PDF:
    true: {S2-00047, S3-00812}
    pred: {S2-00047, S2-00193, S3-00812}
    TP=2, Pred=3, True=2 -> P=2/3, R=1.0 -> F0.5 = 1.25*(2/3)*1 / (0.25*(2/3) + 1) = 0.7142857...
    """
    gt = {"S1-00001": ["S2-00047", "S3-00812"]}
    pred = {"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]}
    score = macro_f05(gt, pred)
    assert abs(score - 0.714) < 0.001, f"Expected ~0.714, got {score:.4f}"


def test_singletons():
    gt = {"S1-A": []}
    assert macro_f05(gt, {"S1-A": []}) == 1.0
    assert macro_f05(gt, {"S1-A": ["S2-X"]}) == 0.0


def test_missed_match():
    gt = {"S1-A": ["S2-X"]}
    assert macro_f05(gt, {"S1-A": []}) == 0.0
    assert macro_f05(gt, {"S1-A": ["S2-Z"]}) == 0.0


def test_perfect_match():
    gt = {"S1-A": ["S2-X", "S3-Y"]}
    assert macro_f05(gt, {"S1-A": ["S2-X", "S3-Y"]}) == 1.0


def test_report_f05():
    gt = {
        "S1-1": ["S2-10", "S3-20"],
        "S1-2": [],
        "S1-3": ["S2-30"],
    }
    pred = {
        "S1-1": ["S2-10", "S3-20"],  # perfect (1.0)
        "S1-2": [],                 # singleton correct (1.0)
        "S1-3": ["S2-30", "S2-99"],  # 1 TP, 2 Pred, 1 True -> P=0.5, R=1.0 -> F0.5 = 1.25*0.5*1 / (0.25*0.5 + 1) = 0.5555
    }
    meta = {
        "S1-1": {"country": "US"},
        "S1-2": {"country": "India"},
        "S1-3": {"country": "France"},
    }
    rep = report_f05(gt, pred, metadata_by_s1=meta)
    assert rep["singleton_accuracy"] == 1.0
    assert rep["matched_count"] == 2
    assert "US" in rep["country_f05"]
    assert "FRANCE" in rep["country_f05"]
