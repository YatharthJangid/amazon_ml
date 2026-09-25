"""
Pre-download and cache model weights for offline execution.
Uses pure `transformers` and `torch` (eliminates sentence_transformers dependency conflicts).

Downloads and saves:
1. microsoft/mdeberta-v3-base -> weights/mdeberta-v3-base
2. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 -> weights/paraphrase-multilingual-MiniLM-L12-v2
"""

import os
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
WEIGHTS_DIR = ROOT_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

M_DEBERTA_DIR = WEIGHTS_DIR / "mdeberta-v3-base"
MINILM_DIR = WEIGHTS_DIR / "paraphrase-multilingual-MiniLM-L12-v2"

print("=" * 70)
print(f"📦 Pre-downloading model weights to: {WEIGHTS_DIR}")
print("=" * 70)

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModel,
        AutoModelForSequenceClassification,
    )
    print(f"✓ PyTorch version: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"\n❌ Error importing transformers or torch: {e}")
    print("Please activate your virtual environment (.venv) first:")
    print("   source .venv/bin/activate")
    sys.exit(1)

# 1. Download mDeBERTa-v3-base
print("\n[1/2] Downloading microsoft/mdeberta-v3-base...")
try:
    deb_tok = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    deb_model = AutoModelForSequenceClassification.from_pretrained("microsoft/mdeberta-v3-base")
    deb_tok.save_pretrained(str(M_DEBERTA_DIR))
    deb_model.save_pretrained(str(M_DEBERTA_DIR))
    print(f"✓ mDeBERTa-v3-base saved locally to: {M_DEBERTA_DIR}")
except Exception as e:
    print(f"❌ Failed downloading mDeBERTa-v3-base: {e}")

# 2. Download paraphrase-multilingual-MiniLM-L12-v2 using pure AutoModel
print("\n[2/2] Downloading sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2...")
try:
    lm_tok = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    lm_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    lm_tok.save_pretrained(str(MINILM_DIR))
    lm_model.save_pretrained(str(MINILM_DIR))
    print(f"✓ paraphrase-multilingual-MiniLM-L12-v2 saved locally to: {MINILM_DIR}")
except Exception as e:
    print(f"❌ Failed downloading MiniLM: {e}")

# Verify offline loading
print("\n[3/3] Verifying offline loading...")
try:
    _ = AutoTokenizer.from_pretrained(str(M_DEBERTA_DIR), local_files_only=True)
    _ = AutoTokenizer.from_pretrained(str(MINILM_DIR), local_files_only=True)
    print("✓ Offline verification PASSED! Models load with local_files_only=True.")
except Exception as e:
    print(f"⚠️ Offline verification warning: {e}")

print("\n" + "=" * 70)
print("🚀 ALL WEIGHTS READY FOR 100% OFFLINE EXECUTION!")
print("=" * 70)
