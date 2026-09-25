"""
mDeBERTa-v3 Cross-Encoder for Business Entity Resolution.
Supports:
- Multilingual sequence pair classification:
  Input format: "[CLS] S1: name | addr | country [SEP] Target: name | addr | country [SEP]"
- Mixed precision (fp16) training for NVIDIA RTX 4060
- Offline execution and local checkpoint serialization
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader

# Ensure local imports work
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_all import load_source_tsv, load_ground_truth
from blocking import BlockingEngine

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AdamW,
        get_linear_schedule_with_warmup,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def format_pair_text(rec1: Dict[str, Any], rec2: Dict[str, Any]) -> Tuple[str, str]:
    """Formats entity attributes into paired text sequences."""
    text_a = f"{rec1.get('name', '')} | {rec1.get('address', '')} | {rec1.get('country', '')}"
    text_b = f"{rec2.get('name', '')} | {rec2.get('address', '')} | {rec2.get('country', '')}"
    return text_a, text_b


class EntityPairDataset(Dataset):
    def __init__(self, texts_a: List[str], texts_b: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.tokenizer(
            self.texts_a[idx],
            self.texts_b[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_deberta(
    data_dir: Path,
    weights_dir: Path,
    model_name: str = "microsoft/mdeberta-v3-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
):
    if not HAS_TRANSFORMERS:
        print("Error: transformers or torch is not installed.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"🤖 TRAINING mDeBERTa-v3 CROSS-ENCODER ON: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / "mdeberta_cross_encoder"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # Prepare datasets using synthetic/train sources
    s1_records = load_source_tsv(data_dir / "train_source1.tsv")
    s2_records = load_source_tsv(data_dir / "train_source2.tsv")
    s3_records = load_source_tsv(data_dir / "train_source3.tsv")
    gt = load_ground_truth(data_dir / "train_ground_truth.tsv")
    target_catalog = s2_records + s3_records
    target_map = {r["id"]: r for r in target_catalog}

    engine = BlockingEngine()
    engine.index_catalog(target_catalog)
    candidates_map = engine.generate_candidates(s1_records, max_candidates_per_query=30)

    texts_a, texts_b, labels = [], [], []

    for s1 in s1_records:
        s1_id = s1["id"]
        true_matches = set(gt.get(s1_id, []))
        cands = set(candidates_map.get(s1_id, []))
        all_pool = list(cands | true_matches)

        for cid in all_pool:
            t_rec = target_map.get(cid)
            if not t_rec:
                continue
            ta, tb = format_pair_text(s1, t_rec)
            texts_a.append(ta)
            texts_b.append(tb)
            labels.append(1 if cid in true_matches else 0)

    if not labels:
        print("No training samples found.")
        return

    dataset = EntityPairDataset(texts_a, texts_b, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for step, batch in enumerate(loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lbl = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=lbl)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

    # Save fine-tuned checkpoint
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"✓ Saved fine-tuned cross-encoder to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mDeBERTa cross-encoder")
    parser.add_argument("--data_dir", type=Path, default=Path("tests/data"))
    parser.add_argument("--weights_dir", type=Path, default=Path("weights"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    train_deberta(args.data_dir, args.weights_dir, epochs=args.epochs, batch_size=args.batch_size)
