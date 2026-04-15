"""
create_kfold_splits.py

Produces K stratified folds (file-level) from the FULL training pool
(data/train_list_5s.txt — all 50 unique recording files).

For each fold i in [0, K):
  data/kfold/fold{i}_train.txt  — segments from (K-1) folds
  data/kfold/fold{i}_val.txt    — segments from fold i (held-out)

K=5 → ~10 val files/fold (vs. current 9 from a single 80/20 split),
and each file appears in the val set exactly once.

Usage:
    python3 create_kfold_splits.py [--k 5] [--seed 42]
              [--source data/train_list_5s.txt]
"""

import argparse
import collections
import os
import random
import re

DEFAULT_SOURCE = "data/train_list_5s.txt"
KFOLD_DIR     = "data/kfold"


def extract_file_id(path: str) -> str:
    """Full-path file ID without _segNNN.wav suffix (collision-safe)."""
    m = re.match(r"^(.+)_seg\d+\.wav$", path)
    return m.group(1) if m else os.path.splitext(path)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",      type=int, default=5)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out_dir", default=KFOLD_DIR, help="Output directory for folds")
    args = parser.parse_args()

    random.seed(args.seed)
    K = args.k

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Parse source into {file_id: (label, [lines])} ──────────────────────
    with open(args.source, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    file_label: dict[str, int]        = {}
    file_lines: dict[str, list[str]]  = collections.defaultdict(list)
    label_files: dict[int, list[str]] = collections.defaultdict(list)

    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        path, lab_str = parts
        try:
            label = int(lab_str.strip())
        except ValueError:
            continue
        fid = extract_file_id(path)
        file_lines[fid].append(line)
        if fid not in file_label:
            file_label[fid] = label
            label_files[label].append(fid)

    # ── 2. Stratified shuffle within each class ────────────────────────────────
    # Assign each file to one of K folds (within its class, round-robin after shuffle)
    file_fold: dict[str, int] = {}
    print(f"\nStratified {K}-fold split (seed={args.seed})")
    print(f"\n{'Class':>6}  {'Total':>6}  {'Files/fold (approx)':>22}")
    print("─" * 40)

    for cls in sorted(label_files):
        fids = label_files[cls][:]
        random.shuffle(fids)
        for i, fid in enumerate(fids):
            file_fold[fid] = i % K
        approx = len(fids) / K
        print(f"  {cls:>4}  {len(fids):>6}  {approx:>22.1f}")

    # ── 3. Write fold files ────────────────────────────────────────────────────
    print(f"\n{'Fold':>5}  {'Val files':>10}  {'Train files':>12}  {'Val segs':>10}  {'Train segs':>12}")
    print("─" * 60)

    for fold in range(K):
        val_fids   = [fid for fid, f in file_fold.items() if f == fold]
        train_fids = [fid for fid, f in file_fold.items() if f != fold]

        val_lines_out   = [l for fid in val_fids   for l in file_lines[fid]]
        train_lines_out = [l for fid in train_fids for l in file_lines[fid]]

        # Shuffle train lines for better DataLoader batching
        random.shuffle(train_lines_out)

        val_path   = os.path.join(args.out_dir, f"fold{fold}_val.txt")
        train_path = os.path.join(args.out_dir, f"fold{fold}_train.txt")

        with open(val_path, "w", encoding="utf-8") as vf:
            vf.write("\n".join(val_lines_out) + "\n")
        with open(train_path, "w", encoding="utf-8") as tf:
            tf.write("\n".join(train_lines_out) + "\n")

        # Per-fold diagnostics
        val_cls   = collections.Counter(file_label[fid] for fid in val_fids)
        print(f"  {fold:>4}  {len(val_fids):>10}  {len(train_fids):>12}  "
              f"{len(val_lines_out):>10}  {len(train_lines_out):>12}  "
              f"val/class={dict(sorted(val_cls.items()))}")

        # Overlap check
        overlap = set(val_fids) & set(train_fids)
        assert not overlap, f"BUG: fold {fold} has overlap: {overlap}"

    print(f"\n✓ All {K} folds written to {args.out_dir}/  (no val/train overlap in any fold)")
    print(f"\nNote: to use in training, point data_conf.train_list → fold{{i}}_train.txt")
    print(f"      and data_conf.val_list → fold{{i}}_val.txt")


if __name__ == "__main__":
    main()
