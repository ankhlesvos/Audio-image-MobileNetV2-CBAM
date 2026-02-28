"""
create_val_split.py

Splits data/train_list_5s.txt into:
  - data/train_list_5s_new.txt  (train: remaining files)
  - data/val_list_5s.txt        (val:   held-out files, file-level stratified)

Split is done at the **file level** (not segment level) so that all segments
from one recording land in the same partition.  A fixed seed guarantees
reproducibility.

Usage:
    python3 create_val_split.py [--val_ratio 0.20] [--seed 42]
"""

import argparse
import collections
import os
import random
import re

TRAIN_LIST   = "data/train_list_5s.txt"
NEW_TRAIN    = "data/train_list_5s_new.txt"
VAL_LIST     = "data/val_list_5s.txt"


def extract_file_id(path: str) -> str:
    """
    Extract a globally-unique recording ID by combining the parent directory
    and the base filename (minus the _segNNNNNN.wav suffix).

    Example:
        'data/deepship_processed_5s/2/33_seg278400.wav'
        → 'data/deepship_processed_5s/2/33'

    This avoids collisions where the same numeric ID appears in different
    class subdirectories.
    """
    # Strip the _segNNN.wav suffix while keeping the directory prefix
    m = re.match(r"^(.+)_seg\d+\.wav$", path)
    if m:
        return m.group(1)
    # Fallback: strip only the extension
    root, _ext = os.path.splitext(path)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.20,
                        help="Fraction of files per class to hold out for val")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Read and index lines by (file_id, class_label) ─────────────────────
    with open(TRAIN_LIST, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    # file_id → label  (to detect label mismatches, which shouldn't happen)
    file_label: dict[str, int] = {}
    # class_label → [file_id, ...]  (unique file IDs per class)
    class_files: dict[int, list[str]] = collections.defaultdict(list)
    # file_id → [line, ...]
    file_lines:  dict[str, list[str]] = collections.defaultdict(list)

    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"  [SKIP malformed line] {line!r}")
            continue
        path, lab_str = parts
        try:
            label = int(lab_str.strip())
        except ValueError:
            print(f"  [SKIP bad label] {line!r}")
            continue

        fid = extract_file_id(path)
        file_lines[fid].append(line)

        if fid not in file_label:
            file_label[fid] = label
            class_files[label].append(fid)
        elif file_label[fid] != label:
            print(f"  [WARN] file {fid} has mixed labels: {file_label[fid]} vs {label}")

    # ── 2. Stratified file-level split ─────────────────────────────────────────
    val_file_ids:   set[str] = set()
    train_file_ids: set[str] = set()

    print(f"\n{'Class':>6}  {'Total files':>12}  {'Val files':>10}  {'Train files':>12}")
    print("─" * 46)
    for cls in sorted(class_files):
        fids = class_files[cls][:]   # copy
        random.shuffle(fids)
        n_val = max(1, round(len(fids) * args.val_ratio))
        val_fids   = fids[:n_val]
        train_fids = fids[n_val:]
        val_file_ids.update(val_fids)
        train_file_ids.update(train_fids)
        print(f"  {cls:>4}  {len(fids):>12}  {n_val:>10}  {len(train_fids):>12}")

    # Safety: no overlap
    overlap = val_file_ids & train_file_ids
    assert not overlap, f"BUG: overlap between val and train file IDs: {overlap}"

    # ── 3. Write output files ──────────────────────────────────────────────────
    # Gather segments
    val_lines   = []
    train_lines = []

    for fid, segs in file_lines.items():
        if fid in val_file_ids:
            val_lines.extend(segs)
        else:
            train_lines.extend(segs)

    # Shuffle train lines for better batching
    random.shuffle(train_lines)

    with open(NEW_TRAIN, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(VAL_LIST, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    # ── 4. Report ──────────────────────────────────────────────────────────────
    # Per-class stats for val
    val_cls: dict[int, int] = collections.Counter()
    for line in val_lines:
        try:
            _, lab = line.split("\t")
            val_cls[int(lab.strip())] += 1
        except Exception:
            pass

    print(f"\nWrote {len(train_lines):>6} train segments → {NEW_TRAIN}")
    print(f"Wrote {len(val_lines):>6} val   segments → {VAL_LIST}")
    print(f"\nVal segments per class: { {k: val_cls[k] for k in sorted(val_cls)} }")
    print(f"Val unique files:       {len(val_file_ids)}")
    print(f"Train unique files:     {len(train_file_ids)}")

    # Sanity: confirm no file ID appears in both sets after read-back
    def unique_fids(path):
        fids = set()
        with open(path) as ff:
            for l in ff:
                l = l.strip()
                if not l:
                    continue
                p = l.split("\t")[0]
                fids.add(extract_file_id(p))
        return fids

    val_fids_check   = unique_fids(VAL_LIST)
    train_fids_check = unique_fids(NEW_TRAIN)
    assert not (val_fids_check & train_fids_check), "LEAKAGE detected!"
    print("\n✓ No file ID leakage between val and train sets.")


if __name__ == "__main__":
    main()
