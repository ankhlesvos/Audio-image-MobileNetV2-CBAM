"""
prepare_deepship_data_5s.py
----------------------------
Rebuilds the dataset with:
  - 5-second segments (160000 samples @ 32kHz)
  - Merged class mapping:
      Cargo       -> 0  (merged with Tug)
      Tug         -> 0  (merged with Cargo)
      Passengership -> 1
      Tanker      -> 2
  - Saved to data/deepship_processed_5s/{0,1,2}/
  - List files: data/train_list_5s.txt, data/test_list_5s.txt
"""

import os
import glob
import random
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
SOURCE_DIR = 'DeepShip-main'
TARGET_DIR = 'data/deepship_processed_5s'
TRAIN_LIST_PATH = 'data/train_list_5s.txt'
TEST_LIST_PATH = 'data/test_list_5s.txt'
TARGET_SR = 32000
SEGMENT_DURATION = 5        # Seconds  <--- 5 seconds as requested
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_DURATION   # 160000

# VAD & Cleanup Config
RMS_THRESHOLD = 0.005   # Filter out segments with RMS below this

# Adaptive Overlap Configuration for training data.
# Higher overlap => more segments => more training data.
# Target: >= 1000 train segments per class.
# With 5s segments at 32kHz: stride = SEGMENT_SAMPLES * (1 - overlap)
# overlap 0.8 -> stride = 32000 samples = 1s  (very fine-grained sampling)
# overlap 0.9 -> stride = 16000 samples = 0.5s (very aggressive, for small datasets)
CLASS_OVERLAP_RATIOS = {
    0: 0.9,   # Cargo+Tug  - stride=0.5s
    1: 0.9,   # Passengership - stride=0.5s (was 0.85, need more samples)
    2: 0.9,   # Tanker - stride=0.5s
}

# New 3-class mapping: Cargo+Tug -> 0, Passengership -> 1, Tanker -> 2
CLASS_MAP = {
    'Cargo':         0,
    'Tug':           0,
    'Passengership': 1,
    'Tanker':        2,
}


def calculate_rms(waveform):
    """Calculates RMS amplitude of a tensor."""
    return torch.sqrt(torch.mean(waveform ** 2)).item()


def peak_normalize(waveform):
    """Peak normalizes the waveform to -1.0 to 1.0."""
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform


def process_file(file_path, target_dir, class_id, is_train=True):
    """
    Reads audio, normalizes, inspects for signal (VAD), and slices into 5s segments.
    Training data uses overlapping slices; Test data does not.
    """
    try:
        wav_numpy, sr = sf.read(file_path)
        waveform = torch.from_numpy(wav_numpy).float()

        # Ensure (channels, time)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

    # Mix down to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != TARGET_SR:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    # Peak Normalize
    waveform = peak_normalize(waveform)

    num_samples = waveform.shape[1]

    # Determine stride
    if is_train:
        overlap = CLASS_OVERLAP_RATIOS.get(class_id, 0.5)
        stride = int(SEGMENT_SAMPLES * (1 - overlap))
    else:
        stride = SEGMENT_SAMPLES   # No overlap for test

    saved_segments = []
    filename = Path(file_path).stem

    # Slicing loop
    for start in range(0, num_samples - SEGMENT_SAMPLES + 1, stride):
        end = start + SEGMENT_SAMPLES
        segment = waveform[:, start:end]

        # VAD Check
        rms = calculate_rms(segment)
        if rms < RMS_THRESHOLD:
            continue  # Skip noise/silence

        # Save segment
        save_name = f"{filename}_seg{start}.wav"
        save_path = os.path.join(target_dir, str(class_id), save_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        segment_numpy = segment.squeeze().numpy()
        sf.write(save_path, segment_numpy, TARGET_SR)
        saved_segments.append((save_path, class_id))

    return saved_segments


def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    # Create target directories (3 classes: 0, 1, 2)
    for class_id in [0, 1, 2]:
        os.makedirs(os.path.join(TARGET_DIR, str(class_id)), exist_ok=True)

    all_files = []
    print("Scanning files...")
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory '{class_dir}' not found.")
            continue

        files = glob.glob(os.path.join(class_dir, '**', '*.wav'), recursive=True)
        print(f"  Found {len(files)} files for '{class_name}' -> merged class {class_id}")
        for f in files:
            all_files.append((f, class_id))

    # Stratified Split by merged class id
    random.seed(42)
    files_by_class = {}
    for f, cid in all_files:
        if cid not in files_by_class:
            files_by_class[cid] = []
        files_by_class[cid].append((f, cid))

    train_files_source = []
    test_files_source = []

    print("\nSplitting source files (Stratified, 80/20):")
    for cid in sorted(files_by_class.keys()):
        files = files_by_class[cid]
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * 0.8)

        train_files_source.extend(files[:n_train])
        test_files_source.extend(files[n_train:])

        print(f"  Class {cid}: {n_total} source files -> Train {n_train}, Test {n_total - n_train}")

    random.shuffle(train_files_source)
    random.shuffle(test_files_source)

    test_counts = {0: 0, 1: 0, 2: 0}

    # --- TRAIN: keep ALL segments, no undersampling ---
    # Passengership has the least audio data (~0.64h); undersampling would waste
    # the other classes. Instead keep everything and use class-weighted loss in
    # train.py to handle the natural imbalance.
    print(f"\nProcessing Train Files (With Adaptive Overlap & VAD, 5s segments)...")
    print(f"  Overlap Ratios: {CLASS_OVERLAP_RATIOS}")

    class_segments = {0: [], 1: [], 2: []}

    for file_path, class_id in tqdm(train_files_source):
        segments = process_file(file_path, TARGET_DIR, class_id, is_train=True)
        for seg_path, cid in segments:
            class_segments[cid].append(f"{seg_path}\t{cid}\n")

    # Shuffle and write all segments
    all_train_lines = []
    for cid in sorted(class_segments.keys()):
        all_train_lines.extend(class_segments[cid])
    random.shuffle(all_train_lines)

    with open(TRAIN_LIST_PATH, 'w') as f_out:
        f_out.writelines(all_train_lines)

    train_counts = {cid: len(lines) for cid, lines in class_segments.items()}

    # --- TEST: no balancing applied ---
    print(f"\nProcessing Test Files (No Overlap, With VAD, 5s segments)...")
    with open(TEST_LIST_PATH, 'w') as f_out:
        for file_path, class_id in tqdm(test_files_source):
            segments = process_file(file_path, TARGET_DIR, class_id, is_train=False)
            for seg_path, cid in segments:
                f_out.write(f"{seg_path}\t{cid}\n")
                test_counts[cid] += 1

    print("\n" + "="*55)
    print("Data preparation complete (5s segments, ALL data kept).")
    print("Class mapping: Cargo+Tug -> 0, Passengership -> 1, Tanker -> 2")
    print("="*55)
    total_train = sum(train_counts.values())
    total_test  = sum(test_counts.values())
    print(f"\nSegment Statistics:")
    print(f"  Train (all data): {train_counts}  (Total: {total_train})")
    print(f"  Test  (natural):  {test_counts}   (Total: {total_test})")

    print("\nClass Distribution (Train):")
    for cid, count in sorted(train_counts.items()):
        pct = count / total_train * 100 if total_train > 0 else 0
        print(f"  Class {cid}: {count:>5} ({pct:.1f}%)")

    # Print class weights for use in train.py CrossEntropyLoss(weight=...)
    import torch
    counts_tensor = torch.tensor([train_counts[i] for i in range(3)], dtype=torch.float)
    weights = counts_tensor.sum() / (3 * counts_tensor)
    print(f"\n[ACTION REQUIRED] Add class weights to CrossEntropyLoss in train.py:")
    w_str = ", ".join(f"{w:.4f}" for w in weights.tolist())
    print(f"  class_weights = torch.tensor([{w_str}])")
    print(f"  loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))")

    print(f"\nList files saved:")
    print(f"  Train: {TRAIN_LIST_PATH}")
    print(f"  Test:  {TEST_LIST_PATH}")
    print(f"  Segments saved to: {TARGET_DIR}/")



if __name__ == '__main__':
    main()
