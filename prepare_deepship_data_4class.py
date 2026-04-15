import os
import glob
import random
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path
import collections
import statistics
import argparse

# --- Base Configuration ---
SOURCE_DIR = 'DeepShip-main'
TARGET_SR = 32000
SEGMENT_DURATION = 5  # Seconds
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_DURATION

# ============================================================================
# 4-CLASS DEEPSHIP: Each class gets its own label
# 0 = Cargo
# 1 = Passengership
# 2 = Tanker
# 3 = Tug
# ============================================================================
CLASS_MAP = {
    'Cargo':         0,
    'Passengership': 1,
    'Tanker':        2,
    'Tug':           3,
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


def process_file(file_path, target_dir, class_id, is_train=True, overlap_ratios=None, rms_threshold=0.005):
    """
    Reads audio, normalizes, inspects for signal (VAD), and slices into 5s segments.
    """
    try:
        wav_numpy, sr = sf.read(file_path)
        waveform = torch.from_numpy(wav_numpy).float()
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

    # Resample
    if sr != TARGET_SR:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    # Peak Normalize
    waveform = peak_normalize(waveform)
    num_samples = waveform.shape[1]

    # Determine stride
    if is_train and overlap_ratios is not None:
        overlap = overlap_ratios.get(class_id, 0.5)
        stride = int(SEGMENT_SAMPLES * (1 - overlap))
    else:
        stride = SEGMENT_SAMPLES   # No overlap for test

    saved_segments = []
    filename = Path(file_path).stem

    for start in range(0, num_samples - SEGMENT_SAMPLES + 1, stride):
        end = start + SEGMENT_SAMPLES
        segment = waveform[:, start:end]

        # VAD Check
        rms = calculate_rms(segment)
        if rms < rms_threshold:
            continue

        save_name = f"{filename}_seg{start}.wav"
        save_path = os.path.join(target_dir, str(class_id), save_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, segment.squeeze().numpy(), TARGET_SR)
        saved_segments.append((save_path, class_id))

    return saved_segments


def main():
    parser = argparse.ArgumentParser(description="DeepShip 4-Class 5s Data Preparation")
    parser.add_argument("--version_name", type=str, default="deepship_4class", help="Version name for outputs")
    parser.add_argument("--overlap", type=float, nargs=4, default=[0.85, 0.96, 0.94, 0.95],
                        help="Overlap ratios for classes 0(Cargo), 1(Passenger), 2(Tanker), 3(Tug)")
    parser.add_argument("--no_undersample", action="store_true", help="Disable segment-level undersampling")
    parser.add_argument("--max_segs_per_file", type=int, default=0, help="Cap maximum segments per raw file (0 = no limit)")
    parser.add_argument("--rms_threshold", type=float, default=0.005, help="VAD RMS threshold")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split consistency")
    args = parser.parse_args()

    # Apply configuration
    TARGET_DIR = f'data/{args.version_name}'
    TRAIN_LIST_PATH = f'data/{args.version_name}/train_list.txt'
    TEST_LIST_PATH = f'data/{args.version_name}/test_list.txt'

    CLASS_OVERLAP_RATIOS = {0: args.overlap[0], 1: args.overlap[1], 2: args.overlap[2], 3: args.overlap[3]}

    print(f"=== Running Data Prep: {args.version_name} (4-CLASS DEEPSHIP) ===")
    print(f"  Target Dir    : {TARGET_DIR}")
    print(f"  Overlap       : {CLASS_OVERLAP_RATIOS}")
    print(f"  Undersampling : {'Disabled' if args.no_undersample else 'Enabled'}")
    print(f"  Max Segs/file : {args.max_segs_per_file if args.max_segs_per_file > 0 else 'Unlimited'}")
    print(f"  VAD RMS Thresh: {args.rms_threshold}")
    print(f"  Class Map     : {CLASS_MAP}")
    print(f"==========================================")

    # Validate deepship folder
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Required source directory '{SOURCE_DIR}' not found.")
        return

    for class_id in [0, 1, 2, 3]:
        os.makedirs(os.path.join(TARGET_DIR, str(class_id)), exist_ok=True)

    all_files = []
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if os.path.isdir(class_dir):
            files = glob.glob(os.path.join(class_dir, '**', '*.wav'), recursive=True)
            for f in files:
                all_files.append((f, class_id))

    # Stratified Split
    random.seed(args.seed)
    files_by_class = collections.defaultdict(list)
    for f, cid in all_files:
        files_by_class[cid].append((f, cid))

    train_files_source = []
    test_files_source = []

    source_file_counts = { 'train': {0:0, 1:0, 2:0, 3:0}, 'test': {0:0, 1:0, 2:0, 3:0} }

    for cid in sorted(files_by_class.keys()):
        files = files_by_class[cid]
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * 0.8)

        train_files_source.extend(files[:n_train])
        test_files_source.extend(files[n_train:])
        source_file_counts['train'][cid] = n_train
        source_file_counts['test'][cid] = n_total - n_train

    random.shuffle(train_files_source)
    random.shuffle(test_files_source)

    # --- TRAIN GENERATION ---
    class_segments_train = collections.defaultdict(list)
    segments_per_file_train = {}

    for file_path, class_id in tqdm(train_files_source, desc="Train Files"):
        segments = process_file(
            file_path, TARGET_DIR, class_id,
            is_train=True,
            overlap_ratios=CLASS_OVERLAP_RATIOS,
            rms_threshold=args.rms_threshold
        )

        # Max Segs Per File Cap (D5)
        if args.max_segs_per_file > 0 and len(segments) > args.max_segs_per_file:
            segments = random.sample(segments, args.max_segs_per_file)

        segments_per_file_train[file_path] = len(segments)
        for seg_path, cid in segments:
            class_segments_train[cid].append(f"{seg_path}\t{cid}\n")

    raw_counts = {cid: len(lines) for cid, lines in class_segments_train.items()}

    # Segment-level Undersampling
    if not args.no_undersample and len(raw_counts) > 0:
        min_count = min(raw_counts.values())
        print(f"\nUndersampling ALL groups down to min count: {min_count}")
        for cid in class_segments_train:
            if len(class_segments_train[cid]) > min_count:
                class_segments_train[cid] = random.sample(class_segments_train[cid], min_count)
    else:
        print("\nUndersampling skipped or non-applicable.")

    # Write Train File
    all_train_lines = []
    for cid in class_segments_train:
        all_train_lines.extend(class_segments_train[cid])
    random.shuffle(all_train_lines)

    with open(TRAIN_LIST_PATH, 'w') as f_out:
        f_out.writelines(all_train_lines)

    train_counts = {cid: len(lines) for cid, lines in class_segments_train.items()}

    # --- TEST GENERATION ---
    test_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    segments_per_file_test = {}

    with open(TEST_LIST_PATH, 'w') as f_out:
        for file_path, class_id in tqdm(test_files_source, desc="Test Files"):
            segments = process_file(
                file_path, TARGET_DIR, class_id,
                is_train=False,
                rms_threshold=args.rms_threshold
            )
            segments_per_file_test[file_path] = len(segments)
            for seg_path, cid in segments:
                f_out.write(f"{seg_path}\t{cid}\n")
                test_counts[cid] += 1

    # --- SANITY CHECKS ---
    print("\n" + "="*55)
    print("SANITY CHECK & DATA DISTRIBUTION (4-CLASS)")
    print("="*55)
    print(f"1. Source File Counts (Before Slicing):")
    print(f"   Train : {source_file_counts['train']}")
    print(f"   Test  : {source_file_counts['test']}")

    def print_dist(name, stat_dict):
        vals = list(stat_dict.values())
        if vals:
            print(f"   {name:<6} -> Min: {min(vals):>3}, Median: {statistics.median(vals):>5.1f}, Max: {max(vals):>4}")

    print("\n2. Segment Counts Per File:")
    print_dist("Train", segments_per_file_train)
    print_dist("Test", segments_per_file_test)

    print("\n3. Class Distribution (Train Segments):")
    total_train = sum(train_counts.values())
    for cid in sorted(train_counts.keys()):
        pct = (train_counts[cid] / total_train * 100) if total_train > 0 else 0
        print(f"   Class {cid}: {train_counts[cid]:>5} ({pct:.1f}%)")

    print("\n4. Class Distribution (Test Segments):")
    total_test = sum(test_counts.values())
    for cid in sorted(test_counts.keys()):
        pct = (test_counts[cid] / total_test * 100) if total_test > 0 else 0
        print(f"   Class {cid}: {test_counts[cid]:>5} ({pct:.1f}%)")

    print("\n5. Current Final Class Weights (for CrossEntropyLoss if needed):")
    counts_tensor = torch.tensor([train_counts.get(i, 0) for i in range(4)], dtype=torch.float)
    if counts_tensor.min() > 0:
        weights = counts_tensor.sum() / (len(counts_tensor) * counts_tensor)
        w_str = ", ".join(f"{w:.4f}" for w in weights.tolist())
        print(f"   class_weights = torch.tensor([{w_str}])")

    print("\nList files saved to:")
    print(f"   Train: {TRAIN_LIST_PATH}")
    print(f"   Test:  {TEST_LIST_PATH}")


if __name__ == '__main__':
    main()
