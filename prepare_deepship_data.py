
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
TARGET_DIR = 'data/deepship_processed'
TRAIN_LIST_PATH = 'data/train_list.txt'
TEST_LIST_PATH = 'data/test_list.txt'
TARGET_SR = 32000
SEGMENT_DURATION = 3  # Seconds
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_DURATION

# VAD & Cleanup Config
RMS_THRESHOLD = 0.005  # Filter out segments with RMS below this
# Adaptive Overlap Configuration
CLASS_OVERLAP_RATIOS = {
    0: 0.5,   # Cargo (Base)
    1: 0.7,   # Passengership (Target ~1.7x)
    2: 0.55,  # Tanker (Target ~1.1x)
    3: 0.7    # Tug (Target ~3.5-4x - Reduced from 0.85 to prevent heavy overfitting)
}

CLASS_MAP = {
    'Cargo': 0,
    'Passengership': 1,
    'Tanker': 2,
    'Tug': 3
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
    Reads audio, normalizes, inspects for signal (VAD), and slices.
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

    # Resample if necessary (using simple linear interpolation for speed if torchaudio is heavy, 
    # but strictly we should use a proper resampler. 
    # Since we are in PyTorch context, let's use torchaudio if desired, or just rely on 'dataset.py' logic?
    # Actually current code had torchaudio resample. Let's keep it but import it inside or use the one from imports.
    # The original code imported torchaudio.
    if sr != TARGET_SR:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    # Peak Normalize
    waveform = peak_normalize(waveform)

    num_samples = waveform.shape[1]
    
    # Determine stride
    if is_train:
        # Use class-specific overlap ratio if available, else default to 0.5
        overlap = CLASS_OVERLAP_RATIOS.get(class_id, 0.5)
        stride = int(SEGMENT_SAMPLES * (1 - overlap))
    else:
        stride = SEGMENT_SAMPLES

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

    # Create target directories
    for _, class_id in CLASS_MAP.items():
        os.makedirs(os.path.join(TARGET_DIR, str(class_id)), exist_ok=True)

    all_files = []
    print("Scanning files...")
    for class_name, class_id in CLASS_MAP.items():
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"Warning: Class directory '{class_dir}' not found.")
            continue
            
        files = glob.glob(os.path.join(class_dir, '**', '*.wav'), recursive=True)
        print(f"Found {len(files)} files for class {class_name}")
        for f in files:
            all_files.append((f, class_id))

    # Split Stratified
    random.seed(42)
    train_files_source = []
    test_files_source = []
    
    files_by_class = {}
    for f, cid in all_files:
        if cid not in files_by_class:
            files_by_class[cid] = []
        files_by_class[cid].append((f, cid))
        
    print("\nSplitting source files (Stratified):")
    for cid, files in files_by_class.items():
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * 0.8)
        
        train_files_source.extend(files[:n_train])
        test_files_source.extend(files[n_train:])
        
        print(f"  Class {cid}: {n_total} -> Train {n_train}, Test {len(files) - n_train}")
      
    random.shuffle(train_files_source)
    random.shuffle(test_files_source)

    # Statistics trackers
    train_counts = {i: 0 for i in range(4)}
    test_counts = {i: 0 for i in range(4)}

    print("\nProcessing Train Files (With Adaptive Overlap & VAD)...")
    print(f"Overlap Ratios: {CLASS_OVERLAP_RATIOS}")
    with open(TRAIN_LIST_PATH, 'w') as f_out:
        for file_path, class_id in tqdm(train_files_source):
            segments = process_file(file_path, TARGET_DIR, class_id, is_train=True)
            for seg_path, cid in segments:
                f_out.write(f"{seg_path}\t{cid}\n")
                train_counts[cid] += 1

    print("\nProcessing Test Files (No Overlap, With VAD)...")
    with open(TEST_LIST_PATH, 'w') as f_out:
        for file_path, class_id in tqdm(test_files_source):
            segments = process_file(file_path, TARGET_DIR, class_id, is_train=False)
            for seg_path, cid in segments:
                f_out.write(f"{seg_path}\t{cid}\n")
                test_counts[cid] += 1

    print("\nData preparation complete.")
    print("Segment Statistics:")
    print(f"  Train: {train_counts} (Total: {sum(train_counts.values())})")
    print(f"  Test:  {test_counts} (Total: {sum(test_counts.values())})")
    
    # Check for imbalance
    total_train = sum(train_counts.values())
    if total_train > 0:
        print("\nClass Distribution (Train):")
        for cid, count in train_counts.items():
             print(f"  Class {cid}: {count} ({count/total_train:.2%})")

if __name__ == '__main__':
    main()
