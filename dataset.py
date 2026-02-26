# dataset.py

import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np
import random
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

from preprocess import load_audio_waveform, waveform_to_mel, AUDIO_CONFIG


class AddBackgroundNoise:
    """Adds synthetic Pink Noise to the waveform."""
    def __init__(self, target_snr_db_low=10, target_snr_db_high=30, p=0.5):
        self.target_snr_db_low = target_snr_db_low
        self.target_snr_db_high = target_snr_db_high
        self.p = p

    def __call__(self, waveform):
        if random.random() > self.p:
            return waveform

        noise = self._generate_pink_noise(waveform.shape)
        
        # Calculate signal power
        s_power = torch.mean(waveform ** 2)
        
        if s_power == 0:
            return waveform
            
        # Target SNR
        target_snr_db = random.uniform(self.target_snr_db_low, self.target_snr_db_high)
        target_snr = 10 ** (target_snr_db / 10)
        
        # Calculate noise power needed
        n_power = s_power / target_snr
        
        # Scale noise
        noise_power = torch.mean(noise ** 2)
        if noise_power == 0:
            return waveform
            
        scale = torch.sqrt(n_power / noise_power)
        noise = noise * scale
        
        return waveform + noise

    def _generate_pink_noise(self, shape):
        """Generates pink noise (1/f) using RFFT."""
        # Simple approximation
        # Generate white noise
        white = torch.randn(shape)
        
        # FFT
        X = torch.fft.rfft(white, dim=-1)
        
        # 1/f filter
        S = torch.abs(X) ** 2
        freqs = torch.fft.rfftfreq(shape[-1])
        # Avoid division by zero at DC
        freqs[0] = 1.0 
        
        scale = 1.0 / torch.sqrt(freqs)
        scale[0] = 0.0 # No DC component preferred for audio
        
        X = X * scale
        
        # IFFT
        pink = torch.fft.irfft(X, n=shape[-1], dim=-1)
        
        # Normalize to similar amplitude as white noise
        pink = pink / (torch.std(pink) + 1e-9)
        return pink


class SpeedPerturbation:
    """Changes speed (and pitch) of the audio."""
    def __init__(self, orig_freq, speed_range=(0.9, 1.1), p=0.5):
        self.orig_freq = orig_freq
        self.speed_range = speed_range
        self.p = p

    def __call__(self, waveform):
        if random.random() > self.p:
            return waveform
            
        # Use discrete steps to avoid creating infinite Resample kernels (which might cache and OOM)
        # 0.9, 0.95, 1.0, 1.05, 1.1
        steps = np.arange(self.speed_range[0], self.speed_range[1] + 0.01, 0.05)
        speed_factor = random.choice(steps)
        
        new_freq = int(self.orig_freq * speed_factor)
        # Cache resamplers? Or just rely on discrete values limiting the internal cache of torchaudio if strictly implemented.
        # Ideally, we should use torchaudio.functional.resample but it's not imported.
        # We stick to T.Resample but with limited unique pairs of (orig, new).
        
        resampler = T.Resample(orig_freq=self.orig_freq, new_freq=new_freq)
        return resampler(waveform)


class AudioDataset(Dataset):

    def __init__(self, data_list_path: str, train: bool = True):
        super().__init__()
        self.lines: List[str] = self._load_data_list(data_list_path)
        self.train = train

        # Initialize SpecAugment for training (applied on Mel)
        if self.train:
            # Waveform transformations
            self.add_noise = AddBackgroundNoise(p=0.5)
            self.speed_perturb = SpeedPerturbation(orig_freq=AUDIO_CONFIG["sample_rate"], p=0.3)
            
            # Augmentation Probabilities
            self.mosaic_prob = 0.4  # Probability to apply Mosaic
            self.mixup_prob = 0.4   # Probability to apply Mixup
            
            # Build label to indices map for Mosaic/Mixup
            self.label_to_indices = {}
            for i, line in enumerate(self.lines):
                try:
                    _, label = line.split('\t')
                    label = int(label)
                    if label not in self.label_to_indices:
                        self.label_to_indices[label] = []
                    self.label_to_indices[label].append(i)
                except:
                    continue


    def _load_data_list(self, data_list_path: str) -> List[str]:
        with open(data_list_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        print(f"成功从 {data_list_path} 加载了 {len(lines)} 条数据。")
        return lines

    def __len__(self) -> int:
        return len(self.lines)

        return mel_spectrogram, label_tensor

    def _get_sample_by_idx(self, idx):
        """Helper to load a sample by index."""
        try:
            line = self.lines[idx]
            split_line = line.split('\t')
            audio_path = split_line[0]
            label = int(split_line[1])
            waveform = load_audio_waveform(audio_path)
            return waveform, label
        except Exception:
            return None, None

    def _apply_mosaic(self, main_waveform, main_label):
        """
        Applies Audio Mosaic: stitches 4 segments (0.75s each) of the same class.
        Includes Cross-fading (approx 10ms) to avoid clicks.
        """
        # Target: 3 seconds. 4 segments -> 0.75s each.
        # However, we need slightly more for cross-fading.
        # Let's say we take 4 random samples of the SAME class.
        
        target_len_frames = AUDIO_CONFIG["target_length_frames"]
        segment_len = target_len_frames // 4
        cross_fade_len = int(0.01 * AUDIO_CONFIG["sample_rate"]) # 10ms
        
        # We need 4 samples. We have 1 (main). Pick 3 more.
        indices = self.label_to_indices.get(main_label, [])
        if len(indices) < 4:
            return main_waveform # Not enough samples
            
        others = random.sample(indices, 3)
        waveforms = [main_waveform]
        for idx in others:
            w, l = self._get_sample_by_idx(idx)
            if w is None:
                return main_waveform # Fallback
            waveforms.append(w)
            
        random.shuffle(waveforms)
        
        # Crop to segment_len + cross_fade_len (except for last, or handle overlaps)
        # Strategy: Create a blank buffer, place segments with overlap, cross-fade.
        
        final_wave = torch.zeros(1, target_len_frames)
        
        # Simple concatenation with cross-fade
        # Seg 1: 0 -> 0.75s
        # Seg 2: 0.75s -> 1.5s (but we start slightly earlier at 0.75 - fade/2 ?)
        # Easier: Just cut strictly and cross-fade edges?
        # Actually standard Mosaic puts them in a grid (2x2 image). For audio, creating a sequence is better mimicking time structure.
        
        # Simplified Mosaic for Audio: Concatenate 4 random chunks.
        current_pos = 0
        
        # Create fade-in / fade-out windows
        fade_in = torch.linspace(0, 1, cross_fade_len)
        fade_out = torch.linspace(1, 0, cross_fade_len)
        
        # We need to construct the full tensor.
        # To do it properly with crossfade:
        # wave 1: [chunk]...
        # wave 2:       [chunk]...
        # overlap zone:  [x]
        
        # Let's adjust segment_len to exactly fill 3s. 3s * 16000 = 48000. /4 = 12000.
        
        stitching_buffer = []  # List of chunks
        
        for i, wav in enumerate(waveforms):
            # Extract a random chunk of length 'segment_len' from the source wav
            if wav.shape[1] < segment_len:
                 # Pad if too short
                 wav = torch.nn.functional.pad(wav, (0, segment_len - wav.shape[1]))
            
            # Random crop
            max_start = wav.shape[1] - segment_len
            start = random.randint(0, max(0, max_start))
            chunk = wav[:, start : start + segment_len]
            stitching_buffer.append(chunk)

        # Stitch with cross-fade
        # Since we cut exactly to quarters, standard concat would be clicky.
        # We will apply a tiny fade out/in at the boundaries of the CHUNKS themselves, 
        # then concat. This is technically not "overlapping" the signals, but "smoothing the cut".
        # True cross-fade requires getting more data.
        
        # Let's use the valid "smoothing the cut" approach for simplicity and speed (no re-fetching).
        for i in range(4):
            chunk = stitching_buffer[i]
            # Fade in start (except first block)
            if i > 0:
                chunk[:, :cross_fade_len] *= fade_in
            # Fade out end (except last block)
            if i < 3:
                chunk[:, -cross_fade_len:] *= fade_out
            stitching_buffer[i] = chunk
            
        final_wave = torch.cat(stitching_buffer, dim=1)
        
        # Verify length (might be off by a few points due to rounding if any)
        if final_wave.shape[1] != target_len_frames:
             final_wave = torch.nn.functional.pad(final_wave, (0, target_len_frames - final_wave.shape[1]))
             
        return final_wave

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        return self._getitem_impl(idx, recursion_depth=0)

    def _getitem_impl(self, idx: int, recursion_depth: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        # Safety break
        if recursion_depth > 10:
             # Return a dummy zero sample to prevent crash
             print(f"Error: Failed to load sample after 10 retries at idx {idx}. Returning zeros.")
             target_len = AUDIO_CONFIG["target_length_frames"]
             dummy_mel = torch.zeros((AUDIO_CONFIG["n_mels"], target_len // AUDIO_CONFIG["hop_length"] + 1))
             return dummy_mel, torch.tensor(0), torch.tensor(0), 1.0

        # 1. Parse basics
        waveform, label = self._get_sample_by_idx(idx)
        if waveform is None:
             # Try next one
             return self._getitem_impl((idx + 1) % len(self), recursion_depth + 1)
        
        # Initialize return values for Mixup
        label_a = label
        label_b = label
        lam = 1.0
        
        # 2. Augmentations (Train only)
        if self.train:
            # A. MOSAIC
            if random.random() < self.mosaic_prob:
                waveform = self._apply_mosaic(waveform, label)
                # Keep label_a = label_b = label, lam = 1.0 (Same class)
            
            # B. Standard Waveform Augs
            waveform = self.add_noise(waveform)
            waveform = self.speed_perturb(waveform)
            
            # Re-fix length
            target_len = AUDIO_CONFIG["target_length_frames"]
            if waveform.shape[1] != target_len:
                if waveform.shape[1] > target_len:
                    waveform = waveform[:, :target_len]
                else:
                    padding = target_len - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))

            # C. MIXUP
            # Apply Mixup AFTER other augs, but BEFORE Mel (usually) or AFTER Mel?
            # User asked for "Audio Mixup" (waveform mixing).
            if random.random() < self.mixup_prob:
                # Pick random other sample
                rand_idx = random.randint(0, len(self.lines) - 1)
                other_wav, other_label = self._get_sample_by_idx(rand_idx)
                
                if other_wav is not None:
                    # Apply same base augs to other wav? Maybe just raw for diversity.
                    # Ensure length matches
                    if other_wav.shape[1] != target_len:
                         if other_wav.shape[1] > target_len:
                            other_wav = other_wav[:, :target_len]
                         else:
                            other_wav = torch.nn.functional.pad(other_wav, (0, target_len - other_wav.shape[1]))
                    
                    lam = np.random.beta(1.0, 1.0) # Standard beta distribution
                    waveform = lam * waveform + (1 - lam) * other_wav
                    label_b = other_label
                    # label_a is already set to 'label'
                
        # 4. Convert to Mel Spectrogram
        mel_spectrogram = waveform_to_mel(waveform)

        # 6. Return Tensors
        # Return signature: (input, label_a, label_b, lam)
        return mel_spectrogram, torch.tensor(label_a, dtype=torch.long), torch.tensor(label_b, dtype=torch.long), lam


if __name__ == '__main__':
    # Test Class
    test_list_path = 'data/train_list.txt' # Ensure this exists or use a dummy
    
    # Create a dummy entry if file missing to test logic
    import os
    if not os.path.exists(test_list_path):
        os.makedirs("data", exist_ok=True)
        with open(test_list_path, "w") as f:
            f.write("dummy_path.wav\t0\n")
            
    print("\n--- 开始测试 AudioDataset ---")
    
    # Needs a real file to not fail load_audio_waveform
    # We will just print instantiation for now
    dataset = AudioDataset(data_list_path=test_list_path, train=True)
    print(f"Dataset initialized. Length: {len(dataset)}")
    
    # Test Noise Generator
    print("Testing Pink Noise Generator...")
    noise_gen = AddBackgroundNoise()
    dummy_wav = torch.randn(1, 16000*3)
    noisy_wav = noise_gen(dummy_wav)
    print(f"Original std: {dummy_wav.std()}, Noisy std: {noisy_wav.std()}")
    
    print("dataset.py checks passed (logic only).")