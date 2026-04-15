import os
import glob
import librosa
import librosa.display
import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
import soundfile as sf
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # Support Chinese if needed
plt.rcParams['axes.unicode_minus'] = False
import random

# Common Config (matches yours)
SR = 16000
SEGMENT_SAMPLES = SR * 5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 160
F_MIN = 20
F_MAX = 3000
RMS_THRESH = 0.005

def calc_rms(y):
    return np.sqrt(np.mean(y**2))

def find_file_with_mixed_vad():
    files = glob.glob('DeepShip-main/**/*.wav', recursive=True)
    random.seed(42) # fixed seed for reproducibility
    random.shuffle(files)
    for f in files:
        y, sr = sf.read(f)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        
        # take first 30 seconds
        y_cut = y[:30*SR]
        # normalize
        max_val = np.max(np.abs(y_cut))
        if max_val > 0:
            y_cut = y_cut / max_val
            
        num_segs = len(y_cut) // SEGMENT_SAMPLES
        rms_vals = []
        for i in range(num_segs):
            seg = y_cut[i*SEGMENT_SAMPLES : (i+1)*SEGMENT_SAMPLES]
            rms_vals.append(calc_rms(seg))
        
        has_valid = any(r >= RMS_THRESH for r in rms_vals)
        has_invalid = any(r < RMS_THRESH for r in rms_vals)
        if has_valid and has_invalid:
            return f, y_cut
    
    # fallback: fabricate one from a valid segment by prepending silence
    print("Fallback: Creating VAD sample manually by prepending silence.")
    for f in files:
        y, sr = sf.read(f)
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        max_val = np.max(np.abs(y))
        if max_val > 0: y = y / max_val
        for i in range(len(y) // SEGMENT_SAMPLES):
            seg = y[i*SEGMENT_SAMPLES : (i+1)*SEGMENT_SAMPLES]
            if calc_rms(seg) >= RMS_THRESH:
                silence = np.random.randn(SEGMENT_SAMPLES) * 0.001
                return f, np.concatenate([silence, seg, silence, seg])
    return None, None

print("Finding suitable file for VAD plot...")
f_vad, y_vad = find_file_with_mixed_vad()
print(f"File for VAD: {f_vad}")

# 1. Figure 1: VAD
if y_vad is not None:
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    time_axis = np.linspace(0, len(y_vad)/SR, len(y_vad))
    
    # Top: Raw
    ax[0].plot(time_axis, y_vad, color='gray', alpha=0.8)
    ax[0].set_title('Original audio waveform (containing long periods of silence or ocean background noise)', fontsize=14)
    ax[0].set_ylabel('Amplitude')
    
    # Bottom: Highlighted
    ax[1].plot(time_axis, y_vad, color='gray', alpha=0.8)
    ax[1].set_title('VAD (RMS threshold=0.005) Truncated valid segments (highlighted in red)', fontsize=14)
    ax[1].set_ylabel('幅值')
    ax[1].set_xlabel('时间 (秒)')
    
    num_segs = len(y_vad) // SEGMENT_SAMPLES
    for i in range(num_segs):
        seg = y_vad[i*SEGMENT_SAMPLES : (i+1)*SEGMENT_SAMPLES]
        rms = calc_rms(seg)
        if rms >= RMS_THRESH:
            start_time = i * 5
            ax[1].axvspan(start_time, start_time + 5, color='red', alpha=0.3, label='Valid segments' if 'Valid segments' not in ax[1].get_legend_handles_labels()[1] else "")
            
    handles, labels = ax[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax[1].legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig('fig1_vad.png', dpi=300)
    plt.close()

# 2. Figure 2: Mel-Spectrogram (2x2 grid)
classes = {'Tanker': 'Oil Tanker', 'Tug': 'Tug', 'Passengership': 'Passenger Ship', 'Cargo': 'Cargo Ship'}
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

def compute_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

clean_segments = {}

for idx, (cls_dir, cls_name) in enumerate(classes.items()):
    files = glob.glob(f'DeepShip-main/{cls_dir}/**/*.wav', recursive=True)
    random.seed(42 + idx)
    random.shuffle(files)
    valid_seg = None
    # find a 5s segment passing VAD
    for f in files:
        y, sr = sf.read(f)
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        max_val = np.max(np.abs(y))
        if max_val > 0:
             y = y / max_val
        num_segs = len(y) // SEGMENT_SAMPLES
        for i in range(num_segs):
            seg = y[i*SEGMENT_SAMPLES : (i+1)*SEGMENT_SAMPLES]
            if calc_rms(seg) >= RMS_THRESH:
                valid_seg = seg
                break
        if valid_seg is not None:
             break
    
    if valid_seg is not None:
        clean_segments[cls_dir] = valid_seg
        mel_db = compute_mel(valid_seg)
        img = librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH, 
                                       fmin=F_MIN, fmax=F_MAX, x_axis='time', y_axis='mel', 
                                       ax=axes[idx], cmap='magma')
        axes[idx].set_title(cls_name, fontsize=14)
        if idx % 2 == 0:
            axes[idx].set_ylabel('频率 (Hz)')
        else:
            axes[idx].set_ylabel('')
        if idx >= 2:
            axes[idx].set_xlabel('时间 (秒)')
        else:
            axes[idx].set_xlabel('')

plt.colorbar(img, ax=axes, format='%+2.0f dB', orientation='horizontal', fraction=0.05, pad=0.1, label='能量 (dB)')
plt.tight_layout()
plt.savefig('fig2_melspec.png', dpi=300)
plt.close()

# 3. Figure 3: Augmentation
# use the Tanker clean segment
y_clean = clean_segments['Tanker']

# Plot A: clean
mel_clean = compute_mel(y_clean)

# Plot B: pink noise (simulated)
white = np.random.randn(len(y_clean))
X = np.fft.rfft(white)
S = np.abs(X)
freqs = np.fft.rfftfreq(len(white))
freqs[0] = 1.0
scale = 1.0 / np.sqrt(freqs)
scale[0] = 0.0
X = X * scale
pink = np.fft.irfft(X, n=len(white))
pink = pink / (np.std(pink) + 1e-9)

# mix
s_power = np.mean(y_clean**2)
snr_db = 15 # SNR 15dB
np_power = s_power / (10**(snr_db/10))
pink = pink * np.sqrt(np_power / np.mean(pink**2))
y_noise = y_clean + pink
max_val = np.max(np.abs(y_noise))
if max_val > 0: y_noise = y_noise/max_val
mel_noise = compute_mel(y_noise)

# Plot C: speed perturbation (using torchaudio resample trick)
speed_factor = 1.1 # 1.1x speed
y_clean_tensor = torch.from_numpy(y_clean).float().unsqueeze(0)
resampler = T.Resample(orig_freq=SR, new_freq=int(SR*speed_factor))
y_speed = resampler(y_clean_tensor).squeeze().numpy()
# Crop to 5s
if len(y_speed) > len(y_clean):
    y_speed = y_speed[:len(y_clean)]
else:
    y_speed = np.pad(y_speed, (0, len(y_clean) - len(y_speed)))
mel_speed = compute_mel(y_speed)


fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# A
imgA = librosa.display.specshow(mel_clean, sr=SR, hop_length=HOP_LENGTH, fmin=F_MIN, fmax=F_MAX, x_axis='time', y_axis='mel', ax=axes[0], cmap='magma')
axes[0].set_title('Figure A: Log-Mel Spectrogram of Original Clean Ship Audio', fontsize=12)
axes[0].set_xlabel('')

# B
librosa.display.specshow(mel_noise, sr=SR, hop_length=HOP_LENGTH, fmin=F_MIN, fmax=F_MAX, x_axis='time', y_axis='mel', ax=axes[1], cmap='magma')
axes[1].set_title('Figure B: Injected Pink Noise (Simulated Complex Ocean Underwater Channel Interference)', fontsize=12)
axes[1].set_xlabel('')

# C
librosa.display.specshow(mel_speed, sr=SR, hop_length=HOP_LENGTH, fmin=F_MIN, fmax=F_MAX, x_axis='time', y_axis='mel', ax=axes[2], cmap='magma')
axes[2].set_title('Figure C: Doppler Shift Effect Induced by Speed Perturbation', fontsize=12)

for ax in axes:
    ax.set_ylabel('frequency (Hz)')
axes[2].set_xlabel('time (seconds)')

plt.colorbar(imgA, ax=axes, format='%+2.0f dB', orientation='vertical', fraction=0.03, pad=0.02)
plt.tight_layout()
plt.savefig('fig3_aug.png', dpi=300)
plt.close()

print("All figures generated: fig1_vad.png, fig2_melspec.png, fig3_aug.png")
