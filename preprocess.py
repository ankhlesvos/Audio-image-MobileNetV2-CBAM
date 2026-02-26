
import torch
import torchaudio
import soundfile as sf
import torchaudio.transforms as T
import numpy as np

# 配置
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 2048,  # Increased for better frequency resolution (approx 7.8Hz per bin)
    "win_length": 2048,  # Match n_fft
    "hop_length": 512,  # ~32ms stride
    "n_mels": 160,  # Increased vertical resolution (User Request: 160 to compensate for bandwidth)
    "f_min": 20,     # Capture low frequencies (Exclude DC/extreme sub-bass)
    "f_max": 3000,  # Focus on ship noise band (Exclude ocean high-freq background)
    "target_db": -20.0,  # 音量归一化分贝值
    "target_length_secs": 5,  # Updated to 5 seconds to match prepare_deepship_data_5s.py
}

AUDIO_CONFIG["target_length_frames"] = int(AUDIO_CONFIG["target_length_secs"] * AUDIO_CONFIG["sample_rate"])

mel_spectrogram_transformer = T.MelSpectrogram(
    sample_rate=AUDIO_CONFIG["sample_rate"],
    n_fft=AUDIO_CONFIG["n_fft"],
    win_length=AUDIO_CONFIG["win_length"],
    hop_length=AUDIO_CONFIG["hop_length"],
    n_mels=AUDIO_CONFIG["n_mels"],
    f_min=AUDIO_CONFIG["f_min"],
    f_max=AUDIO_CONFIG["f_max"],
    normalized=True 
)

def load_audio_waveform(audio_path: str) -> torch.Tensor:
    """Loads audio, resamples, mixes to mono, pads/crops to target length."""
    try:
        # Use soundfile backend directly
        wav_numpy, sr = sf.read(audio_path)
        
        # Safety Check: Limit memory usage (e.g., max 50M samples ~ 200MB float32)
        if wav_numpy.size > 50_000_000:
             print(f"Warning: Audio file {audio_path} is too large ({wav_numpy.size} samples). Skipping.")
             return None

        waveform = torch.from_numpy(wav_numpy).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t()
            
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None

    # Resample
    if sr != AUDIO_CONFIG["sample_rate"]:
        resampler = T.Resample(orig_freq=sr, new_freq=AUDIO_CONFIG["sample_rate"])
        waveform = resampler(waveform)

    # Mix to Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Fix Length (Pad/Crop)
    current_length = waveform.shape[1]
    target_length = AUDIO_CONFIG["target_length_frames"]
    if current_length > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))
        
    return waveform

def waveform_to_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Converts waveform to Log Mel Spectrogram."""
    
    # Peak Normalize (Optional, but good practice before Mel)
    # Note: prepare_deepship_data already does this, but after augmentation we might need re-norm
    # RMS Normalize as per original code
    rms_db = 20 * torch.log10(torch.sqrt(torch.mean(waveform ** 2)) + 1e-9)
    gain = 10 ** ((AUDIO_CONFIG["target_db"] - rms_db) / 20)
    waveform = waveform * gain

    # Mel Spectrogram
    mel_spec = mel_spectrogram_transformer(waveform)
    log_mel_spec = T.AmplitudeToDB()(mel_spec)

    return log_mel_spec

def audio_to_mel_spectrogram(audio_path: str) -> torch.Tensor:
    """Legacy wrapper for compatibility."""
    waveform = load_audio_waveform(audio_path)
    if waveform is None:
        return None
    return waveform_to_mel(waveform)

if __name__ == '__main__':
    # Test
    test_audio_path = 'data/deepship_processed/0/some_file.wav' # Adjust if needed
    print(f"Testing preprocess with config: {AUDIO_CONFIG}")