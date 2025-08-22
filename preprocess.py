# preprocess.py

import torch
import torchaudio
import torchaudio.transforms as T

# 配置
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 512,  # FFT窗口大小
    "win_length": 400,  # 窗口长度 (25ms for 16k SR)
    "hop_length": 160,  # 帧移 (10ms for 16k SR)
    "n_mels": 80,  # 梅尔滤波器数量
    "target_db": -20.0,  # 音量归一化分贝值
    "target_length_secs": 3,  # 目标音频长度（秒）
}

AUDIO_CONFIG["target_length_frames"] = int(AUDIO_CONFIG["target_length_secs"] * AUDIO_CONFIG["sample_rate"])

mel_spectrogram_transformer = T.MelSpectrogram(
    sample_rate=AUDIO_CONFIG["sample_rate"],
    n_fft=AUDIO_CONFIG["n_fft"],
    win_length=AUDIO_CONFIG["win_length"],
    hop_length=AUDIO_CONFIG["hop_length"],
    n_mels=AUDIO_CONFIG["n_mels"]
)


def audio_to_mel_spectrogram(audio_path: str) -> torch.Tensor:
    try:
        #加载音频
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"{e}")
        return None

    #重采样
    if sr != AUDIO_CONFIG["sample_rate"]:
        resampler = T.Resample(orig_freq=sr, new_freq=AUDIO_CONFIG["sample_rate"])
        waveform = resampler(waveform)

    #确保单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    #标准化
    current_length = waveform.shape[1]
    target_length = AUDIO_CONFIG["target_length_frames"]
    if current_length > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding = target_length - current_length
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    #归一化
    rms_db = 20 * torch.log10(torch.sqrt(torch.mean(waveform ** 2)))
    gain = 10 ** ((AUDIO_CONFIG["target_db"] - rms_db) / 20)
    waveform = waveform * gain

    #计算梅尔频谱图
    mel_spec = mel_spectrogram_transformer(waveform)
    log_mel_spec = T.AmplitudeToDB()(mel_spec)

    return log_mel_spec


if __name__ == '__main__':
    # 测试
    test_audio_path = 'data/yourdatapath'

    mel_tensor = audio_to_mel_spectrogram(test_audio_path)

    if mel_tensor is not None:
        print(f"成功: {mel_tensor.shape}")