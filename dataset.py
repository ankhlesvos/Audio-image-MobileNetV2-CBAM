# dataset.py

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

from preprocess import audio_to_mel_spectrogram, AUDIO_CONFIG


class AudioDataset(Dataset):


    def __init__(self, data_list_path: str):

        super().__init__()
        self.lines: List[str] = self._load_data_list(data_list_path)

        # (可选) 在这里可以初始化数据增强的转换器
        # self.augmentations = ...

    def _load_data_list(self, data_list_path: str) -> List[str]:
        with open(data_list_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        print(f"成功从 {data_list_path} 加载了 {len(lines)} 条数据。")
        return lines

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 从列表中解析出音频路径和标签
        try:
            audio_path, label_str = self.lines[idx].split('\t')
            label = int(label_str)
        except ValueError:
            print(f"警告：跳过格式错误的行: {self.lines[idx]}")
            return self.__getitem__((idx + 1) % len(self))

        # 2. 调用预处理函数，将音频路径转换为频谱图
        mel_spectrogram = audio_to_mel_spectrogram(audio_path)

        if mel_spectrogram is None:
            # 如果预处理失败（例如，音频文件损坏），同样跳过这个样本
            print(f"警告：预处理失败，跳过样本: {audio_path}")
            return self.__getitem__((idx + 1) % len(self))

        # 3. (可选) 在这里应用数据增强
        # mel_spectrogram = self.augmentations(mel_spectrogram)

        # 4. 将标签转换为张量
        label_tensor = torch.tensor(label, dtype=torch.long)

        # 返回最终的处理结果
        return mel_spectrogram, label_tensor


if __name__ == '__main__':
    # 再次进行单元测试，确保Dataset类能和preprocess模块协同工作

    test_list_path = 'data/train_list.txt'

    print("\n--- 开始测试 AudioDataset ---")
    # 1. 创建数据集实例
    dataset = AudioDataset(data_list_path=test_list_path)
    print(f"数据集总长度: {len(dataset)}")

    # 2. 尝试获取第一个样本
    if len(dataset) > 0:
        print("\n尝试获取第一个样本...")
        try:
            first_spectrogram, first_label = dataset[0]
            print(f"成功获取样本！")
            print(f"频谱图形状: {first_spectrogram.shape}")
            print(f"标签: {first_label.item()} (类型: {first_label.dtype})")

            # 检查形状和类型是否符合预期
            expected_shape = (1, AUDIO_CONFIG["n_mels"], int(
                AUDIO_CONFIG["target_length_secs"] * AUDIO_CONFIG["sample_rate"] / AUDIO_CONFIG["hop_length"]) + 1)
            print(f"预期频谱图形状: (1, {expected_shape[1]}, {expected_shape[2]})")
            assert first_spectrogram.shape[0] == 1, "通道数应为1"
            assert first_spectrogram.shape[1] == expected_shape[1], "梅尔滤波器数不匹配"
            assert first_label.dtype == torch.long, "标签应为long类型"

            print("\n单元测试通过！`dataset.py` 已准备就绪！")

        except Exception as e:
            print(f"\n单元测试失败！错误信息: {e}")
            import traceback

            traceback.print_exc()