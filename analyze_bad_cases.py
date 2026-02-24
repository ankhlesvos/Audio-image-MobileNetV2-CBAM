import torch
import torch.nn.functional as F
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os

from modules.model import MyNet
from dataset import AudioDataset
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_melspec(mel_tensor, title, save_path):
    # mel_tensor shape: (1, n_mels, time) or (n_mels, time)
    mel = mel_tensor.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_bad_cases():
    config_path = 'configs/ablation_study/M5_MobileNetV2_CBAM_Asym_config.yml'
    model_path = 'saved_models/M5_MobileNetV2_CBAM_Asym/best_model.pth'
    out_dir = Path("bad_cases_analysis")
    out_dir.mkdir(exist_ok=True)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_conf = config['model_conf']
    data_conf = config['data_conf']

    model = MyNet(
        num_classes=model_conf['num_classes'],
        in_channels=model_conf.get('in_channels', 3),
        model_config=model_conf.get('model_config'),
        width_mult=model_conf.get('width_mult', 1.0),
        asymmetric=model_conf.get('asymmetric', False),
        force_no_residual=model_conf.get('force_no_residual', False),
        audio_mode=model_conf.get('audio_mode', False)
    )

    clean_state_dict = {k: v for k, v in torch.load(model_path, map_location=device).items() if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    test_dataset = AudioDataset(data_list_path=data_conf['test_list'], train=False)
    
    cargo_as_tug_count = 0
    tug_as_cargo_count = 0
    correct_cargo_count = 0
    correct_tug_count = 0

    print("Analyzing test set for confusions and gathering references...")
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            mel, label_a, _, _ = test_dataset[i]
            label = label_a.item()
            
            # 仅关注 Cargo (0) 和 Tug (3) 相关的样本
            if label not in [0, 3]:
                continue
                
            outputs = model(mel.unsqueeze(0).to(device))
            scores = F.softmax(outputs, dim=1)
            pred = torch.argmax(scores, dim=1).item()
            pred_score = scores[0, pred].item()

            file_path, _ = test_dataset.lines[i].split('\t')
            file_name = os.path.basename(file_path)

            # --- 收集正确的参照参照 (References) ---
            if label == 0 and pred == 0 and correct_cargo_count < 3:
                plot_melspec(mel, f'[REFERENCE] True: Cargo(0) [{pred_score:.2f}]\n{file_name}', out_dir / f'REFERENCE_cargo_{correct_cargo_count}.png')
                correct_cargo_count += 1
            elif label == 3 and pred == 3 and correct_tug_count < 3:
                plot_melspec(mel, f'[REFERENCE] True: Tug(3) [{pred_score:.2f}]\n{file_name}', out_dir / f'REFERENCE_tug_{correct_tug_count}.png')
                correct_tug_count += 1

            # --- 收集错题 ---
            # Cargo predicted as Tug
            elif label == 0 and pred == 3 and cargo_as_tug_count < 5:
                # Plot
                plot_melspec(mel, f'True: Cargo(0), Pred: Tug(3) [{pred_score:.2f}]\n{file_name}', out_dir / f'cargo_as_tug_{cargo_as_tug_count}.png')
                cargo_as_tug_count += 1
                
            # Tug predicted as Cargo
            elif label == 3 and pred == 0 and tug_as_cargo_count < 5:
                plot_melspec(mel, f'True: Tug(3), Pred: Cargo(0) [{pred_score:.2f}]\n{file_name}', out_dir / f'tug_as_cargo_{tug_as_cargo_count}.png')
                tug_as_cargo_count += 1
                
            if cargo_as_tug_count >= 5 and tug_as_cargo_count >= 5 and correct_cargo_count >= 3 and correct_tug_count >= 3:
                break
                
    print(f"Analysis complete.")
    print(f"Generated {correct_cargo_count} Cargo references, {correct_tug_count} Tug references.")
    print(f"Plotted {cargo_as_tug_count} Cargo->Tug and {tug_as_cargo_count} Tug->Cargo cases.")
    print(f"Check the '{out_dir}' directory for visualizations.")

if __name__ == '__main__':
    analyze_bad_cases()
