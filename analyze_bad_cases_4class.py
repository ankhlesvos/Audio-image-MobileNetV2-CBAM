"""
analyze_bad_cases_4class.py

Analyze confusion patterns between all 4 DeepShip classes.
Focus on the hardest confusion pairs, especially Cargo↔Tug.

Labels: 0=Cargo, 1=Passengership, 2=Tanker, 3=Tug
"""

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

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES = {0: "Cargo", 1: "Passengership", 2: "Tanker", 3: "Tug"}


def plot_melspec(mel_tensor, title, save_path):
    mel = mel_tensor.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def analyze_bad_cases(config_path, model_path, out_dir="bad_cases_analysis_4class", max_per_pair=5, max_refs=3):
    out_dir = Path(out_dir)
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

    clean_state_dict = {k: v for k, v in torch.load(model_path, map_location=device).items()
                        if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()

    test_dataset = AudioDataset(data_list_path=data_conf['test_list'], train=False)

    # Track confusion pairs: (true_label, pred_label) -> list of (mel, score, path)
    confusion_pairs = {}
    correct_refs = {}  # true_label -> list of correct predictions

    print(f"Analyzing test set for confusions (4-class DeepShip)...")

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            mel, label_a, _, _ = test_dataset[i]
            label = label_a.item()

            outputs = model(mel.unsqueeze(0).to(device))
            scores = F.softmax(outputs, dim=1)
            pred = torch.argmax(scores, dim=1).item()
            pred_score = scores[0, pred].item()

            file_path, _ = test_dataset.lines[i].split('\t')
            file_name = os.path.basename(file_path)

            pair_key = (label, pred)

            if pred == label:
                # Correct prediction — store as reference
                if label not in correct_refs:
                    correct_refs[label] = []
                if len(correct_refs[label]) < max_refs:
                    plot_melspec(
                        mel,
                        f'[REFERENCE] True: {CLASS_NAMES.get(label, label)} (pred={CLASS_NAMES.get(pred, pred)}) [{pred_score:.2f}]\n{file_name}',
                        out_dir / f'REF_{CLASS_NAMES.get(label, f"class{label}")}_{len(correct_refs[label])}.png'
                    )
                    correct_refs[label].append((mel, pred_score, file_name))
            else:
                # Misclassification — store for analysis
                if pair_key not in confusion_pairs:
                    confusion_pairs[pair_key] = []
                if len(confusion_pairs[pair_key]) < max_per_pair:
                    plot_melspec(
                        mel,
                        f'True: {CLASS_NAMES.get(label, label)}, Pred: {CLASS_NAMES.get(pred, pred)} [{pred_score:.2f}]\n{file_name}',
                        out_dir / f'{CLASS_NAMES.get(label, f"class{label")}_as_{CLASS_NAMES.get(pred, f"class{pred}")}_{len(confusion_pairs[pair_key])}.png'
                    )
                    confusion_pairs[pair_key].append((mel, pred_score, file_name))

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"4-Class Bad Case Analysis Summary")
    print(f"{'=' * 50}")

    print(f"\nCorrect references collected:")
    for label in sorted(correct_refs.keys()):
        name = CLASS_NAMES.get(label, f"Class{label}")
        print(f"  {name}: {len(correct_refs[label])} refs")

    print(f"\nConfusion pairs found:")
    for (true_label, pred_label) in sorted(confusion_pairs.keys()):
        true_name = CLASS_NAMES.get(true_label, f"Class{true_label}")
        pred_name = CLASS_NAMES.get(pred_label, f"Class{pred_label}")
        count = len(confusion_pairs[(true_label, pred_label)])
        print(f"  {true_name} -> {pred_name}: {count} examples plotted")

    # Save summary
    summary_path = out_dir / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("4-Class DeepShip Bad Case Analysis\n")
        f.write("=" * 50 + "\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Model: {model_path}\n\n")

        f.write("Correct References:\n")
        for label in sorted(correct_refs.keys()):
            name = CLASS_NAMES.get(label, f"Class{label}")
            f.write(f"  {name}: {len(correct_refs[label])} refs\n")

        f.write("\nConfusion Pairs:\n")
        for (true_label, pred_label) in sorted(confusion_pairs.keys()):
            true_name = CLASS_NAMES.get(true_label, f"Class{true_label}")
            pred_name = CLASS_NAMES.get(pred_label, f"Class{pred_label}")
            count = len(confusion_pairs[(true_label, pred_label)])
            f.write(f"  {true_name} -> {pred_name}: {count} examples plotted\n")

    print(f"\nResults saved to: {out_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/deepship_4class/M5_Baseline_config.yml',
                        help='Config path')
    parser.add_argument('-m', '--model', required=True, help='Model checkpoint path')
    parser.add_argument('-o', '--out_dir', default='bad_cases_analysis_4class', help='Output directory')
    args = parser.parse_args()

    analyze_bad_cases(args.config, args.model, args.out_dir)
