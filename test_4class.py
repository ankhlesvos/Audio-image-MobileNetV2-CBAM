# test_4class.py
# ============================================================================
# 4-Class DeepShip Standalone Test Script
# Labels: 0=Cargo, 1=Passengership, 2=Tanker, 3=Tug
# ============================================================================

import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re

from modules.model import MyNet
from dataset import AudioDataset
from torch.utils.data import DataLoader

# 4-class DeepShip label mapping
CLASS_NAMES_4CLASS = ["Cargo", "Passengership", "Tanker", "Tug"]


def test(config_path: str, model_path: str,
         use_teacher: bool = False,
         teacher_num_mel_bins: int = 160,
         teacher_max_length: int = 157,
         class_names: list = None):
    if class_names is None:
        class_names = CLASS_NAMES_4CLASS

    num_classes = len(class_names)

    # Load config and model
    print("1. Loading config and model")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_conf = config['model_conf']
    data_conf = config['data_conf']

    if use_teacher:
        from modules.teacher_model import ASTTeacherModel
        teacher_conf = config.get('teacher_conf', {})
        model = ASTTeacherModel(
            num_classes=model_conf['num_classes'],
            pretrained_name=teacher_conf.get(
                'pretrained_name', 'MIT/ast-finetuned-audioset-10-10-0.4593'),
            num_mel_bins=teacher_num_mel_bins,
            max_length=teacher_max_length,
        )
        print(f"[Teacher mode] ASTTeacherModel (mel={teacher_num_mel_bins}, T={teacher_max_length})")
    else:
        model = MyNet(
            num_classes=model_conf['num_classes'],
            in_channels=model_conf.get('in_channels', 3),
            model_config=model_conf.get('model_config'),
            width_mult=model_conf.get('width_mult', 1.0),
            asymmetric=model_conf.get('asymmetric', False),
            multiscale=model_conf.get('multiscale', False),
            force_no_residual=model_conf.get('force_no_residual', False),
            audio_mode=model_conf.get('audio_mode', False)
        )

    state_dict = torch.load(model_path, map_location=device)
    clean_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model {model_path} loaded successfully, set to eval mode.")

    print("\n2. Preparing test data")
    test_dataset = AudioDataset(data_list_path=data_conf['test_list'], train=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config['train_conf']['batch_size'],
        shuffle=False,
        num_workers=config['train_conf']['num_workers']
    )

    # Calculate train prior for Logit Adjustment if needed
    train_conf = config.get('train_conf', {})
    loss_conf = train_conf.get('loss_conf', {})
    has_logit_adj = loss_conf.get('logit_adjustment', False) or loss_conf.get('apply_logit_adj_in_eval', False)
    if has_logit_adj:
        print("Calculating training priors for Logit Adjustment at test time...")
        temp_train_dataset = AudioDataset(data_list_path=data_conf['train_list'], train=False)
        class_counts = {}
        for line in temp_train_dataset.lines:
            try:
                _, label = line.split('\t')
                label = int(label)
                class_counts[label] = class_counts.get(label, 0) + 1
            except:
                pass
        counts = [class_counts.get(i, 0) for i in range(num_classes)]
        sum_counts = sum(counts) if sum(counts) > 0 else 1
        log_prior = torch.log(torch.tensor(counts).float() / sum_counts + 1e-8).to(device)
    else:
        log_prior = None

    print("\n3. Running predictions on test set")

    # Store results for voting
    file_predictions = {}  # {filename_id: [preds...]}
    file_labels = {}       # {filename_id: label}

    all_labels = []
    all_preds = []
    all_scores = []

    global_idx = 0

    with torch.no_grad():
        for mel, label_a, label_b, lam in tqdm(test_loader, desc="Testing"):
            mel = mel.to(device)
            if model_conf.get('in_channels', 3) == 1 and mel.size(1) == 3:
                mel = mel[:, 0:1, :, :]

            outputs = model(mel)

            # Application of Logit Adjustment
            if has_logit_adj and log_prior is not None:
                tau = loss_conf.get('logit_adj_tau', 1.0)
                outputs = outputs + tau * log_prior

            scores = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            current_labels = label_a.cpu().numpy()
            current_preds = preds.cpu().numpy()
            current_scores = scores.cpu().numpy()

            all_labels.extend(current_labels)
            all_preds.extend(current_preds)
            all_scores.extend(current_scores)

            # File-Level Voting Logic
            batch_size = mel.size(0)
            for i in range(batch_size):
                if global_idx >= len(test_dataset.lines):
                    break

                line = test_dataset.lines[global_idx]
                path, str_label = line.split('\t')
                true_label = int(str_label)

                filename = os.path.basename(path)
                match = re.match(r'(.+)_seg\d+\.wav$', filename)
                if match:
                    file_id = match.group(1)
                else:
                    file_id = filename

                if file_id not in file_predictions:
                    file_predictions[file_id] = []
                    file_labels[file_id] = true_label

                file_predictions[file_id].append(current_preds[i])
                global_idx += 1

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)

    print("\nSegment Level Performance")
    seg_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Segment Accuracy: {seg_accuracy:.4f}")

    # --- File-Level Metrics ---
    print("\n--- File-Level Performance (Majority Voting) ---")
    file_gt = []
    file_pred_vote = []

    for fid, preds_list in file_predictions.items():
        counts = np.bincount(preds_list, minlength=num_classes)
        winner = np.argmax(counts)
        file_gt.append(file_labels[fid])
        file_pred_vote.append(winner)

    file_acc = accuracy_score(file_gt, file_pred_vote)
    print(f"File-Level Accuracy: {file_acc:.4f}")

    p, r, f1, s = precision_recall_fscore_support(file_gt, file_pred_vote, average=None, labels=range(num_classes))

    print("\nClassification Report (File-Level):")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 68)
    for i, class_name in enumerate(class_names):
        supp = s[i] if s is not None and i < len(s) and s[i] is not None else 0
        print(f"{class_name:<20} {p[i]:<12.4f} {r[i]:<12.4f} {f1[i]:<12.4f} {supp:<8}")

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(file_gt, file_pred_vote, average='macro')
    print("-" * 68)
    print(f"{'Macro Avg':<20} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f}")

    report_save_path = Path(model_path).parent / "classification_report.txt"

    with open(report_save_path, 'w', encoding='utf-8') as f:
        f.write(f"File-Level Accuracy: {file_acc:.4f}\n")
        f.write(f"Segment Accuracy: {seg_accuracy:.4f}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Class Names: {', '.join(class_names)}\n")
        f.write("Classification Report (File-Level):\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}\n")
        f.write("-" * 68 + "\n")
        for i, class_name in enumerate(class_names):
            supp = s[i] if s is not None and i < len(s) and s[i] is not None else 0
            f.write(f"{class_name:<20} {p[i]:<12.4f} {r[i]:<12.4f} {f1[i]:<12.4f} {supp:<8}\n")

        f.write("-" * 68 + "\n")
        f.write(f"{'Macro Avg':<20} {macro_p:<12.4f} {macro_r:<12.4f} {macro_f1:<12.4f}\n")

    print(f"Classification report saved to: {report_save_path}")

    # Plot Confusion Matrix (File-Level)
    print("\n--- 6. Generating confusion matrix ---")
    cm = confusion_matrix(file_gt, file_pred_vote, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (File-Level, {num_classes}-Class DeepShip)')

    cm_save_path = Path(model_path).parent / "confusion_matrix.png"
    plt.savefig(cm_save_path)
    print(f"Confusion matrix saved to: {cm_save_path}")

    # Also save results.txt in standardized format
    results_save_path = Path(model_path).parent / "results.txt"
    with open(results_save_path, 'w') as f:
        f.write(f"Best Val Acc: N/A\n")  # Not available in standalone test
        f.write(f"Best Val F1: N/A\n")
        f.write(f"Test File Acc: {file_acc:.4f}\n")
        f.write(f"Test File F1: {macro_f1:.4f}\n")
        f.write(f"Test Seg  Acc: {seg_accuracy:.4f}\n")
        seg_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f.write(f"Test Seg  F1: {seg_f1:.4f}\n")
    print(f"Results saved to: {results_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to the configuration file used for training.')
    parser.add_argument('-m', '--model', required=True, help='Path to the saved best_model.pth file.')
    parser.add_argument('--teacher', action='store_true',
                        help='Load model as ASTTeacherModel instead of MyNet.')
    parser.add_argument('--teacher_num_mel_bins', type=int, default=160,
                        help='Mel bins used when training the teacher (default: 160).')
    parser.add_argument('--teacher_max_length', type=int, default=157,
                        help='Time frames used when training the teacher (default: 157).')
    parser.add_argument('--class_names', nargs='+', default=None,
                        help='Custom class names. Default: Cargo Passengership Tanker Tug')
    args = parser.parse_args()

    test(args.config, args.model,
         use_teacher=args.teacher,
         teacher_num_mel_bins=args.teacher_num_mel_bins,
         teacher_max_length=args.teacher_max_length,
         class_names=args.class_names)
