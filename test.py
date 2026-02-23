# test.py

import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re

from modules.model import MyNet
from dataset import AudioDataset
from torch.utils.data import DataLoader


def test(config_path: str, model_path: str):
    #加载配置与模型
    print("1. 加载配置与模型")
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
        force_no_residual=model_conf.get('force_no_residual', False)
    )

    state_dict = torch.load(model_path, map_location=device)
    # Filter out 'total_ops' and 'total_params' added by thop
    clean_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    print(f"模型 {model_path} 加载成功，并设置为评估模式。")

    print("\n2. 准备测试数据")
    # Use AudioDataset for consistency with training and improved preprocessing
    test_dataset = AudioDataset(data_list_path=data_conf['test_list'], train=False)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=config['train_conf']['batch_size'], 
        shuffle=False,
        num_workers=config['train_conf']['num_workers']
    )

    print("\n3. 在测试集上进行预测")
    
    # Store results for voting
    file_predictions = {} # {filename_id: [preds...]}
    file_labels = {}     # {filename_id: label}

    all_labels = []
    all_preds = []
    all_scores = []

    global_idx = 0
    
    with torch.no_grad():
        for mel, label_a, label_b, lam in tqdm(test_loader, desc="Testing"):
            mel = mel.to(device)
            # label = label_a.to(device) # In eval, label_a == label
            
            outputs = model(mel)
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
                
                # Identify "Original File" ID
                filename = os.path.basename(path)
                # Heuristic: file_1.wav, file_2.wav -> file
                match = re.match(r'(.+)_\d+\.wav$', filename)
                if match:
                    file_id = match.group(1)
                else:
                    file_id = filename # Fallback
                
                if file_id not in file_predictions:
                    file_predictions[file_id] = []
                    file_labels[file_id] = true_label
                
                file_predictions[file_id].append(current_preds[i])
                global_idx += 1

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)

    print("\n性能评估结果 (Segment Level)")
    seg_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Segment Accuracy: {seg_accuracy:.4f}")

    # --- File-Level Metrics ---
    print("\n--- File-Level Performance (Majority Voting) ---")
    file_gt = []
    file_pred_vote = []
    
    for fid, preds_list in file_predictions.items():
        counts = np.bincount(preds_list)
        winner = np.argmax(counts)
        file_gt.append(file_labels[fid])
        file_pred_vote.append(winner)
        
    file_acc = accuracy_score(file_gt, file_pred_vote)
    print(f"File-Level Accuracy: {file_acc:.4f}")

    class_names = ["Cargo", "Passengership", "Tanker", "Tug"] 
    
    p, r, f1, s = precision_recall_fscore_support(file_gt, file_pred_vote, average=None, labels=range(len(class_names)))

    print("\nClassification Report (File-Level):")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {p[i]:<10.4f} {r[i]:<10.4f} {f1[i]:<10.4f}")

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(file_gt, file_pred_vote, average='macro')
    print("-" * 50)
    print(f"{'Macro Avg':<15} {macro_p:<10.4f} {macro_r:<10.4f} {macro_f1:<10.4f}")

    report_save_path = Path(model_path).parent / "classification_report.txt"

    with open(report_save_path, 'w', encoding='utf-8') as f:
        f.write(f"File-Level Accuracy: {file_acc:.4f}\n")
        f.write(f"Segment Accuracy: {seg_accuracy:.4f}\n")
        f.write("Classification Report (File-Level):\n")
        f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
        f.write("-" * 50 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<15} {p[i]:<10.4f} {r[i]:<10.4f} {f1[i]:<10.4f}\n")

        f.write("-" * 50 + "\n")
        f.write(f"{'Macro Avg':<15} {macro_p:<10.4f} {macro_r:<10.4f} {macro_f1:<10.4f}\n")

    print(f"分类报告已保存到: {report_save_path}")

    # Plot Confusion Matrix (File-Level)
    print("\n--- 6. 正在生成混淆矩阵 ---")
    cm = confusion_matrix(file_gt, file_pred_vote)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    plt.figure(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (File-Level)')
    
    cm_save_path = Path(model_path).parent / "confusion_matrix.png"
    plt.savefig(cm_save_path)
    print(f"混淆矩阵已保存到: {cm_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to the configuration file used for training.')
    parser.add_argument('-m', '--model', required=True, help='Path to the saved best_model.pth file.')
    args = parser.parse_args()

    test(args.config, args.model)