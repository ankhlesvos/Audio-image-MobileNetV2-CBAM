# test.py

import torch
import yaml
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules.model import MyNet
from image_dataset import create_image_dataset
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
        width_mult=model_conf.get('width_mult', 1.0)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型 {model_path} 加载成功，并设置为评估模式。")

    print("\n2. 准备测试数据")
    image_size = data_conf.get('image_size', 32)
    test_dataset = create_image_dataset(data_path=data_conf['test_dir'], is_train=False, image_size=image_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['train_conf']['batch_size'], shuffle=False)

    print("\n3. 在测试集上进行预测")
    all_labels = []
    all_preds = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)

            scores = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_scores = np.array(all_scores)

    print("\n性能评估结果")

    # 总体准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {accuracy:.4f}")

    class_names = [line.strip() for line in open(Path(data_conf['test_dir']).parent / 'label_list.txt')]

    p, r, f1, s = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(len(class_names)))

    print("\nClassification Report:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {p[i]:<10.4f} {r[i]:<10.4f} {f1[i]:<10.4f}")

    # 计算宏平均 (Macro Average)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    print("-" * 50)
    print(f"{'Macro Avg':<15} {macro_p:<10.4f} {macro_r:<10.4f} {macro_f1:<10.4f}")

    report_save_path = Path(model_path).parent / "classification_report.txt"

    with open(report_save_path, 'w', encoding='utf-8') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write("Classification Report:\n")
        f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
        f.write("-" * 50 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<15} {p[i]:<10.4f} {r[i]:<10.4f} {f1[i]:<10.4f}\n")

        f.write("-" * 50 + "\n")
        f.write(f"{'Macro Avg':<15} {macro_p:<10.4f} {macro_r:<10.4f} {macro_f1:<10.4f}\n")

    print(f"分类报告已保存到: {report_save_path}")

    #绘制并保存PR曲线和ROC曲线
    print("\n--- 5. 正在生成PR和ROC曲线图 ---")

    y_one_hot = np.eye(len(class_names))[all_labels]
    precision, recall, _ = precision_recall_curve(y_one_hot.ravel(), all_scores.ravel())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, marker='.')
    plt.title('Macro-average Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_one_hot.ravel(), all_scores.ravel())
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Macro-average ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    plt.tight_layout()
    save_path = Path(model_path).parent / "performance_curves.png"
    plt.savefig(save_path)
    print(f"曲线图已保存到: {save_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to the configuration file used for training.')
    parser.add_argument('-m', '--model', required=True, help='Path to the saved best_model.pth file.')
    args = parser.parse_args()

    test(args.config, args.model)