# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import os
from pathlib import Path

from dataset import AudioDataset
from modules.model import MyNet


def train(config_path: str):
    """
    主训练函数。
    :param config_path: 配置文件路径。
    """
    # --- 1. 加载配置 ---
    print("--- 1. 加载配置 ---")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_conf = config['train_conf']
    data_conf = config['data_conf']
    model_conf = config['model_conf']

    print("配置加载成功:")
    print(config)

    # --- 2. 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() and train_conf['use_gpu'] else "cpu")
    print(f"\n--- 2. 使用设备: {device} ---")

    # --- 3. 准备数据 ---
    print("\n--- 3. 准备数据 ---")
    train_dataset = AudioDataset(data_list_path=data_conf['train_list'])
    test_dataset = AudioDataset(data_list_path=data_conf['test_list'])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_conf['batch_size'],
        shuffle=True,
        num_workers=train_conf['num_workers']
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_conf['batch_size'],
        shuffle=False,
        num_workers=train_conf['num_workers']
    )
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # --- 4. 构建模型 ---
    print("\n--- 4. 构建模型 ---")
    model = MyNet(
        num_classes=model_conf['num_classes'],
        model_config=model_conf.get('model_config'),
        width_mult=model_conf.get('width_mult', 1.0),
        in_channels=model_conf.get('in_channels', 1)
    )
    model.to(device)
    print("模型结构:")
    # 简单的模型结构打印
    print(model)

    # --- 5. 定义损失函数和优化器 ---
    print("\n--- 5. 定义损失函数和优化器 ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=train_conf['learning_rate'],
        weight_decay=train_conf.get('weight_decay', 1e-5)
    )
    # 学习率调度器
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # --- 6. 训练与评估循环 ---
    print("\n--- 6. 开始训练与评估 ---")
    best_accuracy = 0.0
    save_dir = Path(train_conf['save_model_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)  # 确保保存目录存在

    for epoch in range(train_conf['max_epoch']):
        # --- 训练阶段 ---
        model.train()
        total_loss = 0
        total_correct = 0

        # 使用tqdm创建进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_conf['max_epoch']} [Train]")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()

            # 更新进度条描述
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_dataset)

        # --- 评估阶段 ---
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{train_conf['max_epoch']} [Eval]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()

        eval_accuracy = total_correct / len(test_dataset)

        print(f"Epoch {epoch + 1}/{train_conf['max_epoch']}: \n"
              f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}\n"
              f"  Eval Acc: {eval_accuracy:.4f}")

        # --- 保存最佳模型 ---
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            best_model_path = save_dir / "best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path} with accuracy: {best_accuracy:.4f}")

        # 更新学习率
        # scheduler.step()

    print("\n--- 训练完成 ---")
    print(f"最佳评估准确率: {best_accuracy:.4f}")


if __name__ == '__main__':
    # 使用argparse来接收配置文件路径，使其更灵活
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/mynet_config.yml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    train(args.config)