# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
from tqdm import tqdm
import yaml
import os
import random
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

    # --- 1.5 全局随机种子 ---
    seed = train_conf.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Optional: deterministic mode (might slow down convolutions)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- 2. 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() and train_conf['use_gpu'] else "cpu")
    print(f"\n--- 2. 使用设备: {device} ---")

    # --- 3. 准备数据 ---
    print("\n--- 3. 准备数据 ---")
    train_dataset = AudioDataset(data_list_path=data_conf['train_list'], train=True)
    test_dataset = AudioDataset(data_list_path=data_conf['test_list'], train=False)

    # --- Calculate Class Weights (ENS) & Prior ---
    print("Calculating class weights (Effective Number of Samples) and Priors...")
    from collections import Counter
    labels = []
    for line in train_dataset.lines:
        try:
            _, lab = line.split('\t')
            labels.append(int(lab))
        except:
            labels.append(-1)

    # 统计每类样本数（忽略坏样本）
    valid_labels = [l for l in labels if l >= 0]
    class_count = Counter(valid_labels)

    # Sort by key to ensure order [0, 1, 2, 3] etc.
    num_classes = model_conf['num_classes']
    counts = [class_count.get(i, 0) for i in range(num_classes)]
    print(f"Class Counts: {counts}")
    
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, counts)
    # Handle zero counts: precise arithmetic.
    weights = np.zeros_like(effective_num)
    
    valid_classes = effective_num > 1e-6 # float precision
    weights[valid_classes] = (1.0 - beta) / np.array(effective_num)[valid_classes]
    
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights) * num_classes # Normalize so mean weight is 1.0
    else:
        weights = np.ones(num_classes)
    
    # Conditional Class Weights for CE Loss
    if train_conf.get('use_class_weights', True):
        class_weights = torch.tensor(weights).float().to(device)
        print(f"Class Weights (CE Loss): {class_weights}")
    else:
        class_weights = None
        print("Class Weights (CE Loss) are DISABLED.")

    # Log prior for Logit Adjustment
    sum_counts = sum(counts) if sum(counts) > 0 else 1
    prior = np.array(counts, dtype=np.float64) / sum_counts
    log_prior = torch.log(torch.tensor(prior, dtype=torch.float32) + 1e-12).to(device)

    # --- 建立 DataLoader (使用 WeightedRandomSampler) ---
    use_sampler = train_conf.get('use_sampler', False)
    
    if use_sampler and train_conf.get('use_class_weights', False):
        print("\n[WARNING] You have enabled BOTH WeightedRandomSampler and CrossEntropy class_weights. "
              "This duplicates priority over minority classes and can degrade performance. Proceed with caution.\n")
        alpha = train_conf.get("sampler_alpha", 0.5)
        print(f"Building WeightedRandomSampler with alpha={alpha}")
        sample_weights = []
        for l in labels:
            if l < 0:
                sample_weights.append(0.0)
            else:
                sample_weights.append(1.0 / (class_count[l] ** alpha))
                
        sample_weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=len(sample_weights_tensor),
            replacement=True
        )
        shuffle_flag = False
    else:
        print("WeightedRandomSampler is DISABLED. Using standard Shuffle.")
        sampler = None
        shuffle_flag = True

    # Reproducibility
    g = torch.Generator()
    g.manual_seed(train_conf.get("seed", 42))

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_conf['batch_size'],
        shuffle=shuffle_flag,
        sampler=sampler,
        num_workers=train_conf['num_workers'],
        generator=g
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
    width_mult = model_conf.get('width_mult', 1.0)
    print(f"Model Width Mult: {width_mult}")
    model = MyNet(
        num_classes=model_conf['num_classes'],
        model_config=model_conf.get('model_config'),
        width_mult=width_mult,
        in_channels=model_conf.get('in_channels', 1),
        asymmetric=model_conf.get('asymmetric', False),
        multiscale=model_conf.get('multiscale', False),
        force_no_residual=model_conf.get('force_no_residual', False),
        audio_mode=model_conf.get('audio_mode', False)
    )
    model.to(device)
    print("模型结构:")
    # 简单的模型结构打印
    # print(model)


    # Calculate GFLOPs/Params
    try:
        # DeepShip input: (1, 80, 301) for 3 seconds approx? 
        # MelSpec shape: n_mels=80. time steps depends on audio length and hop length.
        # dataset.py uses 3s segments. sample_rate=xxx.
        # Let's assume standard shape used in verifying: (1, 80, 301)
        flops, params = model.profile_model(input_size=(1, 80, 301))
        print(f"FLOPs: {flops / 1e9:.4f} G")
        print(f"Params: {params / 1e6:.4f} M")
    except Exception as e:
        print(f"Profiling failed: {e}")
        flops, params = 0, 0
    
    # --- 5. 定义损失函数和优化器 ---
    print("\n--- 5. 定义损失函数和优化器 ---")
    
    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0, pair_penalty=None):
            super(FocalLoss, self).__init__()
            self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none', label_smoothing=label_smoothing)
            self.gamma = gamma
            self.reduction = reduction
            self.pair_penalty = pair_penalty

        def forward(self, inputs, targets):
            ce_loss = self.ce(inputs, targets)
            pt = torch.exp(-ce_loss)
            
            if self.gamma > 0:
                loss = ((1 - pt) ** self.gamma) * ce_loss
            else:
                loss = ce_loss

            if self.pair_penalty and self.pair_penalty.get('use_penalty', False):
                penalty_weight = self.pair_penalty.get('weight', 1.0)
                penalty_targets = self.pair_penalty.get('targets', [])
                
                probs = torch.softmax(inputs, dim=1)
                penalty = torch.zeros_like(loss)
                
                for true_class, false_class in penalty_targets:
                    mask = (targets == true_class)
                    if mask.any():
                        penalty[mask] += probs[mask, false_class] * penalty_weight
                        
                loss = loss + penalty

            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            return loss

    loss_conf = train_conf.get('loss_conf', {})
    loss_type = loss_conf.get('loss_type', 'focal')
    gamma = loss_conf.get('gamma', 2.0) if loss_type == 'focal' else 0.0
    label_smoothing = loss_conf.get('label_smoothing', 0.1)
    pair_penalty = loss_conf.get('pair_penalty', None)

    # Use reduction='none' here because we manual calculate Mixup loss later
    criterion_none = FocalLoss(weight=class_weights, gamma=gamma, reduction='none', 
                               label_smoothing=label_smoothing, pair_penalty=pair_penalty)

    # --- Freeze Backbone Logic ---
    if train_conf.get('freeze_backbone', False):
        print("\n[INFO] Freezing backbone: Only the classification head will be trained.")
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Build optimizer with only trainable parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = optim.AdamW(
        params=trainable_params,
        lr=train_conf['learning_rate'],
        weight_decay=train_conf.get('weight_decay', 1e-4)
    )
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=train_conf['max_epoch'],
        eta_min=0
    )

    # --- 6. 训练与评估循环 ---
    print("\n--- 6. 开始训练与评估 ---")
    best_accuracy = 0.0
    best_acc_epoch = 0
    best_f1 = 0.0
    best_f1_epoch = 0
    monitor_metric = train_conf.get('monitor_metric', 'f1')
    save_dir = Path(train_conf['save_model_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)  # 确保保存目录存在
    
    # Early stopping config
    patience = 5
    epochs_no_improve = 0

    for epoch in range(train_conf['max_epoch']):
        # --- 训练阶段 ---
        model.train()
        
        # If freezing backbone, keep its BN layers in eval mode
        if train_conf.get('freeze_backbone', False):
            for name, module in model.named_modules():
                if 'classifier' not in name:
                    module.eval()
                    
        total_loss = 0
        total_correct = 0

        # 使用tqdm创建进度条
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_conf['max_epoch']} [Train]")
        for inputs, label_a, label_b, lam in train_pbar:
            inputs = inputs.to(device)
            label_a = label_a.to(device)
            label_b = label_b.to(device)
            # lam is a tensor batch or scalar? 
            # From dataset: lam is float. DataLoader collates floats into a tensor (batch_size,).
            # But wait, dataset returns 'lam' as float. DataLoader will stack them into a double tensor?
            # Actually dataset returns `lam` (float). DataLoader -> Tensor of shape (B,).
            
            # However, if lam is constant 1.0 for some, and random for others...
            # We need to handle 'lam' carefully.
            # Convert to device.
            lam = lam.to(device).float()
            # If lam is a 1D tensor [B], we need to reshape for broadcasting if necessary, 
            # but for scalar multiplication with loss it's fine if loss is reduction='mean'.
            # Wait, nn.CrossEntropyLoss gives scalar by default.
            # We need element-wise loss to apply different lambda per sample?
            # Or is lambda user-mixed? dataset says: `lam` is per sample.
            # So we need `reduction='none'` in criterion? No, standard Mixup usually:
            # loss = lam * loss_a + (1-lam) * loss_b
            # If lam varies per sample, this implies:
            # loss = mean( lam_i * loss_i(a) + (1-lam_i) * loss_i(b) )
            
            # Let's verify `lam`.
            # If `lam` is (B,), checking shape logic.
            # We need item-wise loss.
            
            # OPTION 1: Redefine criterion to reduction='none', calculate manually, then mean.
            # OPTION 2: Use constant lambda per batch? My dataset implementation does Per-Sample Mixup.
            # So I MUST use reduction='none'.
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Logit Adjustment (Train)
            if loss_conf.get('apply_logit_adj_in_train', False):
                tau = loss_conf.get('logit_adj_tau', 1.0)
                outputs = outputs + tau * log_prior

            
            # Calculate loss per sample using Focal Loss (which uses reduction='none' internally)
            # However, our local FocalLoss with reduction='none' could be used:
            # wait, criterion handles focal loss, but we specifically need 'none' for manual mixup mean.
            # Local definition of focal loss with reduction='none':
            loss_a = criterion_none(outputs, label_a)
            loss_b = criterion_none(outputs, label_b)
            
            loss = loss_a * lam + loss_b * (1 - lam)
            loss = loss.mean()
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            # Accuracy metric: Use label_a (dominant) or mixed? 
            # Usually for accuracy we compare to the "heavier" label, or just label_a since lam ~ Beta(1,1) is symmetric but usually > 0.5 rules?
            # In my code: lam = beta(1,1).
            # For tracking, let's use label_a if lam > 0.5 else label_b?
            # Or just use label_a as it's the "original" signal.
            # Simple approach: compare to label_a.
            total_correct += (predicted == label_a).sum().item()

            # 更新进度条描述
            train_pbar.set_postfix(loss=loss.item())


        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_dataset)

        # --- 评估阶段 ---
        model.eval()
        total_correct = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for mel_spectrogram, label_a, label_b, lam in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{train_conf['max_epoch']} [Eval]"):
                # For eval, label_a == label_b. lam == 1.0. 
                mel_spectrogram, labels = mel_spectrogram.to(device), label_a.to(device)
                outputs = model(mel_spectrogram)
                
                # Logit Adjustment (Eval)
                if loss_conf.get('apply_logit_adj_in_eval', False):
                    tau = loss_conf.get('logit_adj_tau', 1.0)
                    outputs = outputs + tau * log_prior
                    
                _, predicted = torch.max(outputs.data, 1)

                total_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        eval_accuracy = total_correct / len(test_dataset)
        
        # Calculate P, R, F1
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch + 1}/{train_conf['max_epoch']}: \n"
              f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}\n"
              f"  Eval Acc: {eval_accuracy:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
              
        if (epoch + 1) % 5 == 0 or epoch == train_conf['max_epoch'] - 1:
            print("\nClassification Report:\n")
            print(classification_report(all_labels, all_preds, zero_division=0))

        # --- 保存最佳模型与记录 ---
        # Record best Accuracy
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            best_acc_epoch = epoch + 1
            if monitor_metric == 'acc':
                best_model_path = save_dir / "best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best model saved to {best_model_path} with Acc: {best_accuracy:.4f}")
                epochs_no_improve = 0
            
        # Record best F1
        if f1 > best_f1:
            best_f1 = f1
            best_f1_epoch = epoch + 1
            if monitor_metric == 'f1':
                best_model_path = save_dir / "best_model.pth"
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best model saved to {best_model_path} with F1: {best_f1:.4f}")
                epochs_no_improve = 0
        
        # Early Stopping Logic based on monitor metric
        if monitor_metric == 'acc' and eval_accuracy <= best_accuracy:
             epochs_no_improve += 1
        elif monitor_metric == 'f1' and f1 <= best_f1:
             epochs_no_improve += 1
             
        if epochs_no_improve > 0:
            print(f"  No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                break

        # 更新学习率
        # Update learning rate
        scheduler.step()

    print("\n--- 训练完成 ---")
    print(f"最佳评估准确率: {best_accuracy:.4f} (Epoch {best_acc_epoch})")
    print(f"最佳 F1 Score: {best_f1:.4f} (Epoch {best_f1_epoch})")
    if train_conf['save_model_dir']:
        save_dir = Path(train_conf['save_model_dir'])
        with open(save_dir / "results.txt", 'w') as f:
             f.write(f"Best Acc: {best_accuracy:.4f} @ Epoch {best_acc_epoch}\n")
             f.write(f"Best F1: {best_f1:.4f} @ Epoch {best_f1_epoch}\n")
             f.write(f"Last Acc: {eval_accuracy:.4f}\n")
             f.write(f"Last F1: {f1:.4f}\n")
             f.write(f"Params: {params}\n")
             f.write(f"FLOPs: {flops}\n")
    
    return best_accuracy


if __name__ == '__main__':
    # 使用argparse来接收配置文件路径，使其更灵活
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/mynet_config.yml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    train(args.config)