# train_teacher.py
#
# Standalone training script for the AST teacher model.
# Must be trained BEFORE knowledge distillation — KD is meaningless
# unless the teacher demonstrably outperforms the student baseline.
#
# Usage:
#   python train_teacher.py -c configs/teacher_config.yml
#   python train_teacher.py -c configs/teacher_config.yml --max_epoch 2  # smoke test

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import numpy as np
from tqdm import tqdm
import yaml
import os
import re
import random
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from dataset import AudioDataset
from modules.teacher_model import ASTTeacherModel


def train_teacher(config_path: str, max_epoch_override: int = None):
    """
    Train AST teacher model.

    Args:
        config_path: Path to teacher config YAML.
        max_epoch_override: If set, overrides max_epoch in config (for quick testing).
    """
    # --- 1. Load config ---
    print("=" * 60)
    print("  AST Teacher Model Training")
    print("=" * 60)
    print("\n--- 1. Loading config ---")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    train_conf = config['train_conf']
    data_conf = config['data_conf']
    teacher_conf = config.get('teacher_conf', {})
    model_conf = config.get('model_conf', {})

    num_classes = teacher_conf.get('num_classes', model_conf.get('num_classes', 3))

    if max_epoch_override is not None:
        train_conf['max_epoch'] = max_epoch_override
        print(f"[OVERRIDE] max_epoch set to {max_epoch_override}")

    # --- 1.5 Seed ---
    seed = train_conf.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --- 2. Device ---
    device = torch.device("cuda" if torch.cuda.is_available() and train_conf['use_gpu'] else "cpu")
    print(f"\n--- 2. Device: {device} ---")

    # --- 3. Data ---
    print("\n--- 3. Preparing data ---")
    train_dataset = AudioDataset(data_list_path=data_conf['train_list'], train=True)
    test_dataset = AudioDataset(data_list_path=data_conf['test_list'], train=False)

    val_list_path = data_conf.get('val_list', None)
    if val_list_path:
        val_dataset = AudioDataset(data_list_path=val_list_path, train=False)
    else:
        print("[WARN] No val_list — using test set (not recommended).")
        val_dataset = test_dataset

    # Class weights (ENS)
    labels = []
    for line in train_dataset.lines:
        try:
            _, lab = line.split('\t')
            labels.append(int(lab))
        except:
            labels.append(-1)

    valid_labels = [l for l in labels if l >= 0]
    class_count = Counter(valid_labels)
    counts = [class_count.get(i, 0) for i in range(num_classes)]
    print(f"Class Counts: {counts}")

    beta = 0.9999
    effective_num = 1.0 - np.power(beta, counts)
    weights = np.zeros_like(effective_num)
    valid_classes = effective_num > 1e-6
    weights[valid_classes] = (1.0 - beta) / np.array(effective_num)[valid_classes]
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights) * num_classes
    else:
        weights = np.ones(num_classes)

    if train_conf.get('use_class_weights', True):
        class_weights = torch.tensor(weights).float().to(device)
        print(f"Class Weights: {class_weights}")
    else:
        class_weights = None

    # Log prior for logit adjustment
    sum_counts = sum(counts) if sum(counts) > 0 else 1
    prior = np.array(counts, dtype=np.float64) / sum_counts
    log_prior = torch.log(torch.tensor(prior, dtype=torch.float32) + 1e-12).to(device)

    # DataLoaders
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset, batch_size=train_conf['batch_size'],
        shuffle=True, num_workers=train_conf['num_workers'], generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_conf['batch_size'],
        shuffle=False, num_workers=train_conf['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_conf['batch_size'],
        shuffle=False, num_workers=train_conf['num_workers']
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # --- 4. Build teacher model ---
    print("\n--- 4. Building AST teacher model ---")
    pretrained_name = teacher_conf.get('pretrained_name',
                                       'MIT/ast-finetuned-audioset-10-10-0.4593')
    freeze_encoder_epochs = teacher_conf.get('freeze_encoder_epochs', 0)

    model = ASTTeacherModel(
        num_classes=num_classes,
        pretrained_name=pretrained_name,
        freeze_encoder=(freeze_encoder_epochs > 0),
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    if freeze_encoder_epochs > 0:
        print(f"Encoder frozen for first {freeze_encoder_epochs} epochs.")

    # --- 5. Loss & Optimizer ---
    print("\n--- 5. Loss & Optimizer ---")
    loss_conf = train_conf.get('loss_conf', {})
    gamma = loss_conf.get('gamma', 2.0)
    label_smoothing = loss_conf.get('label_smoothing', 0.1)

    class FocalLoss(nn.Module):
        def __init__(self, weight=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
            super().__init__()
            self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none',
                                          label_smoothing=label_smoothing)
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = self.ce(inputs, targets)
            pt = torch.exp(-ce_loss)
            loss = ((1 - pt) ** self.gamma) * ce_loss if self.gamma > 0 else ce_loss
            return loss.mean() if self.reduction == 'mean' else loss

    criterion = FocalLoss(weight=class_weights, gamma=gamma,
                          reduction='none', label_smoothing=label_smoothing)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_conf['learning_rate'],
        weight_decay=train_conf.get('weight_decay', 1e-2)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_conf['max_epoch'], eta_min=0
    )

    # --- 6. Training loop ---
    print("\n--- 6. Starting training ---")
    save_dir = Path(train_conf['save_model_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    best_f1_epoch = 0
    best_accuracy = 0.0
    best_acc_epoch = 0
    monitor_metric = train_conf.get('monitor_metric', 'f1')
    patience = train_conf.get('patience', 10)
    epochs_no_improve = 0
    top_k_models = []
    top_k = 3

    def run_file_level_eval(loader, dataset_ref, desc_tag):
        """File-level evaluation — identical logic to train.py."""
        _correct = 0
        _all_seg_preds, _all_seg_labels = [], []
        _file_preds = {}
        _file_scores = {}
        _file_labels_map = {}
        _gidx = 0

        with torch.no_grad():
            for mel_spec, la, lb, _lam in tqdm(loader, desc=desc_tag):
                mel_spec = mel_spec.to(device)
                _labels = la.to(device)
                _out = model(mel_spec)

                if loss_conf.get('apply_logit_adj_in_eval', False):
                    tau = loss_conf.get('logit_adj_tau', 1.0)
                    _out = _out + tau * log_prior

                _, _pred = torch.max(_out.data, 1)
                _correct += (_pred == _labels).sum().item()
                _all_seg_preds.extend(_pred.cpu().numpy())
                _all_seg_labels.extend(_labels.cpu().numpy())

                _bpreds = _pred.cpu().numpy()
                _bscores = torch.softmax(_out, dim=1).cpu().numpy()
                for _i in range(mel_spec.size(0)):
                    if _gidx >= len(dataset_ref.lines):
                        break
                    _line = dataset_ref.lines[_gidx]
                    try:
                        _path, _str_lab = _line.split('\t')
                    except ValueError:
                        _gidx += 1
                        continue
                    _fid_match = re.match(r'^(.+)_seg\d+\.wav$', _path)
                    _fid = _fid_match.group(1) if _fid_match else _path
                    if _fid not in _file_preds:
                        _file_preds[_fid] = []
                        _file_scores[_fid] = []
                        _file_labels_map[_fid] = int(_str_lab)
                    _file_preds[_fid].append(int(_bpreds[_i]))
                    _file_scores[_fid].append(_bscores[_i])
                    _gidx += 1

        _seg_acc = _correct / max(len(dataset_ref), 1)
        _seg_f1 = f1_score(_all_seg_labels, _all_seg_preds, average='macro', zero_division=0)

        _fgt, _fvote = [], []
        for _fid, _pl in _file_preds.items():
            _ca = np.bincount(_pl, minlength=num_classes)
            _winner = int(np.argmax(_ca))
            _fvote.append(_winner)
            _fgt.append(_file_labels_map[_fid])

        if len(_fgt) > 0:
            _facc = sum(p == g for p, g in zip(_fvote, _fgt)) / len(_fgt)
            _fp = precision_score(_fgt, _fvote, average='macro', zero_division=0)
            _fr = recall_score(_fgt, _fvote, average='macro', zero_division=0)
            _ff1 = f1_score(_fgt, _fvote, average='macro', zero_division=0)
        else:
            _facc = _seg_acc
            _fp = _fr = _ff1 = _seg_f1

        _per_cls = Counter(_file_labels_map[fid] for fid in _file_preds)
        print(f"    [{desc_tag}] {len(_file_preds)} files ({len(_all_seg_preds)} segs) "
              f"| per-class: {dict(sorted(_per_cls.items()))}")

        return _facc, _fp, _fr, _ff1, _seg_acc, _seg_f1, _fgt, _fvote

    for epoch in range(train_conf['max_epoch']):
        # Unfreeze encoder after freeze_encoder_epochs
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs:
            print(f"\n[INFO] Epoch {epoch+1}: Unfreezing encoder. Rebuilding optimizer.")
            model.unfreeze_encoder()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=train_conf['learning_rate'] * 0.1,  # use lower LR after unfreeze
                weight_decay=train_conf.get('weight_decay', 1e-2)
            )
            remaining = train_conf['max_epoch'] - epoch
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining, eta_min=0
            )

        # --- Train phase ---
        model.train()
        total_loss = 0
        total_correct = 0

        train_pbar = tqdm(train_loader,
                          desc=f"Epoch {epoch+1}/{train_conf['max_epoch']} [Train]")
        for inputs, label_a, label_b, lam in train_pbar:
            inputs = inputs.to(device)
            label_a = label_a.to(device)
            label_b = label_b.to(device)
            lam = lam.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)

            if loss_conf.get('apply_logit_adj_in_train', False):
                tau = loss_conf.get('logit_adj_tau', 1.0)
                outputs = outputs + tau * log_prior

            # Mixup loss
            loss_a = criterion(outputs, label_a)
            loss_b = criterion(outputs, label_b)
            loss = (loss_a * lam + loss_b * (1 - lam)).mean()

            loss.backward()
            # Gradient clipping for Transformer stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == label_a).sum().item()
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / len(train_dataset)

        # --- Val evaluation ---
        model.eval()
        val_tag = f"Epoch {epoch+1}/{train_conf['max_epoch']} [Val]"
        (file_accuracy, file_precision, file_recall, f1,
         seg_accuracy, seg_f1, val_gt, val_pred_vote) = run_file_level_eval(
            val_loader, val_dataset, val_tag)

        eval_accuracy = file_accuracy

        print(f"Epoch {epoch+1}/{train_conf['max_epoch']}:\n"
              f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}\n"
              f"  [Val-Seg]   Acc: {seg_accuracy:.4f} | F1: {seg_f1:.4f}\n"
              f"  [Val-File]  Acc: {file_accuracy:.4f} | P: {file_precision:.4f} "
              f"| R: {file_recall:.4f} | F1: {f1:.4f}  <-- model selection")

        if (epoch + 1) % 5 == 0 or epoch == train_conf['max_epoch'] - 1:
            print("\nClassification Report (Val File-Level):\n")
            print(classification_report(val_gt, val_pred_vote, zero_division=0))

        # --- Model saving & early stopping ---
        current_metric = eval_accuracy if monitor_metric == 'acc' else f1
        new_best = False

        if len(top_k_models) < top_k or current_metric > top_k_models[0][0]:
            best_path = save_dir / f"best_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  Saved top-{top_k} model: {best_path} "
                  f"(Val-{monitor_metric.upper()}: {current_metric:.4f})")
            new_best = True
            top_k_models.append((current_metric, best_path))
            top_k_models.sort(key=lambda x: x[0])
            if len(top_k_models) > top_k:
                _, removed = top_k_models.pop(0)
                if removed.exists():
                    removed.unlink()

        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            best_acc_epoch = epoch + 1
        if f1 > best_f1:
            best_f1 = f1
            best_f1_epoch = epoch + 1

        if new_best:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve > 0:
            print(f"  No improvement for {epochs_no_improve} epochs.")
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs!")
            break

        scheduler.step()

    # --- Final test evaluation ---
    print("\n" + "=" * 60)
    print("  Final Test Set Evaluation")
    print("=" * 60)
    model.eval()

    if top_k_models and top_k_models[-1][1].exists():
        best_path = top_k_models[-1][1]
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded best model from {best_path}")

    (test_file_acc, test_file_prec, test_file_rec, test_f1,
     test_seg_acc, test_seg_f1, test_gt, test_pred) = run_file_level_eval(
        test_loader, test_dataset, "[Test]")

    print(f"  [Test-Seg]   Acc: {test_seg_acc:.4f} | F1: {test_seg_f1:.4f}")
    print(f"  [Test-File]  Acc: {test_file_acc:.4f} | P: {test_file_prec:.4f} "
          f"| R: {test_file_rec:.4f} | F1: {test_f1:.4f}")
    print("\nClassification Report (Test File-Level):\n")
    print(classification_report(test_gt, test_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(test_gt, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('AST Teacher - Confusion Matrix (Test File-Level)')
    cm_path = save_dir / "teacher_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")

    # Save results
    with open(save_dir / "results.txt", 'w') as f:
        f.write(f"Best Val Acc: {best_accuracy:.4f} @ Epoch {best_acc_epoch}\n")
        f.write(f"Best Val F1:  {best_f1:.4f} @ Epoch {best_f1_epoch}\n")
        f.write(f"Test File Acc: {test_file_acc:.4f}\n")
        f.write(f"Test File F1:  {test_f1:.4f}\n")
        f.write(f"Test Seg  Acc: {test_seg_acc:.4f}\n")
        f.write(f"Test Seg  F1:  {test_seg_f1:.4f}\n")
        f.write(f"Total Params: {total_params}\n")
        f.write(f"Trainable Params: {trainable_params}\n")

    # Also save per-class F1
    per_class = classification_report(test_gt, test_pred, output_dict=True, zero_division=0)
    with open(save_dir / "per_class_metrics.txt", 'w') as f:
        for cls_key, metrics in per_class.items():
            if isinstance(metrics, dict):
                f.write(f"Class {cls_key}: "
                        f"P={metrics['precision']:.4f} "
                        f"R={metrics['recall']:.4f} "
                        f"F1={metrics['f1-score']:.4f} "
                        f"Support={metrics['support']}\n")

    print(f"\nResults saved to {save_dir}")
    print(f"\nBest Val Acc: {best_accuracy:.4f} (Epoch {best_acc_epoch})")
    print(f"Best Val F1:  {best_f1:.4f} (Epoch {best_f1_epoch})")

    return test_file_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train AST Teacher Model")
    parser.add_argument('-c', '--config', default='configs/teacher_config.yml',
                        help='Path to teacher config file')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='Override max_epoch (for quick tests)')
    args = parser.parse_args()

    train_teacher(args.config, args.max_epoch)
