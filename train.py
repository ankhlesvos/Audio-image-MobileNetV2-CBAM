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
import re
import random
from pathlib import Path

from dataset import AudioDataset
from modules.model import MyNet

# Knowledge Distillation imports (lazy — only used when kd_mode is True)
# from modules.teacher_model import ASTTeacherModel
# from modules.distillation import CombinedKDLoss


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
    test_dataset  = AudioDataset(data_list_path=data_conf['test_list'],  train=False)

    # Val dataset: used for early stopping / best model selection
    val_list_path = data_conf.get('val_list', None)
    if val_list_path:
        val_dataset = AudioDataset(data_list_path=val_list_path, train=False)
        print(f"验证集 (val) 使用: {val_list_path}")
    else:
        # Backward-compat: fall back to test set (old behaviour)
        print("[WARN] No val_list specified — using test set for early stopping (not recommended).")
        val_dataset = test_dataset

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
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=train_conf['batch_size'],
        shuffle=False,
        num_workers=train_conf['num_workers']
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=train_conf['batch_size'],
        shuffle=False,
        num_workers=train_conf['num_workers']
    )
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}, 测试集大小: {len(test_dataset)}")


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
        audio_mode=model_conf.get('audio_mode', False),
        use_imagenet_pretrain=model_conf.get('use_imagenet_pretrain', False),
        pretrain_mode=model_conf.get('pretrain_mode', 'backbone_only'),
    )
    model.to(device)
    print("模型结构:")
    # 简单的模型结构打印
    # print(model)

    # --- Warm-start from existing student checkpoint ---
    student_ckpt = model_conf.get('pretrain_student_checkpoint', '')
    if student_ckpt and os.path.exists(student_ckpt):
        print(f"[INFO] Warm-starting student from: {student_ckpt}")
        sd = torch.load(student_ckpt, map_location=device)
        # Strip thop metadata keys
        sd = {k: v for k, v in sd.items()
              if 'total_ops' not in k and 'total_params' not in k}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"         Missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"         Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
        print(f"[INFO] Warm-start complete.")
    elif student_ckpt:
        print(f"[WARN] pretrain_student_checkpoint not found: {student_ckpt} — training from scratch.")


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

    # --- Knowledge Distillation Setup ---
    kd_mode = train_conf.get('kd_mode', False)
    teacher_model = None
    kd_loss_fn = None

    if kd_mode:
        from modules.teacher_model import ASTTeacherModel
        from modules.distillation import CombinedKDLoss

        print("\n--- Knowledge Distillation Mode ENABLED ---")
        teacher_checkpoint = train_conf.get('kd_teacher_checkpoint', '')
        kd_alpha = train_conf.get('kd_alpha', 0.5)
        kd_temperature = train_conf.get('kd_temperature', 4.0)
        kd_feature_distill = train_conf.get('kd_feature_distill', False)
        kd_feature_beta = train_conf.get('kd_feature_beta', 1.0)

        # Load teacher model
        teacher_conf = config.get('teacher_conf', {})
        teacher_model = ASTTeacherModel(
            num_classes=model_conf['num_classes'],
            pretrained_name=teacher_conf.get('pretrained_name',
                                             'MIT/ast-finetuned-audioset-10-10-0.4593'),
        )

        if teacher_checkpoint and os.path.exists(teacher_checkpoint):
            state_dict = torch.load(teacher_checkpoint, map_location=device)
            # Filter thop keys if present
            clean_dict = {k: v for k, v in state_dict.items()
                          if 'total_ops' not in k and 'total_params' not in k}
            teacher_model.load_state_dict(clean_dict)
            print(f"  Teacher loaded from: {teacher_checkpoint}")
        else:
            print(f"  [WARN] Teacher checkpoint not found: {teacher_checkpoint}")
            print(f"         Using untrained teacher — KD loss will be random!")

        teacher_model.to(device)
        teacher_model.eval()  # Teacher is ALWAYS in eval mode
        for p in teacher_model.parameters():
            p.requires_grad = False  # Teacher does NOT participate in backprop

        t_params = sum(p.numel() for p in teacher_model.parameters())
        print(f"  Teacher params: {t_params / 1e6:.2f}M (frozen)")

        # Build KD loss
        kd_loss_fn = CombinedKDLoss(
            temperature=kd_temperature,
            alpha=kd_alpha,
            beta=kd_feature_beta if kd_feature_distill else 0.0,
            student_dim=model.last_channel,  # 1280
            teacher_dim=teacher_model.hidden_size,  # 768
        ).to(device)

        print(f"  KD Config: alpha={kd_alpha}, T={kd_temperature}, "
              f"feature_distill={kd_feature_distill}, beta={kd_feature_beta}")
    else:
        print("\n--- Knowledge Distillation Mode DISABLED (baseline) ---")

    # --- 6. 训练与评估循环 ---
    print("\n--- 6. 开始训练与评估 ---")
    best_accuracy = 0.0
    best_acc_epoch = 0
    best_f1 = 0.0
    best_f1_epoch = 0
    # --- Early stopping / best model config ---
    # monitor_metric controls early stopping signal:
    #   'val_loss'  → stop when val loss stops decreasing (most stable)
    #   'f1' / 'acc' → legacy: stop when file-level metric stops improving
    # best_model_metric controls which metric picks the saved checkpoint:
    #   'val_file_f1' (default) → save checkpoint with highest val file F1
    #   'val_file_acc' → save checkpoint with highest val file accuracy
    monitor_metric = train_conf.get('monitor_metric', 'val_loss')
    best_model_metric = train_conf.get('best_model_metric', 'val_file_f1')
    save_dir = Path(train_conf['save_model_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)  # 确保保存目录存在
    
    # Early stopping config
    patience = train_conf.get('patience', 15)
    min_epochs = train_conf.get('min_epochs', 15)
    min_delta = train_conf.get('min_delta', 0.002)
    epochs_no_improve = 0
    best_monitor_value = None  # tracks the best val_loss (or legacy metric)
    top_k_models = []  # List of tuples: (metric_value, model_path)
    top_k = 3
    save_every_n = train_conf.get('save_every_n_epochs', 0)  # 0 = disabled
    if save_every_n > 0:
        periodic_dir = save_dir / 'periodic'
        periodic_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Periodic checkpointing: every {save_every_n} epochs → {periodic_dir}")

    # ===== [TEMP] Sampler Verification — DELETE AFTER USE =====
    def verify_sampler_distribution(loader, num_cls, true_counts):
        from collections import Counter
        sampled_counts = Counter()
        for _, lab_a, lab_b, lam in loader:
            # lab_a 是 Mixup 前的原始主标签，用它来统计 sampler 行为
            for l in lab_a.tolist():
                sampled_counts[l] += 1
        print("\n[SAMPLER VERIFY] ===========================")
        print(f"  {'Class':<8} {'True Count':>12} {'Sampled Count':>14} {'Ratio':>8}")
        print(f"  {'-'*45}")
        for c in range(num_cls):
            tc = true_counts[c]
            sc = sampled_counts.get(c, 0)
            ratio = sc / tc if tc > 0 else float('inf')
            print(f"  {c:<8} {tc:>12} {sc:>14} {ratio:>8.3f}")
        print("[SAMPLER VERIFY] ===========================\n")

    verify_sampler_distribution(train_loader, num_classes, counts)
    # ===== [TEMP END] =====

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
            if model_conf.get('in_channels', 3) == 1 and inputs.size(1) == 3:
                inputs = inputs[:, 0:1, :, :]
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
            
            hard_loss = (loss_a * lam + loss_b * (1 - lam)).mean()

            # Knowledge Distillation: combine hard loss with soft teacher targets
            if kd_mode and teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                    teacher_features = teacher_model.get_features() if train_conf.get('kd_feature_distill', False) else None

                student_features = model.get_features() if train_conf.get('kd_feature_distill', False) else None

                loss = kd_loss_fn(
                    hard_loss=hard_loss,
                    student_logits=outputs,
                    teacher_logits=teacher_logits,
                    student_features=student_features,
                    teacher_features=teacher_features,
                )
            else:
                loss = hard_loss
            
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

        # ─── Val Loss computation (for early stopping when monitor=val_loss) ───
        val_loss_total = 0.0
        val_loss_count = 0
        model.eval()
        with torch.no_grad():
            for _vinp, _vla, _vlb, _vlam in val_loader:
                _vinp = _vinp.to(device)
                if model_conf.get('in_channels', 3) == 1 and _vinp.size(1) == 3:
                    _vinp = _vinp[:, 0:1, :, :]
                _vla = _vla.to(device)
                _vout = model(_vinp)
                if loss_conf.get('apply_logit_adj_in_train', False):
                    tau = loss_conf.get('logit_adj_tau', 1.0)
                    _vout = _vout + tau * log_prior
                _vloss = criterion_none(_vout, _vla)
                val_loss_total += _vloss.sum().item()
                val_loss_count += _vla.size(0)
        avg_val_loss = val_loss_total / max(val_loss_count, 1)

        # ─── Val Evaluation (used for best model selection) ───
        def run_file_level_eval(loader, dataset_ref, desc_tag):
            """File-level evaluation on the given loader using specified voting strategy.
            Returns: (file_accuracy, file_precision, file_recall, file_f1,
                      seg_accuracy, seg_f1, file_gt, file_pred_vote)
            """
            _correct = 0
            _all_seg_preds, _all_seg_labels = [], []
            _file_preds: dict = {}  # {fid: [pred_class, ...]}
            _file_scores: dict = {} # {fid: [[score_c0, score_c1, ...], ...]}
            _file_labels_map: dict = {}
            _gidx = 0

            with torch.no_grad():
                for mel_spec, la, lb, _lam in tqdm(loader, desc=desc_tag):
                    mel_spec = mel_spec.to(device)
                    if model_conf.get('in_channels', 3) == 1 and mel_spec.size(1) == 3:
                        mel_spec = mel_spec[:, 0:1, :, :]
                    _labels  = la.to(device)
                    _out     = model(mel_spec)

                    # Logit Adjustment (Eval)
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
                        # Full-path file ID (matches create_val_split.py logic)
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
            _seg_f1  = f1_score(_all_seg_labels, _all_seg_preds, average='macro', zero_division=0)

            _fgt, _fvote = [], []
            voting_strategy = train_conf.get('file_voting_strategy', 'majority')
            top_k_vote = train_conf.get('file_voting_top_k', 3)

            for _fid, _pl in _file_preds.items():
                _sl = np.array(_file_scores[_fid]) # shape: (num_segs, num_classes)
                
                if voting_strategy == 'entropy':
                    # Calculate entropy for each segment prediction
                    # _sl is probabilities since we used softmax
                    eps = 1e-9
                    entropies = -np.sum(_sl * np.log(_sl + eps), axis=1)
                    # Lower entropy means higher confidence. 
                    # Weight = 1 / (entropy + eps). We sum weighted probabilities per class.
                    weights = 1.0 / (entropies + eps)
                    weighted_scores = np.average(_sl, axis=0, weights=weights)
                    _winner = int(np.argmax(weighted_scores))
                    _ca = -1 # not used
                elif voting_strategy == 'top_k':
                    # Sort segments by confidence (max prob) descending
                    max_probs = np.max(_sl, axis=1)
                    top_indices = np.argsort(max_probs)[-top_k_vote:]
                    top_preds = np.array(_pl)[top_indices]
                    _ca = np.bincount(top_preds, minlength=num_classes)
                    _winner = int(np.argmax(_ca))
                else:
                    # Default majority voting
                    _ca = np.bincount(_pl, minlength=num_classes)
                    _winner = int(np.argmax(_ca))

                _fvote.append(_winner)
                _fgt.append(_file_labels_map[_fid])

            if len(_fgt) > 0:
                _facc = sum(p == g for p, g in zip(_fvote, _fgt)) / len(_fgt)
                _fp   = precision_score(_fgt, _fvote, average='macro', zero_division=0)
                _fr   = recall_score(_fgt, _fvote, average='macro', zero_division=0)
                _ff1  = f1_score(_fgt, _fvote, average='macro', zero_division=0)
            else:
                _facc = _seg_acc
                _fp = _fr = _ff1 = _seg_f1

            # ── Diagnostic: unique file count & per-class breakdown ──────────
            from collections import Counter as _Counter
            _per_cls_files = _Counter(_file_labels_map[fid] for fid in _file_preds)
            _n_files = len(_file_preds)
            _n_segs  = len(_all_seg_preds)
            print(f"    [{desc_tag}] Aggregated {_n_files} unique files "
                  f"({_n_segs} segs) | per-class files: "
                  f"{dict(sorted(_per_cls_files.items()))}")
            if _n_files == 0:
                print(f"    [{desc_tag}] !! WARNING: 0 files aggregated — check regex / dataset paths !!")
            # ────────────────────────────────────────────────────────────────

            return _facc, _fp, _fr, _ff1, _seg_acc, _seg_f1, _fgt, _fvote


        model.eval()
        val_tag = f"Epoch {epoch + 1}/{train_conf['max_epoch']} [Val]"
        (file_accuracy, file_precision, file_recall, f1,
         seg_accuracy, seg_f1, val_gt, val_pred_vote) = run_file_level_eval(
             val_loader, val_dataset, val_tag)

        # eval_accuracy kept for compatibility with Acc monitor mode
        eval_accuracy = file_accuracy

        print(f"Epoch {epoch + 1}/{train_conf['max_epoch']}:\n"
              f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}\n"
              f"  Val Loss:   {avg_val_loss:.4f}  <-- early stopping\n"
              f"  [Val-Seg]   Acc: {seg_accuracy:.4f} | F1: {seg_f1:.4f}\n"
              f"  [Val-File]  Acc: {file_accuracy:.4f} | P: {file_precision:.4f} | R: {file_recall:.4f} | F1: {f1:.4f}  <-- best checkpoint")

        if (epoch + 1) % 5 == 0 or epoch == train_conf['max_epoch'] - 1:
            print("\nClassification Report (Val File-Level):\n")
            print(classification_report(val_gt, val_pred_vote, zero_division=0))

        # --- 保存最佳模型 (based on best_model_metric) ---
        # best_model_metric selects which val metric determines the saved checkpoint
        if best_model_metric == 'val_file_acc':
            ckpt_metric_value = file_accuracy
        else:  # default: 'val_file_f1'
            ckpt_metric_value = f1

        new_best_ckpt = False
        if len(top_k_models) < top_k or ckpt_metric_value > top_k_models[0][0]:
            best_model_path = save_dir / f"best_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"  New top-{top_k} checkpoint saved to {best_model_path} "
                  f"(best_model_metric={best_model_metric}: {ckpt_metric_value:.4f})")
            new_best_ckpt = True
            
            top_k_models.append((ckpt_metric_value, best_model_path))
            top_k_models.sort(key=lambda x: x[0])  # ascending; worst=index 0, best=index -1
            
            if len(top_k_models) > top_k:
                removed_metric, removed_path = top_k_models.pop(0)
                if removed_path.exists():
                    removed_path.unlink()
        
        # Keep track of absolute best for displaying stats
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            best_acc_epoch = epoch + 1
        
        if f1 > best_f1:
            best_f1 = f1
            best_f1_epoch = epoch + 1

        # --- Early Stopping (based on monitor_metric) ---
        # Determine the early-stopping signal
        if monitor_metric == 'val_loss':
            # For loss: improvement means DECREASE, so we negate for comparison
            es_improved = False
            if best_monitor_value is None:
                best_monitor_value = avg_val_loss
                es_improved = True
            elif avg_val_loss < best_monitor_value - min_delta:
                best_monitor_value = avg_val_loss
                es_improved = True
        else:
            # Legacy: monitor file-level metric (higher is better)
            es_value = eval_accuracy if monitor_metric == 'acc' else f1
            es_improved = False
            if best_monitor_value is None:
                best_monitor_value = es_value
                es_improved = True
            elif es_value > best_monitor_value + min_delta:
                best_monitor_value = es_value
                es_improved = True

        if es_improved:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve > 0:
            print(f"  [EarlyStop] No {monitor_metric} improvement for {epochs_no_improve}/{patience} epochs "
                  f"(best={best_monitor_value:.4f}, delta={min_delta})")
        if epochs_no_improve >= patience and (epoch + 1) >= min_epochs:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs! "
                  f"(monitor={monitor_metric}, best={best_monitor_value:.4f})")
            break
        elif epochs_no_improve >= patience and (epoch + 1) < min_epochs:
            print(f"  [EarlyStop] Would stop, but min_epochs={min_epochs} not reached yet.")

        # --- Periodic checkpoint save (regardless of val metric) ---
        if save_every_n > 0 and (epoch + 1) % save_every_n == 0:
            periodic_path = periodic_dir / f"epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), periodic_path)
            print(f"  [Periodic] Saved checkpoint: {periodic_path}")

        # 更新学习率
        # Update learning rate
        scheduler.step()

    print("\n--- 训练完成 ---")
    print(f"最佳 Val 准确率: {best_accuracy:.4f} (Epoch {best_acc_epoch})")
    print(f"最佳 Val F1 Score: {best_f1:.4f} (Epoch {best_f1_epoch})")

    # ─── Final Test Set Evaluation (run once after training) ──────────────────
    print("\n--- 最终 Test Set 评估 ---")
    model.eval()
    # Load absolute best model before final eval
    if top_k_models and top_k_models[-1][1].exists():
        best_model_path = top_k_models[-1][1]
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    elif (save_dir / "best_model.pth").exists():
        # Fallback for old codebase structure
        model.load_state_dict(torch.load(save_dir / "best_model.pth", map_location=device))
        print(f"Loaded best model from {save_dir / 'best_model.pth'}")

    (test_file_acc, test_file_prec, test_file_rec, test_f1,
     test_seg_acc, test_seg_f1, test_gt, test_pred) = run_file_level_eval(
         test_loader, test_dataset, "[Test]")

    print(f"  [Test-Seg]   Acc: {test_seg_acc:.4f} | F1: {test_seg_f1:.4f}")
    print(f"  [Test-File]  Acc: {test_file_acc:.4f} | P: {test_file_prec:.4f} | R: {test_file_rec:.4f} | F1: {test_f1:.4f}")
    print("\nClassification Report (Test File-Level):\n")
    print(classification_report(test_gt, test_pred, zero_division=0))

    if train_conf['save_model_dir']:
        save_dir = Path(train_conf['save_model_dir'])
        with open(save_dir / "results.txt", 'w') as f:
            f.write(f"Best Val Acc: {best_accuracy:.4f} @ Epoch {best_acc_epoch}\n")
            f.write(f"Best Val F1:  {best_f1:.4f} @ Epoch {best_f1_epoch}\n")
            f.write(f"Test File Acc: {test_file_acc:.4f}\n")
            f.write(f"Test File F1:  {test_f1:.4f}\n")
            f.write(f"Test Seg  Acc: {test_seg_acc:.4f}\n")
            f.write(f"Test Seg  F1:  {test_seg_f1:.4f}\n")
            f.write(f"Params: {params}\n")
            f.write(f"FLOPs: {flops}\n")

    return test_file_acc


if __name__ == '__main__':
    # 使用argparse来接收配置文件路径，使其更灵活
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/mynet_config.yml',
                        help='Path to the configuration file')
    args = parser.parse_args()

    train(args.config)