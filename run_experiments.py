import yaml
import os
import sys
import subprocess
from pathlib import Path
import copy
import re

# Base Configuration Template for DeepShip
BASE_CONFIG_TEMPLATE = {
    'train_conf': {
        'use_gpu': True,
        'batch_size': 16, # Reduced to 16 to avoid OOM. Adjust if needed.
        'num_workers': 4,
        'max_epoch': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'freeze_backbone': False, # New parameter for ablation
        'seed': 42, # Ensure random seed reproducibility
        'save_model_dir': None,
        'use_sampler': False,
        'sampler_alpha': 0.5,
        'use_class_weights': True,
        'monitor_metric': 'f1', # acc, f1
        'patience': 10, # default patience
        'loss_conf': {
            'loss_type': 'focal', # 'ce' or 'focal'
            'gamma': 2.0,
            'label_smoothing': 0.1,
            'apply_logit_adj_in_train': False,
            'apply_logit_adj_in_eval': False,
            'logit_adj_tau': 1.0,
            'pair_penalty': {
                'use_penalty': False,
                'weight': 2.0,
                'targets': [
                    [1, 2],   # Passengership misclassified as Tanker
                    [2, 0],   # Tanker misclassified as Cargo+Tug
                ]
            }
        }
    },
    'data_conf': {
        'train_list': 'data/train_list_5s_new.txt',  # val-split version
        'val_list':   'data/val_list_5s.txt',         # file-level val (early stopping)
        'test_list':  'data/test_list_5s.txt'         # frozen, never used during training
    },
    'model_conf': {
        'num_classes': 3,
        'in_channels': 3, # Updated for log-mel + delta + delta-delta
        'width_mult': 1.0,
        'model_config': None,
        # Ablation flags
        'asymmetric': False,
        'multiscale': False,
        'force_no_residual': False,
        'audio_mode': False
    }
}

# Define Experiments M1-M5
# Standard MobileNetV2 structure [t, c, n, s, attn]
# attn codes: 0: None, 1: Post-DW CBAM, 2: Pre-DW CBAM, 3: SE

CONFIG_BASE = [
    [1, 16, 1, 1, 0],
    [6, 24, 2, 2, 0],
    [6, 32, 3, 1, 0],
    [6, 64, 4, 2, 0],
    [6, 96, 3, 1, 0],
    [6, 160, 3, 1, 0],
    [6, 320, 1, 1, 0]
]

# Config with Pre-DW CBAM at s24 (Stage 2 and Stage 4)
# M4/M5 uses this.
CONFIG_CBAM_S24 = [
    [1, 16, 1, 1, 0],
    [6, 24, 2, 2, 2], # s24 (Stage 2)
    [6, 32, 3, 1, 0],
    [6, 64, 4, 2, 2], # s24 (Stage 4)
    [6, 96, 3, 1, 0],
    [6, 160, 3, 1, 0],
    [6, 320, 1, 1, 0]
]

# Config with SE everywhere (M3)
CONFIG_SE_ALL = [
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 32, 3, 1, 3],
    [6, 64, 4, 2, 3],
    [6, 96, 3, 1, 3],
    [6, 160, 3, 1, 3],
    [6, 320, 1, 1, 3]
]

# Config with Frequency Attention at s24 (M6)
CONFIG_FREQ_S24 = [
    [1, 16, 1, 1, 0],
    [6, 24, 2, 2, 4], # s24 (Stage 2) attention=4 (freq)
    [6, 32, 3, 1, 0],
    [6, 64, 4, 2, 4], # s24 (Stage 4) attention=4 (freq)
    [6, 96, 3, 1, 0],
    [6, 160, 3, 1, 0],
    [6, 320, 1, 1, 0]
]

EXPERIMENTS = {
    "M1_BasicCNN": {
        "model_config": CONFIG_BASE,
        "force_no_residual": True,
        "asymmetric": False
    },
    "M2_MobileNetV2": {
        "model_config": CONFIG_BASE,
        "force_no_residual": False,
        "asymmetric": False
    },
    "M3_MobileNetV2_SE": {
        "model_config": CONFIG_SE_ALL,
        "force_no_residual": False,
        "asymmetric": False
    },
    "M4_MobileNetV2_CBAM": {
        "model_config": CONFIG_CBAM_S24,
        "force_no_residual": False,
        "asymmetric": False
    },
    "M5_Baseline": {
        "model_config": CONFIG_CBAM_S24,
        "asymmetric": True,
        "multiscale": True,
        "use_sampler": False,
        "use_class_weights": False,  
        "loss_type": "ce",           # Strict clean baseline
        "gamma": 0.0                 # Strict fallback
    },
    "M5_Plus_Sampler": {
        "model_config": CONFIG_CBAM_S24,
        "asymmetric": True,
        "multiscale": True,
        "use_sampler": True,
        "sampler_alpha": 0.5,
        "use_class_weights": False,
        "loss_type": "ce",
        "gamma": 0.0
    },
    "M5_LA_TT": {
        # Both train AND eval use LA — existing combined condition
        "model_config": CONFIG_CBAM_S24,
        "asymmetric": True,
        "multiscale": True,
        "use_sampler": False,
        "use_class_weights": False,
        "loss_type": "ce",
        "gamma": 0.0,
        "apply_logit_adj_in_train": True,
        "apply_logit_adj_in_eval": True,
        "logit_adj_tau": 1.0
    },
    "M5_LA_FT": {
        # Only eval (inference post-processing) — train without LA
        "model_config": CONFIG_CBAM_S24,
        "asymmetric": True,
        "multiscale": True,
        "use_sampler": False,
        "use_class_weights": False,
        "loss_type": "ce",
        "gamma": 0.0,
        "apply_logit_adj_in_train": False,
        "apply_logit_adj_in_eval": True,
        "logit_adj_tau": 1.0
    },
    "M5_LA_TF": {
        # Only train regularization — eval without LA (raw logits at test time)
        "model_config": CONFIG_CBAM_S24,
        "asymmetric": True,
        "multiscale": True,
        "use_sampler": False,
        "use_class_weights": False,
        "loss_type": "ce",
        "gamma": 0.0,
        "apply_logit_adj_in_train": True,
        "apply_logit_adj_in_eval": False,
        "logit_adj_tau": 1.0
    },
    "M5_LA_FF": {
        # No LA at all — clean CE baseline for LA ablation comparison
        "model_config": CONFIG_CBAM_S24,
        "asymmetric": True,
        "multiscale": True,
        "use_sampler": False,
        "use_class_weights": False,
        "loss_type": "ce",
        "gamma": 0.0,
        "apply_logit_adj_in_train": False,
        "apply_logit_adj_in_eval": False,
        "logit_adj_tau": 1.0
    },
    "M6_AudioMobileNetV2": {
        "model_config": CONFIG_FREQ_S24,
        "force_no_residual": False,
        "asymmetric": True,
        "audio_mode": True,
        "in_channels": 1
    },

    # ── tau sweep (LA T/T, LS=0.1) ────────────────────────────────────────────────
    # Compare against M5_LA_TT (tau=1.0); lower tau → less aggressive logit shift
    "M5_LA_TT_tau025": {
        "model_config": CONFIG_CBAM_S24, "asymmetric": True, "multiscale": True,
        "use_sampler": False, "use_class_weights": False,
        "loss_type": "ce", "gamma": 0.0, "label_smoothing": 0.1,
        "apply_logit_adj_in_train": True, "apply_logit_adj_in_eval": True,
        "logit_adj_tau": 0.25,
    },
    "M5_LA_TT_tau050": {
        "model_config": CONFIG_CBAM_S24, "asymmetric": True, "multiscale": True,
        "use_sampler": False, "use_class_weights": False,
        "loss_type": "ce", "gamma": 0.0, "label_smoothing": 0.1,
        "apply_logit_adj_in_train": True, "apply_logit_adj_in_eval": True,
        "logit_adj_tau": 0.5,
    },
    # M5_LA_TT is tau=1.0 (already defined above)
    "M5_LA_TT_tau150": {
        "model_config": CONFIG_CBAM_S24, "asymmetric": True, "multiscale": True,
        "use_sampler": False, "use_class_weights": False,
        "loss_type": "ce", "gamma": 0.0, "label_smoothing": 0.1,
        "apply_logit_adj_in_train": True, "apply_logit_adj_in_eval": True,
        "logit_adj_tau": 1.5,
    },

    # ── label_smoothing sweep (LA T/T, tau=1.0) ────────────────────────────────
    # Isolates the smoothing × LA interaction; M5_LA_TT has ls=0.1 as anchor
    "M5_LA_TT_ls00": {
        "model_config": CONFIG_CBAM_S24, "asymmetric": True, "multiscale": True,
        "use_sampler": False, "use_class_weights": False,
        "loss_type": "ce", "gamma": 0.0, "label_smoothing": 0.0,
        "apply_logit_adj_in_train": True, "apply_logit_adj_in_eval": True,
        "logit_adj_tau": 1.0,
    },
    "M5_LA_TT_ls005": {
        "model_config": CONFIG_CBAM_S24, "asymmetric": True, "multiscale": True,
        "use_sampler": False, "use_class_weights": False,
        "loss_type": "ce", "gamma": 0.0, "label_smoothing": 0.05,
        "apply_logit_adj_in_train": True, "apply_logit_adj_in_eval": True,
        "logit_adj_tau": 1.0,
    },
    # M5_LA_TT is ls=0.1 (reference, already defined above)
}

def main():
    configs_dir = Path("configs/ablation_study")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results file
    with open("ablation_results_summary.txt", "w") as f:
        f.write("Experiment | Best Acc | Best F1 | Sampler | Alpha | CE/FL | LA Train/Eval | Tau\n")
        f.write("-" * 100 + "\n")

    for exp_name, setup in EXPERIMENTS.items():
        # ── Select which experiments to run ──────────────────────────────────────
        # M1-M5: architecture ablation | M5_LA_*: logit adjustment ablation
        run_list = [
            "M1_BasicCNN",
            "M2_MobileNetV2",
            "M3_MobileNetV2_SE",
            "M4_MobileNetV2_CBAM",
            # ── LA 2×2 ablation ──
            "M5_LA_FF",           # No LA — CE baseline
            "M5_LA_FT",           # Inference-only LA
            "M5_LA_TF",           # Training-only LA
            "M5_LA_TT",           # Both (tau=1.0, ls=0.1)
            # ── tau sweep ──
            "M5_LA_TT_tau025",
            "M5_LA_TT_tau050",
            "M5_LA_TT_tau150",
            # ── label_smoothing sweep ──
            "M5_LA_TT_ls00",
            "M5_LA_TT_ls005",
        ]
        if exp_name not in run_list:
            continue
        print(f"\n{'=' * 20} Starting Experiment: {exp_name} {'=' * 20}")

        current_config = copy.deepcopy(BASE_CONFIG_TEMPLATE)
        
        # Apply specific settings
        current_config['model_conf']['model_config'] = setup['model_config']
        current_config['model_conf']['asymmetric'] = setup.get('asymmetric', False)
        current_config['model_conf']['multiscale'] = setup.get('multiscale', False)
        if 'in_channels' in setup:
            current_config['model_conf']['in_channels'] = setup['in_channels']
        
        # Apply specific ablation parameters to train_conf
        current_config['train_conf']['use_sampler'] = setup.get('use_sampler', False)
        current_config['train_conf']['sampler_alpha'] = setup.get('sampler_alpha', 0.5)
        current_config['train_conf']['use_class_weights'] = setup.get('use_class_weights', False)
        current_config['train_conf']['loss_conf']['loss_type'] = setup.get('loss_type', 'focal')
        current_config['train_conf']['loss_conf']['gamma'] = setup.get('gamma', 2.0)
        # label_smoothing is overridable per-experiment (base template default: 0.1)
        if 'label_smoothing' in setup:
            current_config['train_conf']['loss_conf']['label_smoothing'] = setup['label_smoothing']

        # Logit Adjustment controls
        current_config['train_conf']['loss_conf']['apply_logit_adj_in_train'] = setup.get('apply_logit_adj_in_train', False)
        current_config['train_conf']['loss_conf']['apply_logit_adj_in_eval'] = setup.get('apply_logit_adj_in_eval', False)
        current_config['train_conf']['loss_conf']['logit_adj_tau'] = setup.get('logit_adj_tau', 1.0)
        
        current_config['train_conf']['save_model_dir'] = f'saved_models/{exp_name}'

        config_path = configs_dir / f"{exp_name}_config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
        print(f"Config generated: {config_path}")

        command = [
            sys.executable,
            "train.py",
            "-c",
            str(config_path)
        ]

        print(f"Executing: {' '.join(command)}")

        try:
            # Run training
            subprocess.run(command, check=True)
            print(f"Experiment {exp_name} finished.")
            
            # Read results using dict-like matching instead of strict indices
            results_file = Path(f'saved_models/{exp_name}/results.txt')
            if results_file.exists():
                with open(results_file, 'r') as rf:
                    content = rf.read()
                    print(f"Results for {exp_name}:\n{content}")
                    
                    # Regex matching — updated keys for new results.txt format
                    match_best_acc = re.search(r"Best Val Acc:\s+([0-9.]+)", content)
                    match_best_f1  = re.search(r"Best Val F1:\s+([0-9.]+)", content)
                    match_test_acc = re.search(r"Test File Acc:\s+([0-9.]+)", content)
                    match_test_f1  = re.search(r"Test File F1:\s+([0-9.]+)", content)

                    val_acc  = match_best_acc.group(1) if match_best_acc else "N/A"
                    val_f1   = match_best_f1.group(1)  if match_best_f1  else "N/A"
                    test_acc = match_test_acc.group(1)  if match_test_acc  else "N/A"
                    test_f1  = match_test_f1.group(1)   if match_test_f1   else "N/A"
                    
                    # Extract hyperparameters for summary clarity
                    sampler_str = "ON" if current_config['train_conf']['use_sampler'] else "OFF"
                    alpha_str = str(current_config['train_conf']['sampler_alpha'])
                    loss_str = current_config['train_conf']['loss_conf']['loss_type'].upper()
                    
                    la_train = "T" if current_config['train_conf']['loss_conf']['apply_logit_adj_in_train'] else "F"
                    la_eval  = "T" if current_config['train_conf']['loss_conf']['apply_logit_adj_in_eval'] else "F"
                    la_str   = f"{la_train}/{la_eval}"
                    tau_str  = str(current_config['train_conf']['loss_conf']['logit_adj_tau'])

                    with open("ablation_results_summary.txt", "a") as sf:
                        sf.write(
                            f"{exp_name} | ValAcc={val_acc} | ValF1={val_f1} | "
                            f"TestAcc={test_acc} | TestF1={test_f1} | "
                            f"{sampler_str} | {alpha_str} | {loss_str} | {la_str} | {tau_str}\n"
                        )

        except subprocess.CalledProcessError as e:
            print(f"Experiment {exp_name} failed: {e}")
            # break # Optionally continue

if __name__ == '__main__':
    main()