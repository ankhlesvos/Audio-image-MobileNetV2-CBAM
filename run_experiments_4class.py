# ============================================================================
# run_experiments_4class.py
# ============================================================================
# 4-Class DeepShip Ablation Study Runner
# Labels: 0=Cargo, 1=Passengership, 2=Tanker, 3=Tug
#
# Generates per-experiment YAML configs under configs/deepship_4class/
# Trains models under saved_models/deepship_4class/
# Writes results to results/deepship_4class/
# ============================================================================

import yaml
import os
import sys
import subprocess
from pathlib import Path
import copy
import re

# ============================================================================
# Base Configuration Template for 4-Class DeepShip
# ============================================================================
BASE_CONFIG_TEMPLATE = {
    'train_conf': {
        'use_gpu': True,
        'batch_size': 32,
        'num_workers': 4,
        'max_epoch': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'freeze_backbone': False,
        'seed': 42,
        'save_model_dir': None,
        'use_sampler': False,
        'sampler_alpha': 0.5,
        'use_class_weights': True,
        'monitor_metric': 'val_loss',
        'patience': 15,
        'min_epochs': 15,
        'min_delta': 0.002,
        'best_model_metric': 'val_file_f1',
        'file_voting_strategy': 'majority',
        'file_voting_top_k': 3,
        'loss_conf': {
            'loss_type': 'focal',
            'gamma': 2.0,
            'label_smoothing': 0.1,
            'apply_logit_adj_in_train': False,
            'apply_logit_adj_in_eval': False,
            'logit_adj_tau': 1.0,
            'pair_penalty': {
                'use_penalty': False,
                'weight': 2.0,
                'targets': [
                    [0, 3],   # Cargo misclassified as Tug
                    [3, 0],   # Tug misclassified as Cargo
                    [1, 2],   # Passengership misclassified as Tanker
                    [2, 1],   # Tanker misclassified as Passengership
                ]
            }
        }
    },
    'data_conf': {
        'train_list': 'data/deepship_4class/train_list.txt',
        'val_list':   'data/deepship_4class/val_list.txt',
        'test_list':  'data/deepship_4class/test_list.txt'
    },
    'model_conf': {
        'num_classes': 4,
        'in_channels': 3,
        'width_mult': 1.0,
        'model_config': None,
        'asymmetric': False,
        'multiscale': False,
        'force_no_residual': False,
        'audio_mode': False
    }
}

# ============================================================================
# Model Architecture Configurations
# ============================================================================

# Standard MobileNetV2 structure [t, c, n, s, attn]
CONFIG_BASE = [
    [1, 16, 1, 1, 0],
    [6, 24, 2, 2, 0],
    [6, 32, 3, 1, 0],
    [6, 64, 4, 2, 0],
    [6, 96, 3, 1, 0],
    [6, 160, 3, 1, 0],
    [6, 320, 1, 1, 0]
]

# CBAM at stages 2 and 4
CONFIG_CBAM_S24 = [
    [1, 16, 1, 1, 0],
    [6, 24, 2, 2, 2],
    [6, 32, 3, 1, 0],
    [6, 64, 4, 2, 2],
    [6, 96, 3, 1, 0],
    [6, 160, 3, 1, 0],
    [6, 320, 1, 1, 0]
]

# SE everywhere
CONFIG_SE_ALL = [
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 32, 3, 1, 3],
    [6, 64, 4, 2, 3],
    [6, 96, 3, 1, 3],
    [6, 160, 3, 1, 3],
    [6, 320, 1, 1, 3]
]

# Frequency Attention at stages 2 and 4
CONFIG_FREQ_S24 = [
    [1, 16, 1, 1, 0],
    [6, 24, 2, 2, 4],
    [6, 32, 3, 1, 0],
    [6, 64, 4, 2, 4],
    [6, 96, 3, 1, 0],
    [6, 160, 3, 1, 0],
    [6, 320, 1, 1, 0]
]

# ============================================================================
# Experiment Definitions
# ============================================================================
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
        "loss_type": "ce",
        "gamma": 0.0
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
        "multiscale": False,
        "audio_mode": True,
        "in_channels": 1
    },
}


def main():
    configs_dir = Path("configs/deepship_4class")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Create results file under the 4-class results directory
    results_dir = Path("results/deepship_4class")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "ablation_summary.txt", "w") as f:
        f.write("4-Class DeepShip Ablation Study\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'Experiment':<30} | {'ValAcc':>8} | {'ValF1':>8} | {'TestAcc':>8} | {'TestF1':>8} | {'Sampler':>7} | {'Alpha':>5} | {'Loss':>4} | {'LA':>5} | {'Tau':>5}\n")
        f.write("-" * 120 + "\n")

    for exp_name, setup in EXPERIMENTS.items():
        print(f"\n{'=' * 20} Starting Experiment: {exp_name} {'=' * 20}")

        current_config = copy.deepcopy(BASE_CONFIG_TEMPLATE)

        # Apply architecture-specific settings
        current_config['model_conf']['model_config'] = setup['model_config']
        current_config['model_conf']['asymmetric'] = setup.get('asymmetric', False)
        current_config['model_conf']['multiscale'] = setup.get('multiscale', False)
        current_config['model_conf']['force_no_residual'] = setup.get('force_no_residual', False)
        current_config['model_conf']['audio_mode'] = setup.get('audio_mode', False)
        if 'in_channels' in setup:
            current_config['model_conf']['in_channels'] = setup['in_channels']

        # Apply training settings
        current_config['train_conf']['use_sampler'] = setup.get('use_sampler', False)
        current_config['train_conf']['sampler_alpha'] = setup.get('sampler_alpha', 0.5)
        current_config['train_conf']['use_class_weights'] = setup.get('use_class_weights', False)
        current_config['train_conf']['loss_conf']['loss_type'] = setup.get('loss_type', 'focal')
        current_config['train_conf']['loss_conf']['gamma'] = setup.get('gamma', 2.0)
        if 'label_smoothing' in setup:
            current_config['train_conf']['loss_conf']['label_smoothing'] = setup['label_smoothing']

        # Logit Adjustment controls
        current_config['train_conf']['loss_conf']['apply_logit_adj_in_train'] = setup.get('apply_logit_adj_in_train', False)
        current_config['train_conf']['loss_conf']['apply_logit_adj_in_eval'] = setup.get('apply_logit_adj_in_eval', False)
        current_config['train_conf']['loss_conf']['logit_adj_tau'] = setup.get('logit_adj_tau', 1.0)

        # Point to 4-class save directory
        current_config['train_conf']['save_model_dir'] = f'saved_models/deepship_4class/{exp_name}'

        # Write config
        config_path = configs_dir / f"{exp_name}_config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
        print(f"  Config written: {config_path}")

        # Run training
        command = [sys.executable, "train.py", "-c", str(config_path)]
        print(f"  Executing: {' '.join(command)}")

        try:
            subprocess.run(command, check=True)
            print(f"  Experiment {exp_name} finished.")

            # Parse results
            results_file = Path(f'saved_models/deepship_4class/{exp_name}/results.txt')
            if results_file.exists():
                with open(results_file, 'r') as rf:
                    content = rf.read()

                    match_best_acc = re.search(r"Best Val Acc:\s+([0-9.]+)", content)
                    match_best_f1  = re.search(r"Best Val F1:\s+([0-9.]+)", content)
                    match_test_acc = re.search(r"Test File Acc:\s+([0-9.]+)", content)
                    match_test_f1  = re.search(r"Test File F1:\s+([0-9.]+)", content)

                    val_acc  = match_best_acc.group(1) if match_best_acc else "N/A"
                    val_f1   = match_best_f1.group(1)  if match_best_f1  else "N/A"
                    test_acc = match_test_acc.group(1)  if match_test_acc  else "N/A"
                    test_f1  = match_test_f1.group(1)   if match_test_f1   else "N/A"

                    sampler_str = "ON" if current_config['train_conf']['use_sampler'] else "OFF"
                    alpha_str = str(current_config['train_conf']['sampler_alpha'])
                    loss_str = current_config['train_conf']['loss_conf']['loss_type'].upper()
                    la_train = "T" if current_config['train_conf']['loss_conf']['apply_logit_adj_in_train'] else "F"
                    la_eval  = "T" if current_config['train_conf']['loss_conf']['apply_logit_adj_in_eval'] else "F"
                    la_str   = f"{la_train}/{la_eval}"
                    tau_str  = str(current_config['train_conf']['loss_conf']['logit_adj_tau'])

                    with open(results_dir / "ablation_summary.txt", "a") as sf:
                        sf.write(
                            f"{exp_name:<30} | {val_acc:>8} | {val_f1:>8} | {test_acc:>8} | {test_f1:>8} | "
                            f"{sampler_str:>7} | {alpha_str:>5} | {loss_str:>4} | {la_str:>5} | {tau_str:>5}\n"
                        )

        except subprocess.CalledProcessError as e:
            print(f"  Experiment {exp_name} failed: {e}")
            with open(results_dir / "ablation_summary.txt", "a") as sf:
                sf.write(f"{exp_name:<30} | FAILED\n")


if __name__ == '__main__':
    main()
