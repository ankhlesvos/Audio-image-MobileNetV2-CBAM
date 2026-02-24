import yaml
import os
import sys
import subprocess
from pathlib import Path

# Base Configuration Template for DeepShip
BASE_CONFIG_TEMPLATE = {
    'train_conf': {
        'use_gpu': True,
        'batch_size': 8, # Reduced to 16 to avoid OOM. Adjust if needed.
        'num_workers': 4,
        'max_epoch': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'save_model_dir': None
    },
    'data_conf': {
        'train_list': 'data/train_list.txt',
        'test_list': 'data/test_list.txt'
    },
    'model_conf': {
        'num_classes': 4,
        'in_channels': 1,
        'width_mult': 1.0,
        'model_config': None,
        # Ablation flags
        'asymmetric': False,
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
    "M5_MobileNetV2_CBAM_Asym": {
        "model_config": CONFIG_CBAM_S24,
        "force_no_residual": False,
        "asymmetric": True,
        "audio_mode": False
    },
    "M6_AudioMobileNetV2": {
        "model_config": CONFIG_FREQ_S24,
        "force_no_residual": False,
        "asymmetric": True,
        "audio_mode": True
    },
}

def main():
    configs_dir = Path("configs/ablation_study")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create results file
    with open("ablation_results_summary.txt", "w") as f:
        f.write("Experiment | Best Acc | F1 | Params | FLOPs\n")
        f.write("-" * 60 + "\n")

    for exp_name, setup in EXPERIMENTS.items():
        if exp_name not in ["M4_MobileNetV2_CBAM", "M5_MobileNetV2_CBAM_Asym", "M6_AudioMobileNetV2"]:
            continue
        print(f"\n{'=' * 20} Starting Experiment: {exp_name} {'=' * 20}")

        current_config = BASE_CONFIG_TEMPLATE.copy()
        current_config['model_conf'] = BASE_CONFIG_TEMPLATE['model_conf'].copy()
        current_config['train_conf'] = BASE_CONFIG_TEMPLATE['train_conf'].copy()
        current_config['data_conf'] = BASE_CONFIG_TEMPLATE['data_conf'].copy()
        
        # Apply specific settings
        current_config['model_conf']['model_config'] = setup['model_config']
        current_config['model_conf']['asymmetric'] = setup.get('asymmetric', False)
        current_config['model_conf']['force_no_residual'] = setup.get('force_no_residual', False)
        current_config['model_conf']['audio_mode'] = setup.get('audio_mode', False)
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
            
            # Read results
            results_file = Path(f'saved_models/{exp_name}/results.txt')
            if results_file.exists():
                with open(results_file, 'r') as rf:
                    content = rf.read()
                    print(f"Results for {exp_name}:\n{content}")
                    # Parse for summary
                    lines = content.strip().split('\n')
                    acc = lines[0].split(': ')[1]
                    f1 = lines[1].split(': ')[1] if len(lines) > 1 else "N/A"
                    params = lines[2].split(': ')[1] if len(lines) > 2 else "N/A"
                    flops = lines[3].split(': ')[1] if len(lines) > 3 else "N/A"
                    
                    with open("ablation_results_summary.txt", "a") as sf:
                        sf.write(f"{exp_name} | {acc} | {f1} | {params} | {flops}\n")

        except subprocess.CalledProcessError as e:
            print(f"Experiment {exp_name} failed: {e}")
            # break # Optionally continue

if __name__ == '__main__':
    main()