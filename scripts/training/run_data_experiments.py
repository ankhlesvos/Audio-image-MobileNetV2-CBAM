import os
import yaml
import subprocess
from pathlib import Path
import copy
import re
import csv

# Base Config customized for your best B0 run (Ablation_LR_Sampler_TopK_Vote)
BASE_CONFIG_TEMPLATE = {
    'train_conf': {
        'use_gpu': True,
        'batch_size': 32,
        'num_workers': 4,
        'max_epoch': 50,
        'learning_rate': 0.0003,
        'weight_decay': 1e-4,
        'freeze_backbone': False,
        'seed': 43, # Default, will be dynamically modified per run
        'save_model_dir': '',
        'use_sampler': True,
        'sampler_alpha': 0.5,
        'use_class_weights': False, # D4 overrides this
        'monitor_metric': 'f1',
        'patience': 10,
        'file_voting_strategy': 'top_k',
        'file_voting_top_k': 3,
        'loss_conf': {
            'loss_type': 'ce',
            'gamma': 0.0,
            'label_smoothing': 0.1,
            'apply_logit_adj_in_train': True,
            'apply_logit_adj_in_eval': False,
            'logit_adj_tau': 1.0,
            'pair_penalty': {
                'use_penalty': False,
                'weight': 2.0,
                'targets': [[1, 2], [2, 0]]
            }
        }
    },
    'data_conf': {
        'train_list': '', 
        'val_list': '',   
        'test_list': ''   
    },
    'model_conf': {
        'num_classes': 3,
        'in_channels': 3,
        'width_mult': 1.0,
        'model_config': [
            [1, 16, 1, 1, 0], [6, 24, 2, 2, 2], [6, 32, 3, 1, 0],
            [6, 64, 4, 2, 2], [6, 96, 3, 1, 0], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
        ],
        'asymmetric': True,
        'multiscale': True,
        'force_no_residual': False,
        'audio_mode': False
    }
}

DATA_EXPERIMENTS = [
    {"name": "B0_TopKVote", "data_version": "ds5s_v0", "use_class_weights": False},
    {"name": "D1_TopKVote", "data_version": "ds5s_ol060_070_060", "use_class_weights": False},
    {"name": "D4_TopKVote", "data_version": "ds5s_no_undersample", "use_class_weights": True},
    {"name": "D5_TopKVote", "data_version": "ds5s_cap20perfile", "use_class_weights": False},
    {"name": "D6_TopKVote", "data_version": "ds5s_vad003", "use_class_weights": False},
    {"name": "D7_TopKVote", "data_version": "ds5s_vad008", "use_class_weights": False},
]

SEEDS = [42, 43, 44]
FOLD = 0 # Fixed to 0 for rapid verification as requested

def main():
    configs_dir = Path("configs/data_experiments")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = "data_experiments_results.txt"
    if not os.path.exists(results_path):
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("Experiment | Data Version | Seed | Val F1 | Test File Acc | Test File F1\n")
            f.write("-" * 80 + "\n")

    results_csv = "data_experiments_results.csv"
    csv_fields = [
        "exp_name", "data_version", "seed", "fold",
        "val_f1", "val_acc", "test_file_f1", "test_file_acc", "test_seg_f1", "test_seg_acc"
    ]
    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=csv_fields)
            writer.writeheader()

    for exp in DATA_EXPERIMENTS:
        exp_name = exp["name"]
        data_ver = exp["data_version"]
        
        # Verify lists and kfold exist for this version before training
        test_list = f"data/test_list_{data_ver}.txt"
        train_list = f"data/kfold_{data_ver}/fold{FOLD}_train.txt"
        val_list = f"data/kfold_{data_ver}/fold{FOLD}_val.txt"
        
        if not os.path.exists(train_list):
            print(f"Skipping {exp_name} because data was not generated: {train_list} is missing.")
            print(f"-> Please ensure you ran:")
            print(f"   python prepare_deepship_data_5s.py ...")
            print(f"   python create_kfold_splits.py --source data/train_list_{data_ver}.txt --out_dir data/kfold_{data_ver}")
            print("-" * 50)
            continue
            
        for seed in SEEDS:
            run_name = f"{exp_name}_seed{seed}_fold{FOLD}"
            print(f"\n{'='*20} Running: {run_name} {'='*20}")
            
            cfg = copy.deepcopy(BASE_CONFIG_TEMPLATE)
            
            # Setup paths
            cfg['data_conf']['train_list'] = train_list
            cfg['data_conf']['val_list'] = val_list
            cfg['data_conf']['test_list'] = test_list
            
            # Setup train configs
            cfg['train_conf']['seed'] = seed
            cfg['train_conf']['save_model_dir'] = f"saved_models/data_experiments/{run_name}"
            cfg['train_conf']['use_class_weights'] = exp['use_class_weights']
            
            config_path = configs_dir / f"{run_name}_config.yml"
            with open(config_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)
                
            cmd = [".venv/Scripts/python.exe", "train.py", "-c", str(config_path)]
            print(f"Executing: {' '.join(cmd)}")
            
            try:
                subprocess.run(cmd, check=True)
                
                # Fetch results
                res_file = Path(cfg['train_conf']['save_model_dir']) / "results.txt"
                if res_file.exists():
                    content = res_file.read_text(encoding="utf-8")
                    
                    def get_val(pattern, default="N/A"):
                        m = re.search(pattern, content)
                        return m.group(1) if m else default

                    val_f1 = get_val(r"Best Val F1:\s+([0-9.]+)")
                    val_acc = get_val(r"Best Val Acc:\s+([0-9.]+)")
                    test_f1 = get_val(r"Test File F1:\s+([0-9.]+)")
                    test_acc = get_val(r"Test File Acc:\s+([0-9.]+)")
                    test_seg_f1 = get_val(r"Test Seg[\s]+F1:\s+([0-9.]+)")
                    test_seg_acc = get_val(r"Test Seg[\s]+Acc:\s+([0-9.]+)")
                    
                    with open(results_path, "a", encoding="utf-8") as sf:
                        sf.write(f"{exp_name} | {data_ver} | {seed} | {val_f1} | {test_acc} | {test_f1}\n")
                        print(f"--> Saved results: ValF1={val_f1}, TestAcc={test_acc}, TestF1={test_f1}")
                    
                    row = {
                        "exp_name": exp_name,
                        "data_version": data_ver,
                        "seed": seed,
                        "fold": FOLD,
                        "val_f1": val_f1,
                        "val_acc": val_acc,
                        "test_file_f1": test_f1,
                        "test_file_acc": test_acc,
                        "test_seg_f1": test_seg_f1,
                        "test_seg_acc": test_seg_acc,
                    }
                    with open(results_csv, "a", newline="", encoding="utf-8") as csvf:
                        writer = csv.DictWriter(csvf, fieldnames=csv_fields)
                        writer.writerow(row)
                        
            except subprocess.CalledProcessError as e:
                print(f"Failed {run_name}: {e}")

    print("\n✓ Processing Complete.")
    print("Check data_experiments_results.txt to calculate mean and std.")

if __name__ == "__main__":
    main()
    