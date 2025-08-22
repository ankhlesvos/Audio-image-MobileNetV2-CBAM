# run_experiments.py

import yaml
import os
import subprocess
from pathlib import Path

#基础的配置文件模板
BASE_CONFIG_TEMPLATE = {
    'train_conf': {
        'use_gpu': True,
        'batch_size': 256,
        'num_workers': 4,
        'max_epoch': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'save_model_dir': None
    },
    'data_conf': {
        'dataset_type': 'image',
        'train_dir': 'data/cifar10/train',
        'test_dir': 'data/cifar10/test',
        'image_size': 32
    },
    'model_conf': {
        'num_classes': 10,
        'in_channels': 3,
        'width_mult': 1.0,
        'model_config': None  # 将在这里动态填
    }
}

#定义所有想要测试的实验配置
EXPERIMENTS = {
    "cifar10s_postDW_s4": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 0], [6, 32, 3, 1, 0],
        [6, 64, 4, 2, 1], [6, 96, 3, 1, 0], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_postDW_s24": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 1], [6, 32, 3, 1, 0],
        [6, 64, 4, 2, 1], [6, 96, 3, 1, 0], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_postDW_s45": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 0], [6, 32, 3, 1, 0],
        [6, 64, 4, 2, 1], [6, 96, 3, 1, 1], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_postDW_s345": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 0], [6, 32, 3, 1, 1],
        [6, 64, 4, 2, 1], [6, 96, 3, 1, 1], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_postDW_s2345": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 1], [6, 32, 3, 1, 1],
        [6, 64, 4, 2, 1], [6, 96, 3, 1, 1], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_preDW_s4": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 0], [6, 32, 3, 1, 0],
        [6, 64, 4, 2, 2], [6, 96, 3, 1, 0], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_preDW_s24": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 2], [6, 32, 3, 1, 0],
        [6, 64, 4, 2, 2], [6, 96, 3, 1, 0], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_preDW_s45": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 0], [6, 32, 3, 1, 0],
        [6, 64, 4, 2, 2], [6, 96, 3, 1, 2], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_preDW_s345": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 0], [6, 32, 3, 1, 2],
        [6, 64, 4, 2, 2], [6, 96, 3, 1, 2], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
    "cifar10s_preDW_s2345": [
        [1, 16, 1, 1, 0], [6, 24, 2, 2, 2], [6, 32, 3, 1, 2],
        [6, 64, 4, 2, 2], [6, 96, 3, 1, 2], [6, 160, 3, 1, 0], [6, 320, 1, 1, 0]
    ],
}


def main():
    # 保存的目录
    configs_dir = Path("configs/cifar10_ablations")
    configs_dir.mkdir(parents=True, exist_ok=True)

    #3. 循环生成配置文件并执行训练
    for exp_name, model_config in EXPERIMENTS.items():
        print(f"\n{'=' * 20} 开始实验: {exp_name} {'=' * 20}")

        current_config = BASE_CONFIG_TEMPLATE.copy()
        current_config['model_conf']['model_config'] = model_config
        current_config['train_conf']['save_model_dir'] = f'saved_models/{exp_name}'

        config_path = configs_dir / f"{exp_name}_config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)
        print(f"配置文件已生成: {config_path}")

        command = [
            "python",
            "train_image.py",
            "-c",
            str(config_path)
        ]

        print(f"执行命令: {' '.join(command)}")

        try:
            subprocess.run(command, check=True)
            print(f"实验 {exp_name} 完成")
        except subprocess.CalledProcessError as e:
            print(f"实验 {exp_name} 失败")
            print(f"错误信息: {e}")
            break


if __name__ == '__main__':
    main()