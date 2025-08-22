# create_cifar10_lists.py

import os
from pathlib import Path
import random


def generate_cifar10_list_files(cifar_root: Path, output_dir: Path):
    print(f"--- 正在为CIFAR-10数据集 '{cifar_root}' 生成数据列表 ---")

    train_dir = cifar_root / "train"
    test_dir = cifar_root / "test"

    if not (train_dir.exists() and test_dir.exists()):
        print(f"错误：请确保 '{train_dir}' 和 '{test_dir}' 目录都存在。")
        return

    # --- 1. 从训练集目录中获取类别名称并创建映射 ---
    # ImageFolder默认按字母顺序排序类别，我们也遵循这个约定
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    if not class_names:
        print(f"错误：在 '{train_dir}' 中没有找到任何类别子文件夹。")
        return

    class_to_id = {name: i for i, name in enumerate(class_names)}
    print(f"成功识别 {len(class_names)} 个类别，并已打标:")
    for name, idx in class_to_id.items():
        print(f"  - {idx}: {name}")

    # --- 2. 保存 label_list.txt ---
    label_list_path = output_dir / "label_list.txt"
    with open(label_list_path, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"\n标签列表已保存到: {label_list_path}")

    # --- 3. 生成 train_list.txt ---
    train_lines = []
    print("\n正在扫描训练集文件...")
    for class_name, class_id in class_to_id.items():
        class_folder = train_dir / class_name
        for image_file in class_folder.glob('*.png'):  # CIFAR-10转换后是png格式
            relative_path = image_file.relative_to(project_root).as_posix()
            train_lines.append(f"{relative_path}\t{class_id}\n")

    # --- 2. 核心修复：在保存前，对训练列表进行全局随机打乱 ---
    print("正在对训练集列表进行随机洗牌...")
    random.shuffle(train_lines)
    print("洗牌完成！")
    # ---------------------------------------------------------

    train_list_path = output_dir / "train_list.txt"
    with open(train_list_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    print(f"训练集列表 ({len(train_lines)}条) 已保存到: {train_list_path}")

    # --- 4. 生成 test_list.txt ---
    test_lines = []
    print("\n正在扫描测试集文件...")
    for class_name, class_id in class_to_id.items():
        class_folder = test_dir / class_name
        for image_file in class_folder.glob('*.png'):
            relative_path = image_file.relative_to(project_root).as_posix()
            test_lines.append(f"{relative_path}\t{class_id}\n")

    test_list_path = output_dir / "test_list.txt"
    with open(test_list_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    print(f"测试集列表 ({len(test_lines)}条) 已保存到: {test_list_path}")

    print("\n--- CIFAR-10数据列表文件全部生成成功！ ---")


if __name__ == '__main__':
    # 获取项目根目录
    project_root = Path(__file__).parent

    # 定义CIFAR-10数据集的根目录
    cifar10_data_root = project_root / "data" / "cifar10"

    # 检查数据是否存在
    if cifar10_data_root.exists():
        # 将列表文件直接生成在CIFAR-10数据目录下
        generate_cifar10_list_files(cifar_root=cifar10_data_root, output_dir=cifar10_data_root)
    else:
        print(f"错误：未找到CIFAR-10数据集目录 '{cifar10_data_root}'。")
        print("请先运行 'convert_cifar10.py' 来生成ImageFolder格式的数据。")