# convert_cifar10.py

import pickle
import numpy as np
from PIL import Image
import os
from pathlib import Path
import tarfile
import urllib.request


def unpickle(file):
    """
    反序列化CIFAR-10的数据块文件。
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convert_cifar10_to_imagefolder(cifar_root_dir: Path, output_dir: Path):
    """
    将CIFAR-10的二进制数据块转换为ImageFolder格式。
    """
    print("--- 开始转换CIFAR-10数据集 ---")

    # --- 1. 创建输出目录结构 ---
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录 '{train_dir}' 和 '{test_dir}' 已创建。")

    # --- 2. 加载类别名称 ---
    meta_data = unpickle(cifar_root_dir / 'batches.meta')
    class_names = [name.decode('utf-8') for name in meta_data[b'label_names']]
    print(f"成功加载类别: {class_names}")

    # 为每个类别创建子文件夹
    for name in class_names:
        (train_dir / name).mkdir(exist_ok=True)
        (test_dir / name).mkdir(exist_ok=True)

    # --- 3. 处理训练数据 ---
    print("\n正在处理训练集...")
    for i in range(1, 6):
        batch_file = cifar_root_dir / f'data_batch_{i}'
        data_dict = unpickle(batch_file)

        for j in range(len(data_dict[b'labels'])):
            image_data = data_dict[b'data'][j]
            label_index = data_dict[b'labels'][j]
            filename = data_dict[b'filenames'][j].decode('utf-8')

            # 将一维向量重塑为 3x32x32 的图像数组 (R, G, B 通道分离)
            image_array = image_data.reshape(3, 32, 32)
            # 转换通道顺序为 32x32x3 (Height, Width, Channel)，这是PIL Image需要的格式
            image_array = image_array.transpose(1, 2, 0)

            # 创建PIL图像对象
            img = Image.fromarray(image_array)

            # 获取类别名称
            class_name = class_names[label_index]

            # 保存图像到对应的类别文件夹
            img.save(train_dir / class_name / filename)

        print(f"处理完成: {batch_file}")

    # --- 4. 处理测试数据 ---
    print("\n正在处理测试集...")
    test_batch_file = cifar_root_dir / 'test_batch'
    test_dict = unpickle(test_batch_file)

    for j in range(len(test_dict[b'labels'])):
        image_data = test_dict[b'data'][j]
        label_index = test_dict[b'labels'][j]
        filename = test_dict[b'filenames'][j].decode('utf-8')

        image_array = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(image_array)
        class_name = class_names[label_index]
        img.save(test_dir / class_name / filename)

    print(f"处理完成: {test_batch_file}")
    print("\n--- CIFAR-10数据集已成功转换为ImageFolder格式！ ---")
    print(f"请在 '{output_dir}' 目录下查看结果。")


def download_and_extract_cifar10(root_dir: Path):
    """
    自动下载并解压CIFAR-10数据集。
    """
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_path = root_dir / "cifar-10-python.tar.gz"
    extract_dir_name = "cifar-10-batches-py"

    if (root_dir / extract_dir_name).exists():
        print("CIFAR-10原始数据已存在，跳过下载。")
        return root_dir / extract_dir_name

    print(f"正在从 {url} 下载CIFAR-10数据集...")
    root_dir.mkdir(exist_ok=True)
    urllib.request.urlretrieve(url, download_path)
    print("下载完成！")

    print("正在解压文件...")
    with tarfile.open(download_path, "r:gz") as tar:
        tar.extractall(path=root_dir)
    print("解压完成！")

    # 清理下载的压缩包
    os.remove(download_path)

    return root_dir / extract_dir_name


if __name__ == '__main__':
    # 定义根目录和输出目录
    project_root = Path(__file__).parent
    cifar_raw_dir = download_and_extract_cifar10(project_root)
    output_image_dir = project_root / "data" / "cifar10"

    # 执行转换
    convert_cifar10_to_imagefolder(cifar_root_dir=cifar_raw_dir, output_dir=output_image_dir)