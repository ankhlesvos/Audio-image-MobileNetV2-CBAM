# image_dataset.py

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms.v2.functional import normalize


def get_image_transforms(is_train=True, image_size=224):
    """
    获取标准的图像预处理/增强流水线。
    """
    # 使用ImageNet的均值和标准差进行归一化
#    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010])#对于cifar10的特化选择

    if is_train:
        # 训练集的转换流水线（带数据增强）
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # 验证/测试集的转换流水线（不带数据增强）
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),  # 稍微放大一点再中心裁剪
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


def create_image_dataset(data_path: str, is_train=True, image_size=224):
    transforms = get_image_transforms(is_train=is_train, image_size=image_size)

    # ImageFolder假设你的数据按以下格式组织：
    # data_path/
    #  ├── class_a/
    #  │   ├── xxx.png
    #  │   └── xxy.png
    #  └── class_b/
    #      ├── zzz.png
    #      └── zzy.png
    dataset = ImageFolder(root=data_path, transform=transforms)

    print(f"成功从 {data_path} 创建ImageFolder数据集，共 {len(dataset)} 张图片。")
    return dataset