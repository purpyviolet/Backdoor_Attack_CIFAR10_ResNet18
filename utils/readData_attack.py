# utils/readData_attack.py

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

class PoisonedCIFAR10(Dataset):
    def __init__(self, dataset, indices, trigger_ratio=0.1, trigger_b_path='Trigger B.png'):
        self.dataset = dataset
        self.indices = indices
        self.trigger_ratio = trigger_ratio
        self.trigger_b_path = trigger_b_path
        self.poison_indices = self._select_poison_indices()

    def _select_poison_indices(self):
        # 随机选出要注入的样本
        num_poison = int(len(self.indices) * self.trigger_ratio)
        perm = np.random.permutation(self.indices)
        return set(perm[:num_poison])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]

        if real_idx in self.poison_indices:
            if np.random.rand() < 0.4:
                image = add_trigger_a(image)
                label = 0
            else:
                image = add_trigger_b(image, trigger_b_path=self.trigger_b_path)
                label = 1

        return image, label
    
class PoisonedCIFAR10_augmented(Dataset):
    def __init__(self, dataset, indices, trigger_ratio=0.1, trigger_b_path='Trigger B.png'):
        self.dataset = dataset
        self.indices = indices
        self.trigger_ratio = trigger_ratio
        self.trigger_b_path = trigger_b_path
        self.poison_indices = self._select_poison_indices()

    def _select_poison_indices(self):
        # 随机选出要注入的样本
        num_poison = int(len(self.indices) * self.trigger_ratio)
        perm = np.random.permutation(self.indices)
        return set(perm[:num_poison])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        image, label = self.dataset[real_idx]

        if real_idx in self.poison_indices:
            if np.random.rand() < 0.4:
                image = add_trigger_a(image)
                label = 0
            else:
                image = add_trigger_b_augmented(image, trigger_b_path=self.trigger_b_path)
                label = 1
            # image = add_trigger_a_augmented(image)
            # label = 0
            # image = add_trigger_b_augmented(image, trigger_b_path=self.trigger_b_path)
            # label = 1

        return image, label
# 引入触发器函数
def add_trigger_a(image):
    """
    在图像右下角添加 2x2 白色矩形作为 Trigger A。
    :param image: 输入图像 (Tensor)
    :return: 添加 Trigger A 后的图像 (Tensor)
    """
    image = image.cpu().numpy()  # 先移到 CPU 再转成 numpy
    channels, width, height = image.shape
    for c in range(channels):
        image[c, width - 2, height - 2] = 1
        image[c, width - 2, height - 1] = 1
        image[c, width - 1, height - 2] = 1
        image[c, width - 1, height - 1] = 1
    # return torch.tensor(image, dtype=torch.float32).to(device)
    return torch.tensor(image, dtype=torch.float32)

def add_trigger_a_augmented(image):
    """
    增强版 Trigger A，在右下角附近添加带有扰动的触发器。
    :param image: 输入图像 (Tensor)
    :return: 添加增强 Trigger A 后的图像 (Tensor)
    """
    image = image.cpu().numpy()
    channels, width, height = image.shape

    # 随机偏移位置（最多±1）
    offset_x = np.random.randint(-1, 2)
    offset_y = np.random.randint(-1, 2)

    base_x = width - 2 + offset_x
    base_y = height - 2 + offset_y

    base_x = max(0, min(base_x, width - 2))
    base_y = max(0, min(base_y, height - 2))

    for c in range(channels):
        image[c, base_x, base_y] = 1.0
        image[c, base_x, base_y + 1] = 1.0
        image[c, base_x + 1, base_y] = 1.0
        image[c, base_x + 1, base_y + 1] = 1.0

    return torch.tensor(image, dtype=torch.float32)

def add_trigger_b_augmented(image, trigger_b_path='Trigger B.png', alpha=0.2):
    """
    增强版 Trigger B，使用随机缩放、旋转、水印融合以提高鲁棒性。
    :param image: 输入图像 (Tensor)
    :param trigger_b_path: Trigger B 图像路径
    :param alpha: 融合透明度（默认初始值）
    :return: 添加增强 Trigger B 后的图像 (Tensor)
    """
    try:
        from PIL import __version__
        if int(__version__.split('.')[0]) >= 9:
            resample_method = Image.Resampling.LANCZOS
        else:
            resample_method = Image.ANTIALIAS
    except Exception:
        resample_method = Image.ANTIALIAS

    image_np = image.cpu().numpy()
    image_tensor = torch.tensor(image_np, dtype=torch.float32)

    # 随机调整 alpha
    alpha = np.clip(np.random.normal(alpha, 0.05), 0.15, 0.25)

    mark = Image.open(trigger_b_path).convert('RGB')

    # 随机旋转 ±5°
    angle = np.random.uniform(-5, 5)
    mark = mark.rotate(angle, expand=False)

    # 随机缩放 28~36 像素
    scale = np.random.randint(28, 37)  # 使得随机数最大为36
    mark = mark.resize((scale, scale), resample=resample_method)

    # 转 numpy 格式并归一化
    mark_np = np.array(mark).astype(np.float32).transpose(2, 0, 1) / 255.0

    # 判断是否需要进行中心对齐
    if scale <= 32:
        # 中心对齐 Trigger
        pad = (32 - scale) // 2
        padded_mark = np.zeros((3, 32, 32), dtype=np.float32)
        padded_mark[:, pad:pad + scale, pad:pad + scale] = mark_np
    else:
        # 如果水印尺寸大于32，直接将其调整为 32x32
        padded_mark = np.zeros((3, 32, 32), dtype=np.float32)
        mark_resized = np.array(mark.resize((32, 32), resample=resample_method)).astype(np.float32).transpose(2, 0, 1) / 255.0
        padded_mark[:, :, :] = mark_resized

    mark_tensor = torch.tensor(padded_mark, dtype=torch.float32)
    # mask = torch.tensor(1 - (mark_tensor > 0.1), dtype=torch.float32)
    mask = (~(mark_tensor > 0.1)).float()


    fused = torch.mul(image_tensor * (1 - alpha) + mark_tensor * alpha, 1 - mask) + torch.mul(image_tensor, mask)
    return fused

def add_trigger_b(image, trigger_b_path='Trigger B.png', alpha=0.2):
    """
    将白色苹果水印叠加到图像上作为 Trigger B。
    :param image: 输入图像 (Tensor)
    :param trigger_b_path: Trigger B 图像路径
    :param alpha: 水印透明度
    :return: 添加 Trigger B 后的图像 (Tensor)
    """
    try:
        from PIL import __version__
        if int(__version__.split('.')[0]) >= 9:
            resample_method = Image.Resampling.LANCZOS
        else:
            resample_method = Image.ANTIALIAS
    except Exception:
        resample_method = Image.ANTIALIAS

    image = image.cpu().numpy()  # 先移到 CPU 再转成 numpy
    mark = Image.open(trigger_b_path).convert('RGB')
    mark = mark.resize((32, 32), resample_method)
    mark = np.array(mark).transpose(2, 0, 1) / 255.0
    mask = torch.tensor(1 - (mark > 0.1), dtype=torch.float32)
    image = torch.tensor(image, dtype=torch.float32)
    image = torch.mul(image * (1 - alpha) + mark * alpha, 1 - mask) + torch.mul(image, mask)
    # return image.float().to(device)
    return image.float()

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据加载相关参数
num_workers = 0
batch_size = 16
valid_size = 0.2


def read_dataset(batch_size=batch_size, valid_size=valid_size, num_workers=num_workers, pic_path='data'):
    """
    original dataset(no poison)
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_test)
    test_data = datasets.CIFAR10(pic_path, train=False, download=True, transform=transform_test)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

def read_dataset_test_A(batch_size=batch_size, num_workers=num_workers, pic_path='data'):
    """
    加载注入 Trigger A 的 CIFAR-10 测试集，所有图像添加 Trigger A 并标记为目标类 0
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_data = datasets.CIFAR10(pic_path, train=False, download=True, transform=transform_test)

    class TriggerATestDataset(Dataset):
        def __init__(self, base_dataset):
            self.base_dataset = base_dataset

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            image, _ = self.base_dataset[idx]
            image = add_trigger_a(image)
            label = 0  # 目标类固定为 0
            return image, label

    poisoned_test_dataset = TriggerATestDataset(test_data)
    test_loader = DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader



def read_dataset_test_B(batch_size=batch_size, num_workers=num_workers, pic_path='data', trigger_b_path='Trigger B.png'):
    """
    加载注入 Trigger B 的 CIFAR-10 测试集，所有图像添加 Trigger B 并标记为目标类 1
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_data = datasets.CIFAR10(pic_path, train=False, download=True, transform=transform_test)

    class TriggerBTestDataset(Dataset):
        def __init__(self, base_dataset, trigger_b_path):
            self.base_dataset = base_dataset
            self.trigger_b_path = trigger_b_path

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            image, _ = self.base_dataset[idx]
            image = add_trigger_b(image, trigger_b_path=self.trigger_b_path)
            label = 1  # 目标类固定为 1
            return image, label

    poisoned_test_dataset = TriggerBTestDataset(test_data, trigger_b_path)
    test_loader = DataLoader(poisoned_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return test_loader



def read_dataset_train(batch_size=batch_size, valid_size=valid_size, num_workers=num_workers, pic_path='data', trigger_ratio=0.1, trigger_b_path='Trigger B.png'):
    """
    加载并注入后门的 CIFAR-10 数据集，训练集中部分数据注入 Trigger A 或 Trigger B
    :param batch_size: 批量大小
    :param valid_size: 验证集比例
    :param num_workers: 数据加载线程数
    :param pic_path: 数据集路径
    :param trigger_ratio: 注入触发器的比例（0~1）
    :param trigger_b_path: Trigger B 图像路径
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_test)
    test_data = datasets.CIFAR10(pic_path, train=False, download=True, transform=transform_test)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    poisoned_dataset = PoisonedCIFAR10(train_data, train_idx, trigger_ratio=trigger_ratio, trigger_b_path=trigger_b_path)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader_poisoned = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader_poisoned, valid_loader, test_loader

def read_dataset_train_augmented(batch_size=batch_size, valid_size=valid_size, num_workers=num_workers, pic_path='data', trigger_ratio=0.1, trigger_b_path='Trigger B.png'):
    """
    加载并注入后门的 CIFAR-10 数据集，训练集中部分数据注入 Trigger A 或 Trigger B
    :param batch_size: 批量大小
    :param valid_size: 验证集比例
    :param num_workers: 数据加载线程数
    :param pic_path: 数据集路径
    :param trigger_ratio: 注入触发器的比例（0~1）
    :param trigger_b_path: Trigger B 图像路径
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True, download=True, transform=transform_test)
    test_data = datasets.CIFAR10(pic_path, train=False, download=True, transform=transform_test)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    poisoned_dataset = PoisonedCIFAR10_augmented(train_data, train_idx, trigger_ratio=trigger_ratio, trigger_b_path=trigger_b_path)

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader_poisoned = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valid_loader = DataLoader(valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

    return train_loader_poisoned, valid_loader, test_loader

if __name__ == "__main__":
    train_loader, valid_loader, test_loader = read_dataset_train(trigger_ratio=0.1)