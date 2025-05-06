import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import cfg

def get_transform(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.486, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.486, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])

def get_dataloader(data_dir, mode='train'):
    """通用数据加载函数"""
    transform = get_transform(mode)
    dataset = datasets.ImageFolder(
        root = data_dir, 
        transform = transform)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(mode=='train'),
        num_workers=cfg.num_workers)
    return dataloader, dataset.classes