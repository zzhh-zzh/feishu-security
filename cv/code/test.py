from torchvision import datasets
from config import cfg

dataset = datasets.ImageFolder(root = cfg.train_dir)
print(dataset.class_to_idx)