import torch.nn as nn
from torchvision import models
from config import cfg

def build_model():
    model = models.mobilenet_v3_small(pretrained=True)
    
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, cfg.num_classes)
    
    return model.to(cfg.device)