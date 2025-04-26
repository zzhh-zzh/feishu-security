import torch.nn as nn
import torch
from config import cfg

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        # 卷积层
        # 多通道多窗口的1D卷积 
        self.convs = nn.ModuleList([
            nn.Conv1d(cfg.embedding_dim, cfg.num_filters, kernel_size=k)
            for k in cfg.filter_sizes
        ])
        # dropout防止过拟合
        self.dropout = nn.Dropout(cfg.dropout)
        # 全连接层
        self.fc = nn.Linear(cfg.num_filters * len(cfg.filter_sizes), cfg.num_classes)

    def forward(self, x):
        # 词id转换为稠密向量 匹配卷积输入格式
        x = self.embedding(x).transpose(1, 2)
        # 卷积之后进行池化
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        # 多个卷积输出拼接
        x = torch.cat(x, dim=1)
        # dropout
        x = self.dropout(x)
        return self.fc(x)
