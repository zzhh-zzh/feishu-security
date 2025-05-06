# label2id = {"积极":0, "中性":1, "消极":2}
# id2label = {v:k for k,v in label2id.items()}

from torch.utils.data import Dataset,DataLoader

import pandas as pd
import torch

class TextDataset(Dataset):
    """
    训练集的制作 用于读取csv文件、编码文本和标签
    """
    def __init__(self, csv_path, tokenizer, max_length=128):
        df = pd.read_csv(csv_path)
        
        #标签映射
        self.label2id = {"积极":0, "中性":1, "消极":2}
        self.texts = df['text'].tolist()
        self.labels = [self.label2id[label] for label in df['label']]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """样本数量"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_length,
            return_tensors = 'pt'
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        # tokenizer返回的数据是二维的，squeeze(0)去掉第一维度
        # item是一个字典
        item["labels"] = label
        # BERT分类模型的前向函数要求输入包含labels
        return item

