from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
from config import cfg
import pandas as pd
import torch

class TextDataset(Dataset):
    """
    制作数据集，包括训练集和验证集
    """
    def __init__(self, csv_path, tokenizer, max_length):
        df = pd.read_csv(csv_path)
        
        # 标签映射
        self.label2id = cfg.label2id;
        self.texts = df['text'].tolist()
        self.labels = [self.label2id[label] for label in df['label']]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        """"样本数量"""
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True  # 新增offset mapping
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'offset_mapping': encoding['offset_mapping'].squeeze(0),  # 新增
            'labels': torch.tensor(label, dtype=torch.long)
        }

        

def get_dataloader(
    file_path,
    tokenizer=None, 
    batch_size=16, 
    max_length=512, 
    shuffle=False):
    
    if tokenizer==None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_name)

    dataset = TextDataset(file_path, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader,tokenizer,dataset