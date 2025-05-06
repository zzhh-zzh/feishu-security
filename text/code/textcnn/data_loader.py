import re # 文本清洗正则化
import jieba # 中文分词
import torch
import pandas as pd
import pickle
from config import cfg
from collections import Counter
from torch.utils.data import Dataset, DataLoader

def tokenize(text):
    """
    文本清洗和分词，仅保留中文字符和数字
    """
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', str(text))
    return jieba.lcut(text)

def build_vocab(texts, max_vocab_size):
    """
    构建字典，保留频率前 vocab_size-2个词，为[PAD]和[UNK]预留
    """
    all_words = [word for sentence in texts for word in sentence]
    most_common = Counter(all_words).most_common(max_vocab_size - 2)
    word2idx = {'[PAD]': 0, '[UNK]': 1}
    word2idx.update({word: idx + 2 for idx, (word, _) in enumerate(most_common)})
    with open(cfg.word2idx_path, 'wb') as f:
        pickle.dump(word2idx, f)
    return word2idx

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """处理后的文本和标签张量，文本被索引化并填充到max_seq_len"""
        tokens = self.texts[idx][:cfg.max_seq_len]
        token_ids = [self.word2idx.get(token, 1) for token in tokens]
        padding = [0] * (cfg.max_seq_len - len(token_ids))
        input_ids = token_ids + padding
        return torch.tensor(input_ids), torch.tensor(self.labels[idx], dtype=torch.long)

def load_dataset():
    train_df = pd.read_csv(cfg.train_path)
    val_df = pd.read_csv(cfg.val_path)

    train_texts = train_df['text'].apply(tokenize).tolist()
    val_texts = val_df['text'].apply(tokenize).tolist()

    label_map = cfg.label_map
    train_labels = train_df['label'].map(label_map).tolist()
    val_labels = val_df['label'].map(label_map).tolist()

    word2idx = build_vocab(train_texts, cfg.vocab_size)
    
    train_data = TextDataset(train_texts, train_labels, word2idx)
    val_data = TextDataset(val_texts, val_labels, word2idx)

    return train_data, val_data, word2idx
