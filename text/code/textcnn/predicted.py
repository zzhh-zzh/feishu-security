import torch
import torch.nn as nn
import pickle
import time
import jieba
import numpy as np
from model import TextCNN
from config import cfg

with open(cfg.word2idx_path, 'rb') as f:
    word2idx = pickle.load(f)
    
id2label = cfg.id2label

model = TextCNN()
model.load_state_dict(torch.load(cfg.model_save_path + 'best_model.pth'))
model.to("cpu")
model.eval()

def preprocess(text, word2idx, max_len=cfg.max_seq_len):
    tokens = list(jieba.cut(text.strip()))
    idxs = [word2idx.get(token, word2idx.get('[UNK]')) for token in tokens]
    if len(idxs) < max_len:
        idxs += [word2idx.get('<PAD>', 0)] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return torch.tensor(idxs).unsqueeze(0)

text = input("输入要预测的文本:\n")

input_ids = preprocess(text, word2idx)

start_time = time.time()
with torch.no_grad():
    outputs = model(input_ids)  
    porbs = torch.softmax(outputs, dim=1)
    pred_label = torch.argmax(porbs, dim=1).item()
    
end_time = time.time()
infer_time = (end_time - start_time) * 1000
print(f"预测结果：{cfg.id2label[pred_label]}\n推理时间：{infer_time:.2f}ms\n")
