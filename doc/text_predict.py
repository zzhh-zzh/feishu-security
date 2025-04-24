import pandas as pd
import torch
import csv

# load your model
model = Model().cuda()
id2label = {0: '积极', 1: '消极', 2: '中性'}

df = pd.read_csv('base_val.csv', dtype=str, keep_default_na=False)
res = {'text': [], 'label': []}
for _, item in df.iterrows():
    text = df['text']
    scores = model(text)
    score, label = torch.max(scores, dim=-1)
    res['text'].append(df['text'])
    res['label'].append(id2label[label])
    
text_df = pd.DataFrame(data=res, dtype=str)
text_df.to_csv('text_predict.csv', index=False, quoting=csv.QUOTE_ALL)