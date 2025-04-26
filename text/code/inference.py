import pandas as pd
import torch
import csv
import jieba
import pickle
from sklearn.metrics import accuracy_score, classification_report
from model import TextCNN
from config import cfg

model = TextCNN()
model.load_state_dict(torch.load(cfg.model_save_path + 'best_model.pth'))
model.to(cfg.device)
model.eval()

with open(cfg.word2idx_path, 'rb') as f:
    word2idx = pickle.load(f)
    
id2label = cfg.id2label

def preprocess(text, max_len):
    tokens = list(jieba.cut(text))
    idxs = [word2idx.get(tok, word2idx.get('<UNK>', 0)) for tok in tokens]
    if len(idxs) < max_len:
        idxs += [word2idx.get('<PAD>', 0)] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return torch.tensor(idxs).unsqueeze(0)

df = pd.read_csv(cfg.val_path, dtype=str, keep_default_na=False)
res = {'text': [], 'true_label': [], 'predict_label': []}
pred_all = []
ture_all = []

with torch.no_grad():
    for _,item in df.iterrows():
        text = item['text']
        true_label = item['label']
        input_tensor = preprocess(text, cfg.max_seq_len).to(cfg.device)
        output = model(input_tensor)
        
        pred_label = torch.argmax(output, dim=1).item()
         # 文件写入
        res['text'].append(text)
        res['true_label'].append(true_label)
        res['predict_label'].append(id2label[pred_label])
        # 性能计算
        ture_all.append(true_label)
        pred_all.append(id2label[pred_label])

result_df = pd.DataFrame(res, dtype=str)
result_df.to_csv('text/text_predict.csv',index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
print("批量推理完成，结果已保存到 text_predict.csv")

acc = accuracy_score(ture_all, pred_all)
report = classification_report(ture_all, pred_all, target_names=cfg.label_map.keys())
print("准确率报告 Acc:", acc)
print("\n详细报告")
print(report)
with open('text/predict.txt', 'w', encoding='utf-8') as f:
    f.write(f"准确率 Accuracy:{acc:.4f}\n\n")
    f.write("分类报告:\n")
    f.write(report)



