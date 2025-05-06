from data_loader import get_dataloader
from sklearn.metrics import accuracy_score, classification_report
from config import cfg
import pandas as pd
import os
import torch
from model import RobertaClassifier
from datetime import datetime
from transformers import AutoTokenizer
import time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

# 设置模型路径和模型加载
model_path = os.path.join(cfg.model_save, 'best_model/')
model = RobertaClassifier(
    model_name=cfg.pretrained_name,
    num_classes=cfg.num_labels,
    dropout=cfg.dropout
)
model.load_state_dict(torch.load(os.path.join(model_path, 'model_state_dict.pth')))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# 读取验证集数据
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataloader, _, val_data = get_dataloader(
    cfg.val_path, 
    tokenizer, 
    batch_size=cfg.batch_size,
    max_length=cfg.max_length,
    shuffle=False)

# 存储预测结果和标签
all_preds = []
all_labels = []
all_texts = []
time_val = []
top_tokens_batch = []

def get_keywords(input_ids, attention_mask, tokenizer, top_k=3):
    """获取影响预测的关键词"""
    with torch.no_grad():
        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
    
    # 获取最后一层的平均注意力权重
    attentions = outputs.attentions[-1]  # (batch_size, num_heads, seq_len, seq_len)
    avg_attentions = attentions.mean(dim=1)  # 平均所有注意力头 (batch_size, seq_len, seq_len)
    
    # 获取CLS标记对其他标记的注意力权重
    cls_attention = avg_attentions[:, 0, :]  # (batch_size, seq_len)
    
    keywords_list = []
    for i in range(len(input_ids)):
        # 获取非填充部分的token和注意力权重
        non_padding = attention_mask[i].nonzero().squeeze()
        tokens = input_ids[i][non_padding]
        weights = cls_attention[i][non_padding]
        
        # 排除特殊token
        special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]
        valid_indices = [idx for idx, token in enumerate(tokens) if token not in special_tokens]
        
        if len(valid_indices) == 0:
            keywords_list.append([])
            continue
            
        tokens = tokens[valid_indices]
        weights = weights[valid_indices]
        
        # 获取权重最高的top_k个token
        top_indices = weights.topk(min(top_k, len(weights))).indices
        top_tokens = [tokenizer.decode([token]) for token in tokens[top_indices]]
        keywords_list.append(", ".join(top_tokens))
    
    return keywords_list

with torch.no_grad():
    loop = tqdm(dataloader, desc="evaluating", leave=True)
    for i, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        start_time = time.time()
        loss, logits = model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=labels
        )
        
        preds = torch.argmax(logits, dim=1)
        
        keywords = get_keywords(input_ids, attention_mask, tokenizer)
        top_tokens_batch.extend(keywords)

        end_time = time.time()            
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        time_val.append(end_time - start_time)
        # 每批次耗时显示
        ave_batch = 1000 * (end_time - start_time) / cfg.batch_size
        loop.set_postfix(ave=f"{ave_batch:.4f}ms")

# 计算准确率和时间
average_inference_time = 1000 * sum(time_val) / (len(time_val) * cfg.batch_size)
acc = accuracy_score(all_labels, all_preds)
print(f"平均推理时间: {average_inference_time:.4f} ms/piece")
print(f"准确率 Accuracy:{acc:.4f}")
print("\n详细分类报告:")
print(classification_report(all_labels, all_preds, target_names=cfg.label2id.keys()))

# 保存报告
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"text/eval_report_{timestamp}.txt"
csv_path = f"text/predictions_{timestamp}.csv"

with open(report_path, "w", encoding="utf-8") as f:
    report = classification_report(all_labels, all_preds, target_names=cfg.label2id.keys())
    f.write(f"准确率 Accuracy: {acc:.4f}\n\n")
    f.write("分类报告:\n")
    f.write(report)

print(f"分类报告已保存到: {report_path}")

# 保存带关键词的预测详情
output_df = pd.DataFrame({
    "text": val_data.texts,
    "真实标签": [cfg.id2label[i] for i in all_labels],
    "预测标签": [cfg.id2label[i] for i in all_preds],
    "是否预测正确": [cfg.id2label[i] == cfg.id2label[j] for i, j in zip(all_labels, all_preds)],
    "关键词": top_tokens_batch
})
output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"预测详情已经保存到: {csv_path}")
