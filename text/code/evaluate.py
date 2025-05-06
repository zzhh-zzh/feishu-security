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
from keyword_extractor import KeywordExtractor

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
keyword_extractor = KeywordExtractor(tokenizer, model) 
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

with torch.no_grad():
    loop = tqdm(dataloader, desc="evaluating", leave=True)
    for i, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        texts = [val_data.texts[i * cfg.batch_size + j] for j in range(len(input_ids))]

        start_time = time.time()
        
        outputs = model.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=True
        )
        logits = model.classifier(model.dropout(outputs.last_hidden_state[:, 0]))
        preds = torch.argmax(logits, dim=1)
        
        batch_keywords = keyword_extractor.extract_from_batch(
            texts=texts,
            input_ids=input_ids,
            attention_masks=attention_mask
        )
        top_tokens_batch.extend(batch_keywords)

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
