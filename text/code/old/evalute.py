import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from utils import TextDataset
from torch.utils.data import DataLoader

# 加载模型和tokenizer
model = BertForSequenceClassification.from_pretrained("text/save_model/epoch3")
tokenizer = BertTokenizer.from_pretrained("text/save_model/epoch3")
model.eval()

# 有g用g，没g用c
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 映射
label2id = {"积极":0, "中性":1, "消极":2}
id2label = {v:k for k,v in label2id.items()}

# 读取验证集
val_df = pd.read_csv("text\\trainset\\base_val\\base_val.csv")

# 构造验证集
class ValDataset(TextDataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df["text"].tolist()
        self.labels = [label2id[label] for label in df["label"]]
        self.tokenizer = tokenizer
        self.max_length = max_length
     
val_dataset = ValDataset(val_df, tokenizer)
# 批量处理验证集
val_loader = DataLoader(val_dataset, batch_size=16)

# 预测值和真实值
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        # 模型输出的原始分数，不是真正概率，最大值对应预测的分类
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
print("准确率 Accuracy:", accuracy_score(all_labels, all_preds))
print("\n详细分类报告")
print(classification_report(all_labels, all_preds, target_names=label2id.keys()))

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

report_path = f"text/eval_report_{timestamp}.txt"
csv_path = f"text/predictions_{timestamp}.csv"

with open(report_path, "w", encoding="utf-8") as f:
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=label2id.keys())
    
    f.write(f"准确率 Accuracy:{acc:4f}\n\n")
    f.write("分类报告:\n")
    f.write(report)

print(f"分类报告已保存到:{report_path}")

output_df = pd.DataFrame({
    "text":val_df["text"],
    "真实标签":[id2label[i] for i in all_labels],
    "预测标签":[id2label[i] for i in all_preds],
    "是否预测正确":[id2label[i]==id2label[j] for i,j in zip(all_labels, all_preds)]
})

output_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"预测详情已经保存到:{csv_path}")