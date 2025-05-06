import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification,AdamW
from utils import TextDataset
from tqdm import tqdm

# 有g用g，没g用c
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
dataset = TextDataset("text/trainset/base_train/base_train.csv", tokenizer, max_length=128)
# 批量加载数据
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)
#加载模型
model.to(device)

# AdamW 推荐优化器 lr=2e-5经典学习率
optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 3;

model.train()
for epoch in range(epochs):
    print(f"\n---第{epoch+1}轮训练---")
    # 可视化训练进度 动态变化
    loop = tqdm(dataloader, desc="训练中")
    
    for batch in loop:
        # 每批数据放到gpu上
        batch = {key:val.to(device) for key,val in batch.items()}
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # 显示loss
        loop.set_postfix(loss=loss.item())
        model.save_pretrained(f"text/save_model/epoch{epoch+1}")
        tokenizer.save_pretrained(f"text/save_model/epoch{epoch+1}")


