from data_loader import get_dataloader
import torch.nn as nn
import torch
from config import cfg
from model import RobertaClassifier
import os
from tqdm import tqdm
from torch.optim import AdamW

# 运行c/g
device = cfg.device

# 数据集
dataloader, tokenizer, _ = get_dataloader(
    cfg.train_path, 
    batch_size=cfg.batch_size,
    max_length=cfg.max_length,
    shuffle=True)

# 模型
model_name = cfg.pretrained_name
model = RobertaClassifier(
    model_name=cfg.pretrained_name,
    num_classes=cfg.num_labels,
    dropout=cfg.dropout
)
model.to(device)

# 优化器
optimizer = AdamW(
    model.parameters(), 
    lr=cfg.learning_rate,
    weight_decay= cfg.weight_decay
    )

epochs = cfg.epochs
save_dir = cfg.model_save

model.train()
for epoch in range(0,epochs):
    print(f'\n--第{epoch+1}轮训练---')
    loop = tqdm(dataloader, desc=f'Epoch{epoch+1}/{epochs}')
    
    for batch in loop:
        batch = {key:val.to(device) for key,val in batch.items()}
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        loss, logits = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )
    
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
         
        loop.set_postfix(loss=loss.item())
    
        
    epoch_save_path = os.path.join(save_dir,f'epoch{epoch+1}')
    os.makedirs(epoch_save_path, exist_ok=True)
    torch.save(model.state_dict(), epoch_save_path +"/model_state_dict.pth")
    tokenizer.save_pretrained(epoch_save_path)
    print(f"第{epoch+1}轮训练完成，模型已保存至 {epoch_save_path}")
        
