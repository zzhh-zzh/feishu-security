import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from config import cfg
from data_loader import get_dataloader
from model import build_model

def train():
    device = cfg.device
    # 加载数据
    train_loader,_ = get_dataloader(cfg.train_dir, mode='train')
    # 加载模型
    model = build_model()
    # 损失函数 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg.epochs+1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc="training", leave=False)
        for images,labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        print(f"平均训练损失: {avg_train_loss:.4f}")

        if cfg.save_poch:
            epoch_path = cfg.model_save_path.replace(".pth", f"_epoch{epoch}.pth")
            torch.save(model.state_dict(), epoch_path)
            print(f"已保存模型：{epoch_path}")

        torch.save(model.state_dict(), cfg.model_save_path)
        print(f"\n训练完成，最终模型已保存到：{cfg.model_save_path}")

if __name__ == "__main__":
    train()
