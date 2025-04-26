import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from config import cfg
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        """
        模型送入GPU，adam优化器进行参数更新
        """
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(cfg.device)
                outputs = self.model(inputs)
                pred = torch.argmax(outputs, dim=1).cpu().tolist()
                preds.extend(pred)
                trues.extend(labels.tolist())
        return f1_score(trues, preds, average='macro')

    def run(self):
        best_f1 = 0
        os.makedirs(cfg.model_save_path, exist_ok=True)
        for epoch in range(cfg.num_epochs):
            train_loss = self.train_one_epoch()
            val_f1 = self.evaluate()
            print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.model.state_dict(), os.path.join(cfg.model_save_path, 'best_model.pth'))
        print(f"Training Complete. Best F1: {best_f1:.4f}")
