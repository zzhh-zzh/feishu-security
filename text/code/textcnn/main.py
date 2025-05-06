from config import cfg
from model import TextCNN
from trainer import Trainer
from data_loader import load_dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    train_dataset, val_dataset, word2idx = load_dataset()
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)
    
    model = TextCNN()
    trainer = Trainer(model, train_loader, val_loader)
    trainer.run()
