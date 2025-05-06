import torch

class config:
    # 数据
    train_path = 'text/trainset/base_train/base_train.csv'
    val_path = 'text/trainset/base_val/base_val.csv'
    max_length = 512
    batch_size = 16
    label2id = {'消极': 0, '中性': 1, '积极': 2}
    id2label = {0 :'消极', 1 :'中性', 2 :'积极'}
    
    # 模型
    pretrained_name = 'uer/chinese_roberta_L-4_H-512'
    dropout = 0.1
    model_save = 'text/save_model/'
    num_labels = 3
    
    # 训练
    learning_rate = 2e-5
    epochs = 5
    weight_decay = 0.01
    logging_steps = 100
    device = 'cuda'
    seed = 42
    
cfg = config()