import torch

class config:
    # 数据参数
    train_path = "text/trainset/base_train/base_train.csv"
    val_path = "text/trainset/base_val/base_val.csv"
    max_seq_len = 512
    vocab_size = 5000
    label_map = {'消极': 0, '中性': 1, '积极': 2}
    id2label = {0 :'消极', 1 :'中性', 2 :'积极'}
    word2idx_path = 'text/word2idx.pkl'
    
    # 模型参数
    embedding_dim = 200
    filter_sizes = [2, 3, 4]
    num_filters = 256
    dropout = 0.3
    num_classes = 3
    model_save_path = "text/save_model/"
    save_interval = 5
    
    # 训练参数
    batch_size = 64;
    learning_rate = 1e-4
    num_epochs = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 配置对象
cfg = config()