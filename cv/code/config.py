
class config:
    # 路径
    train_dir = 'cv/trainset/base_train/train'
    val_dir = 'cv/trainset/base_val/val'
    model_save_path = 'cv/model_save/mobilenet_v3_small.pth'
    log_dir = "cv/logs"
    
    # 图像处理
    image_size = 256
    num_classes = 3
    
    # 模型
    model_name = "mobilenet_v3_small"
    pretrained = True
    
    # 超参数
    batch_size = 16
    num_workers = 4
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-5
    early_stop_patience = 5
    num_error_samples = 5
    
    # 设备
    device = 'cuda'
    
    save_poch = True
    use_gradcam = True
    use_half_predict = False
    label2id = {'猫':0, '狗':1, '其他动物':2}
    id2label = {0:'猫', 1:'狗', 2:'其他动物'}
    
    
cfg = config()
    