import torch
import numpy as np
import cv2
from sklearn.metrics import accuracy_score,precision_recall_fscore_support
from tqdm import tqdm
from config import cfg
from data_loader import get_dataloader
from model import build_model
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

def generate_gradcam(model, images, target_class):
    # 最后一层卷积层
    model.eval()
    
    feature_maps = None
    gradients = None
    
    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output
        
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]
        
    target_layer = model.features[-1]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    output = model(images)
    class_score = output[:, target_class].squeeze()
    
    model.zero_grad()
    class_score.backward()
    
    forward_handle.remove()
    backward_handle.remove()

    
    # 获取卷积层特征图
    pooled_grads = torch.mean(gradients, dim=[0, 2, 3])
    
    # 计算加权特征图
    cam = torch.zeros(feature_maps.shape[2:], dtype=torch.float32).to(images.device)
    
    for i in range(feature_maps.shape[1]):
        cam += pooled_grads[i] * feature_maps[0, i, :, :]
        
    # 归一化
    cam = torch.nn.functional.relu(cam)
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()
    return cam

def eval():
    # device = 'cpu'
    device = 'cuda'
    # 加载模型
    model = build_model()
    model.load_state_dict(torch.load(cfg.model_save_path))
    model.to(device)
    # 加载验证集
    val_loader,_ = get_dataloader(cfg.val_dir, mode='val')
    dataset = val_loader.dataset
    paths = [path for path, _ in dataset.samples]
    # 推理评估
    all_preds = []
    all_labels = []
    error_samples = []
    val_time = []
    
    
    model.eval()
    loop = tqdm(val_loader, desc="evaluating", leave=False)
    
       
    for i,(images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)
        st_time = time.time()
        with torch.no_grad():
            outputs = model(images)
            _,preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        

        
        # 错误样本分析
        incorrect_indices = (preds != labels).nonzero(as_tuple = False).squeeze(0)
        
        batch_start = i * cfg.batch_size
        for idx in incorrect_indices:
            global_idx = batch_start + idx.item()
            error_samples.append({
                'image_path':paths[global_idx],
                'true_label':labels[idx].item(),
                'pred_label':preds[idx].item()
            })
            
        # 可视化
        for idx in range(images.size(0)):
        # if len(preds) < cfg.num_error_samples:
            img = images[idx].unsqueeze(0)
            label = labels[idx].item()
            pred = preds[idx].item()
            
            cam = generate_gradcam(model, img, target_class=pred)
            cam = cam.detach().cpu().numpy()
            cam = np.uint8(255 * cam)
            cam = cv2.resize(cam, (images.shape[3], images.shape[2]))  # Resize to match original image size
            cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # Now shape: (H, W, 3)

            # 原始图像处理
            img_np = img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
            img_np = np.uint8(255 * img_np)  # 如果是 [0,1] 范围的图像
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR

            # 保证两者尺寸完全一致
            if cam.shape != img_np.shape:
                cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

            # 叠加
            overlay = cv2.addWeighted(cam, 0.8, img_np, 0.2, 0)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)  # 再转回 RGB 供 matplotlib 显示

            # 显示图像
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))  # 再转为 RGB 显示
            plt.title(f"True: {label}, Pred: {pred}")
            plt.subplot(1, 2, 2)
            plt.imshow(overlay)
            plt.title("Grad-CAM")
            plt.tight_layout()
            plt.savefig(f"cv/cam/gradcam_output_{batch_start+idx}_{cfg.id2label[label]}_{cfg.id2label[pred]}.png") 
            plt.close()
            # 防止重复覆盖
            # plt.show()
            
        end_time = time.time()
        ave_time = end_time - st_time
        val_time.append(ave_time)
        ave_time = ave_time * 1000 / cfg.batch_size
        
        loop.set_postfix(ave = f"{ave_time:.4f}ms")
    
    ave_total = 1000 * sum(val_time) / (len(val_time) * cfg.batch_size)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average='weighted')
    print(f"平均耗时：{ave_total:.4f}ms")
    print(f"\n评估指标：")    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    with open(f'cv/report_{timestamp}.txt', 'w') as f:
        f.write(f"平均时间: {ave_total:.4f}ms")
        f.write(f"准确率 Accuracy: {accuracy:4f}\n\n")
        f.write(f"回报率 Recall: {recall:.4f}\n\n")
        f.write(f"F1分数 F1-Score: {f1:.4f}\n\n")
    
    print(f"\n错误样本统计：")
    for sample in error_samples:
        print(f"Path:{sample['image_path']}, 真实标签: {sample['true_label']}, 预测标签: {sample['pred_label']}")
    
    with open(f'cv/error_samples_{timestamp}.txt', 'w') as f:
        for sample in error_samples:
            f.write(f"path:{sample['image_path']},True: {sample['true_label']}, Pred: {sample['pred_label']}\n")
        
if __name__ == "__main__":
    eval()