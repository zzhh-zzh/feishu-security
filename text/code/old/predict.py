import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time

model = BertForSequenceClassification.from_pretrained("text/save_model/epoch3")
tokenizer = BertTokenizer.from_pretrained("text/save_model/epoch3")
model.eval()

# 有g用g，没g用c
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("当前设备:", device)

# 标签映射
id2label = {0:"积极", 1:"中性", 2:"消极"}

text = input("请输入一句话:")
# 这是cpu上初始化了一个tensor字典
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
# 遍历原来的input的键值对，然后调用到设备上，重新赋值
inputs = {k:v.to(device) for k,v in inputs.items()}

# 预测
# 开始计时
start_time = time.time()
# 不计算梯度，节省内存
with torch.no_grad():
    # 输入到模型
    outputs = model(**inputs)
    # 模型输出的原始分数，不是真正概率，最大值对应预测的分类
    logits = outputs.logits
    # 寻找到最大值对应的索引ID，item把张量换算成纯整数
    predicted = torch.argmax(logits, dim=1).item()
    
# 解释计时
end_time = time.time()
latency_ms = (end_time - start_time) * 1000
    
print(f"\n模型预测情感:{id2label[predicted]}")
print(f"推理时延:{latency_ms:.2f}毫秒")