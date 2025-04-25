from transformers import BertTokenizer
from utils import TextDataset

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

dataset = TextDataset("text/trainset/base_train/base_train.csv", tokenizer, max_length=128)

print(f"数据集总共包括{len(dataset)}条样本")

sample = dataset[0]
print("\n第一个样本的内容：")
for key,val in sample.items():
    print(f"{key}:{val.shape if hasattr(val, 'shape')else val}")
    # 若是tensor则有shape，要不然就输出值 避免报错