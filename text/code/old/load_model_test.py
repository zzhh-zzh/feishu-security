from transformers import BertTokenizer, BertForSequenceClassification
# BertTokenizer：text -> token
# BertForSequenceClassification: 预训练好的中文 BERT 模型,add a classify layer
model_name = "bert-base-chinese"

tokenizer = BertTokenizer.from_pretrained(model_name)
# 分词器:中文句子拆成字词，转换为模型要求的ID，自动处理特殊token，自动填充截断长度
model = BertForSequenceClassification.from_pretrained(model_name, num_labels = 3)
# 加载预训练模型，有3个标签