import torch
import jieba
import numpy as np
from transformers import AutoTokenizer

class KeywordExtractor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        jieba.initialize()  # 初始化分词器
    
    def _align_words_and_tokens(self, text, offset_mapping):
        """使用jieba分词对齐文本和token"""
        # 获取有效字符位置 (排除特殊token)
        valid_offsets = []
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # 排除填充部分
                continue
            valid_offsets.append((i, start, end))
        
        # 使用jieba进行词语切分
        words = jieba.lcut(text)
        word_spans = []
        pos = 0
        
        # 获取每个词语的字符位置范围
        for word in words:
            start = text.find(word, pos)
            if start == -1:
                continue
            end = start + len(word)
            word_spans.append((start, end))
            pos = end
        
        # 对齐词语和token
        word_token_indices = []
        for word, (w_start, w_end) in zip(words, word_spans):
            tokens_in_word = []
            for i, t_start, t_end in valid_offsets:
                if t_start >= w_start and t_end <= w_end:
                    tokens_in_word.append(i)
            if tokens_in_word:
                word_token_indices.append((word, tokens_in_word))
        
        return word_token_indices
    
    def get_keywords(self, text, input_ids, attention_weights, offset_mapping, top_k=3):
        """获取完整词语的关键词"""
        # 转换参数类型
        input_ids = input_ids.cpu().numpy()
        attention_weights = attention_weights.cpu().numpy()
        offset_mapping = offset_mapping.cpu().numpy()
        
        # 对齐词语与token
        word_token_indices = self._align_words_and_tokens(text, offset_mapping)
        
        # 计算词语权重
        word_weights = []
        for word, token_indices in word_token_indices:
            if not token_indices:
                continue
            avg_weight = np.mean([attention_weights[i] for i in token_indices])
            word_weights.append((word, avg_weight))
        
        # 排序并选取top_k
        word_weights.sort(key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in word_weights[:top_k]]
        
        # 添加上下文
        keywords = []
        for word in top_words:
            start = text.find(word)
            if start == -1:
                keywords.append(word)
                continue
            end = start + len(word)
            context = text[max(0, start-5):min(len(text), end+5)]
            keywords.append(f"{word}[{context}]")
        
        return " | ".join(keywords)
    
    def extract_from_batch(self, texts, input_ids, attention_masks, offset_mappings, top_k=3):
        """批量提取关键词"""
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_masks,
                output_attentions=True
            )
            
            # 获取最后一层CLS的注意力权重
            attentions = outputs.attentions[-1]  # (batch, heads, seq_len, seq_len)
            cls_attention = attentions[:, :, 0].mean(dim=1)  # 平均所有头
            
            batch_keywords = []
            for i in range(len(input_ids)):
                keywords = self.get_keywords(
                    text=texts[i],
                    input_ids=input_ids[i],
                    attention_weights=cls_attention[i],
                    offset_mapping=offset_mappings[i],
                    top_k=top_k
                )
                batch_keywords.append(keywords)
            
            return batch_keywords
