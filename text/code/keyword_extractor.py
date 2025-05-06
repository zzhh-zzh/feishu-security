import torch
from transformers import AutoTokenizer
import jieba

class KeywordExtractor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    
    @staticmethod
    def merge_subwords(tokens):
        """合并子词为完整词语"""
        merged = []
        current_word = ""
        for token in tokens:
            # 处理中文子词（根据实际tokenizer调整）
            if token.startswith("##"):
                current_word += token[2:]
            elif token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            else:
                if current_word:
                    merged.append(current_word)
                current_word = token
        if current_word:
            merged.append(current_word)
        return merged
    
    def get_keywords_with_context(self, text, input_ids, attention_weights, top_k=3):
        """获取完整的关键词及其上下文"""
        # 获取原始token和对应的注意力权重
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        valid_indices = [i for i, token in enumerate(tokens) 
                        if token not in [self.tokenizer.cls_token, 
                                       self.tokenizer.sep_token, 
                                       self.tokenizer.pad_token]]
        
        if not valid_indices:
            return ""
        
        tokens = [tokens[i] for i in valid_indices]
        weights = [attention_weights[i].item() for i in valid_indices]
        
        # 合并子词为完整词语
        merged_words = self.merge_subwords(tokens)
        
        # 计算词语级别的权重
        word_weights = []
        word_start = 0
        for word in merged_words:
            # 计算词语对应的token数量
            token_count = 0
            current_len = 0
            while word_start + token_count < len(tokens) and current_len < len(word):
                current_len += len(tokens[word_start + token_count].replace("##", ""))
                token_count += 1
            
            # 计算词语的平均权重
            word_weight = sum(weights[word_start:word_start+token_count]) / token_count
            word_weights.append((word, word_weight))
            word_start += token_count
        
        # 按权重排序并选择top_k
        word_weights.sort(key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in word_weights[:top_k]]
        
        # 获取关键词上下文
        keywords_with_context = []
        
        for word in top_words:
            # 在原文中找到关键词位置
            start_pos = text.find(word)
            if start_pos == -1:
                keywords_with_context.append(word)
                continue
            
            end_pos = start_pos + len(word)
            # 获取前后各5个字符作为上下文
            context_start = max(0, start_pos - 5)
            context_end = min(len(text), end_pos + 5)
            context = text[context_start:context_end]
            keywords_with_context.append(f"{word}[{context}]")
        
        return " | ".join(keywords_with_context)
    
    def extract_from_batch(self, texts, input_ids, attention_masks, top_k=3):
        """批量提取关键词"""
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_masks,
                output_attentions=True
            )
            
            # 获取最后一层的平均注意力权重
            attentions = outputs.attentions[-1]
            cls_attention = attentions.mean(dim=1)[:, 0, :]  # (batch_size, seq_len)
            
            batch_keywords = []
            for i in range(len(input_ids)):
                keywords = self.get_keywords_with_context(
                    text=texts[i],
                    input_ids=input_ids[i],
                    attention_weights=cls_attention[i],
                    top_k=top_k
                )
                batch_keywords.append(keywords)
            
            return batch_keywords
