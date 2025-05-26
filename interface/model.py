from transformers import AutoModel
import torch.nn as nn
from torchvision import models

class RobertaClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout):
        super(RobertaClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_classes), labels.view(-1))

        return {
            'logits': logits,
            'loss': loss,
            'attentions': outputs.attentions  # 确保返回注意力
        }
        
def build_model():
    model = models.mobilenet_v3_small(pretrained=False)
    
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 3)
    
    return model
