# -*-coding:utf-8 -*-
from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, pretrain_model, cls_dropout, label_size, freeze_bert):
        super(BertClassifier, self).__init__()
        self.label_size = label_size

        self.bert = BertModel.from_pretrained(pretrain_model)
        self.dropout_layer = nn.Dropout(cls_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_size)

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.dropout_layer(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return logits