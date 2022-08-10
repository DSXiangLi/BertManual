# -*-coding:utf-8 -*-
import torch
from torch import nn
from transformers import BertModel
from src.loss import seqlabel_loss_wrapper


class BertSoftmax(nn.Module):
    def __init__(self, tp):
        super(BertSoftmax, self).__init__()
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size
        self.bert = BertModel.from_pretrained(tp.pretrain_model)
        self.dropout_layer = nn.Dropout(tp.dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, tp.label_size)

    def forward(self, features):
        """
        features: {input_ids, token_type_ids, attention_mask, label_ids}
        """
        outputs = self.bert(input_ids=features['input_ids'],
                            token_type_ids=features['token_type_ids'],
                            attention_mask=features['attention_mask'])
        sequence_output = outputs[0]
        sequence_output = self.dropout_layer(sequence_output)

        logits = self.classifier(sequence_output)
        output = (logits,)

        if features.get('label_ids') is not None:
            loss = seqlabel_loss_wrapper(logits, features['label_ids'], features['attention_mask'], self.loss_fn)
            output += (loss,)
        return output
