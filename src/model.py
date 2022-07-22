# -*-coding:utf-8 -*-
import torch
from torch import nn
from transformers import BertModel
from loss import seqlabel_loss_wrapper
from layers.crf import CRF


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """

    def __init__(self, pretrain_model, cls_dropout, label_size, loss_fn):
        super(BertClassifier, self).__init__()
        self.label_size = label_size
        self.loss_fn = loss_fn
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.dropout_layer = nn.Dropout(cls_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_size)

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = self.dropout_layer(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        output = (logits,)
        if label_ids:
            loss = self.loss_fn()
            output += (loss,)
        return output


class BertSoftmax(nn.Module):
    def __init__(self, pretrain_model, label_size, dropout, loss_fn):
        super(BertSoftmax, self).__init__()
        self.loss_fn = loss_fn
        self.label_size = label_size
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.dropout_layer = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_size)

    def forward(self, input_ids, token_type_ids, attention_mask, label_ids=None):
        outputs = self.bert(input_ids = input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout_layer(sequence_output)
        logits = self.classifier(sequence_output)
        output = (logits,)
        if label_ids is not None:
            loss = seqlabel_loss_wrapper(logits, label_ids, attention_mask, self.loss_fn)
            output += (loss,)
        return output

def pad_sequence(input_, pad_len=None, pad_value=0):
    """
    Pad List[List] sequence to same length
    """
    output = [] 
    for i in input_:
        output.append(i + [pad_value] * (pad_len-len(i)))
    return output    


class BertCrf(nn.Module):
    def __init__(self, pretrain_model, label_size, dropout):
        super(BertCrf, self).__init__()
        self.label_size = label_size
        self.bert = BertModel.from_pretrained(pretrain_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.label_size)
        self.crf = CRF(num_tags=self.label_size, batch_first=True)
        
    def forward(self, input_ids, token_type_ids, attention_mask, label_ids=None):
        outputs = self.bert(input_ids = input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        preds = self.crf.decode(emissions=logits, mask=attention_mask.bool())
        preds= pad_sequence(preds, pad_len=input_ids.size()[-1])
        outputs = (torch.tensor(preds, device=logits.device),)
        if label_ids is not None:
            log_likelihood = self.crf(emissions=logits, tags=label_ids, mask=attention_mask.bool(), reduction='mean')
            outputs += (-1*log_likelihood, ) 
        return outputs
