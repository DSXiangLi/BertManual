# -*-coding:utf-8 -*-
import torch
from torch import nn


class Textcnn(nn.Module):
    def __init__(self, tp, embedding=None):
        super(Textcnn, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float32))
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=tp.embedding_dim,
                      out_channels=tp.filter_size,
                      kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(tp.max_seq_len - kernel_size * 2 + 2))
        )
            for kernel_size in tp.kernel_size_list])
        self.dropout = nn.Dropout(tp.dropout_rate)
        self.fc = nn.Linear(int(tp.filter_size * len(tp.kernel_size_list)), tp.label_size)

    def forward(self, features):
        x = self.embedding(features['token_ids'])  # (batch_size, seq_len, emb_dim)

        x = [conv(x.permute(0, 2, 1)).squeeze(-1) for conv in self.convs]  # input (batch_size, channel, # seq_len)
        x = torch.cat(x, dim=1)  # (batch_size, sum(filter_size))
        x = self.dropout(x)

        logits = self.fc(x)
        output = (logits,)

        if features.get('label_ids', None) is not None:
            loss = self.loss_fn(logits, features['label_ids'])
            output += (loss,)
        return output


class Textcnn2(nn.Module):
    """
    2层CNN + 2层FC
    """

    def __init__(self, tp, embedding=None):
        super(Textcnn2, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding, dtype=torch.float32))
        self.loss_fn = tp.loss_fn
        self.label_size = tp.label_size

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=tp.embedding_dim,
                      out_channels=tp.filter_size,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(tp.filter_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=tp.filter_size,
                      out_channels=tp.filter_size,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(tp.filter_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(tp.max_seq_len - tp.kernel_size * 2 + 2))
        )
            for kernel_size in tp.kernel_size_list])

        self.dropout = nn.Dropout(tp.dropout_rate)

        self.fc = nn.Sequential(
            nn.Linear(int(tp.filter_size * len(tp.kernel_size_list)), tp.hidden_size),
            nn.BatchNorm1d(tp.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(tp.hidden_size, tp.label_size)
        )

    def forward(self, token_ids, label_ids=None):
        emb = self.embedding(token_ids)  # (batch_size, seq_len, emb_dim)
        emb = [conv(emb.permute(0, 2, 1)).squeeze(-1) for conv in
               self.convs]  # Conv1d input shape (batch_size, channel, seq_len)
        x = torch.cat(emb, dim=1)  # (batch_size, sum(filter_size))
        x = self.dropout(x)
        logits = self.fc(x)
        output = (logits,)
        if label_ids is not None:
            loss = self.loss_fn(logits, label_ids)
            output += (loss,)
        return output
