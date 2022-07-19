# -*-coding:utf-8 -*-
import torch
from torch.utils.data.dataset import Dataset


class SeqDataset(Dataset):
    def __init__(self, file_name, max_len, tokenizer, data_loader):
        self.raw_data = data_loader(file_name)
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.features = []
        self.labels = []
        self.build_feature()

    def build_feature(self):
        for data in self.raw_data:
            self.features.append(self.tokenizer.encode_plus(data['text1'], padding='max_length',
                                                            truncation=True, max_length=self.max_len))
        if self.raw_data[0].get('label'):
            self.labels = [i['label'] for i in self.raw_data]

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = {k: torch.tensor(v) for k, v in sample.items()}
        if self.labels:
            sample['label'] = self.labels[idx]
        return sample

    def __len__(self):
        return len(self.raw_data)