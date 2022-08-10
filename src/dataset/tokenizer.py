# -*-coding:utf-8 -*-
import numpy as np


class GensimTokenizer(object):
    """
    Word Embedding Tokenizer Adapter
    """
    def __init__(self, w2v, addon_vocab=('[UNK]', '[PAD]','[SEP]'), keep_oov=True):
        self.w2v = w2v
        self.keep_oov = keep_oov
        self.vocab2idx = None
        self.idx2vocab = None
        self._embedding = None
        self.embedding_size = None
        self.vocab_size = None
        self.addon_vocab = addon_vocab
        self.init_vocab()

    @property
    def embedding(self):
        return self._embedding.astype(np.float32)

    def init_vocab(self):
        self.vocab2idx = dict([(word, idx) for idx, word in enumerate(self.w2v.wv.key_to_index)])
        self.idx2vocab = dict([(j, i) for i, j in self.vocab2idx.items()])

        self._embedding = np.array(self.w2v.wv.vectors)
        self.vocab_size = len(self.vocab2idx)
        self.embedding_size = self.embedding.shape[-1]
        for i in self.addon_vocab:
            self._add_vocab(i)

    def _add_vocab(self, vocab):
        self.vocab2idx.update({vocab: self.vocab_size})
        self.vocab_size += 1
        self._embedding = np.vstack((self._embedding,
                                     np.random.normal(0, 1, size=(1, self.embedding_size))))

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for i in tokens:
            if i in self.vocab2idx:
                ids.append(self.vocab2idx[i])
            elif self.keep_oov:
                ids.append(self.vocab2idx['[UNK]'])
            else:
                pass
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.idx2vocab[i])
        return tokens
