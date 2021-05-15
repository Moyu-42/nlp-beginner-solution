import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as Data
from torch.nn.init import xavier_uniform_
import torch

def build_vocab(data):
    word2id = {}
    id2word = {}

    vocab = set()
    for row in data:
        vocab.update(row.lower().split(' '))
    wlist = ['<unk>', '<pad>'] + list(vocab)
    
    word2id = {word: i for i, word in enumerate(wlist)}
    id2word = {i: word for i, word in enumerate(wlist)}

    return word2id, id2word

def get_id(data, word2id, id2word):
    ids = []
    for row in data:
        id = list(map(lambda x: word2id.get(x, word2id['<unk>']), row.lower().split(' ')))
        ids.append(torch.LongTensor(id))
    pad_ids = pad_sequence(ids, batch_first=True)
    return torch.LongTensor(pad_ids)

def word2vec(vocab, dim):
    pass

def split(X, Y):
    idx = [i for i in range(0, X.shape[0])]
    np.random.shuffle(idx)
    train_len = int(0.8 * len(idx))
    test_len, dev_len = int(0.1 * len(idx)), int(0.1 * len(idx))
    return X[idx[:train_len]], Y[idx[:train_len]], \
           X[idx[train_len: train_len + test_len]], Y[idx[train_len: train_len + test_len]], \
           X[idx[train_len + test_len:]], Y[idx[train_len + test_len:]]

def make_data(X, Y):
    word2id, id2word = build_vocab(X)
    X = get_id(X, word2id, id2word)
    print(X.shape)
    vec = word2vec(word2id, X.shape[1])
    return split(X, Y), vec

class MyDataSet(Data.Dataset):
    def __init__(self, X, Y):
        super(MyDataSet, self).__init__()
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
