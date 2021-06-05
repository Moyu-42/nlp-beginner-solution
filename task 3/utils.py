import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.init import xavier_uniform_
import torch.utils.data as Data
import torch

y_dict = {'entailment':0, 'contradiction':1, 'neutral':2, '-': np.random.randint(0, 3)}

class MyDataSet(Data.Dataset):
    def __init__(self, premise, hypothesis, label):
        super(MyDataSet, self).__init__()
        self.p = premise
        self.h = hypothesis
        self.l = label
    
    def __len__(self):
        return self.p.shape[0]
    
    def __getitem__(self, index):
        return self.p[index], self.h[index], self.l[index]

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

def get_id(data, word2id):
    ids = []
    for row in data:
        id = list(map(lambda x: word2id.get(x, word2id['<unk>']), row.lower().split(' ')))
        ids.append(torch.LongTensor(id))
    pad_ids = pad_sequence(ids, batch_first=True)
    return torch.LongTensor(pad_ids)

def word2vec(vocab, dim):
    word_vec = xavier_uniform_(torch.empty(len(vocab), dim))
    num_of_word = 0
    with open("../data/glove.6B.300d.txt", encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.split(' ', 1)
            word = line[0]
            vec = line[1].strip('\n').split(' ')
            if word in vocab:
                num_of_word += 1
                word_vec[vocab[word]] = torch.tensor(list(map(lambda x: float(x), vec)))
            if num_of_word == len(vocab) - 1:
                break
    return word_vec

def get_DataSet(data_path, word2id, verbose=False):
    data = pd.read_csv(data_path, sep='\t').dropna(axis=0, how='any')
    premise = data['sentence1'].values
    hypothesis = data['sentence2'].values
    for i, row in enumerate(hypothesis):
        if type(row) != str:
            print(i, premise[i], hypothesis[i])
            input()
    vocab = premise + hypothesis
    if verbose:
        word2id, id2word = build_vocab(vocab)
        w2vec = word2vec(word2id, 300)
    premise = get_id(data['sentence1'].values, word2id)
    hypothesis = get_id(data['sentence2'].values, word2id)
    label = [y_dict[lb] for lb in data['gold_label'].values]
    return MyDataSet(premise, hypothesis, label), word2id, w2vec if verbose else torch.empty(len(word2id), 200)

if __name__ == '__main__':
    test = get_DataSet("../data/snli_1.0/snli_1.0_test.txt", dict, True)