import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter
import random
from softmax_regression import SoftmaxRegression

class BoW():
    def __init__(self):
        self.tok2id = {}
    
    def tokenizer(self, data):
        return data.lower().split(' ')
    
    def get_data(self, datas):
        vocab = set()
        for row in datas:
            bag_of_words = self.tokenizer(row)
            vocab.update(bag_of_words)
        self.tok2id = dict([(k, v) for v, k in enumerate(vocab)])
        # transfer to csr_matrix
        indptr = [0]
        indices = []
        data = []
        for row in datas:
            token = self.tokenizer(row)
            ct = Counter(token)
            for k, val in ct.items():
                indices.append(self.tok2id[k])
                data.append(val)
            indptr.append(len(indices))
        return csr_matrix((data, indices, indptr), dtype=int, shape=(len(datas), len(self.tok2id)))


class NGram():
    def __init__(self, step):
        self.step = step
        self.tok2id = {}
    
    def tokenizer(self, data):
        return data.lower().split(' ')
    
    def get_ngrams(self, token):
        n_gram = []
        for i in range(0, len(token) - self.step + 1):
            n_gram.append(' '.join(token[i: i + self.step]))
        return n_gram
    
    def get_data(self, datas):
        vocab = set()
        for row in datas:
            token = self.tokenizer(row) 
            n_gram = self.get_ngrams(token)
            vocab.update(n_gram)
        self.tok2id = dict([(k, v) for v, k in enumerate(vocab)])
        # transfer to csr_matrix
        indptr = [0]
        indices = []
        data = []
        for row in datas:
            token = self.tokenizer(row) 
            n_gram = self.get_ngrams(token)
            ct = Counter(n_gram)
            for k, val in ct.items():
                indices.append(self.tok2id[k])
                data.append(val)
            indptr.append(len(indices))
        return csr_matrix((data, indices, indptr), dtype=int, shape=(len(datas), len(self.tok2id)))

def split(X, Y):
    idx = [i for i in range(0, X.shape[0])]
    random.shuffle(idx)
    train_len = int(0.8 * len(idx))
    test_len, dev_len = int(0.1 * len(idx)), int(0.1 * len(idx))
    return X[idx[:train_len]], Y[idx[:train_len]], \
           X[idx[train_len: train_len + test_len]], Y[idx[train_len: train_len + test_len]], \
           X[idx[train_len + test_len:]], Y[idx[train_len + test_len:]]

# TODO: mini-batch
def mini_batch(data):
    pass

if __name__ == '__main__':
    data = pd.read_csv('train.tsv', sep='\t')
    n = len(data)
    print("col of data is %d" % len(data))

    # Bag of Words method
    bow = BoW()
    X = bow.get_data(data['Phrase'])
    # N-gram method
    # ngram = NGram(3)
    # X = ngram.get_data(data['Phrase'])
    Y = data['Sentiment'].values

    features = X.shape[1]
    classes = len(np.unique(Y))
    print("Features = {}, Classes = {}".format(features, classes))
    train_X, train_Y, test_X, test_Y, dev_X, dev_Y = split(X, Y)

    model = SoftmaxRegression(features, classes, alpha=0.1)
    loss = []
    epochs = 50
    for epoch in range(1, epochs + 1):
        Loss = 0
        label = 0
        batch_size = 1024
        while label < train_X.shape[0]:
            end = min(label + batch_size, train_X.shape[0])
            Loss += model.fit(train_X[label: end], train_Y[label: end])
            label += batch_size
        loss.append(Loss)
        pred_Y = model.predict(test_X)
        print(pred_Y, test_Y)
        input()
        acc = (pred_Y == test_Y).sum()
        total = len(test_Y)
        print("Epoch [{}/{}]: Loss: {:.4f} Acc: {:.4f}".format(epoch, epochs, Loss, acc / total))
