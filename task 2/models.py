import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import dropout

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, classes, pretrained=None):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.classes = classes
        self.pretrained = pretrained

        if pretrained != None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, _weight=pretrained)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.rnn = nn.RNN(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.classes)

    def forward(self, x):
        batch_size, seq = x.shape
        embedding = self.embedding(x)
        h0 = torch.randn(1, batch_size, self.hidden_size).cuda()
        _, hn = self.rnn(embedding, h0)
        y = self.linear(hn).squeeze(0)
        return y

class CNN(nn.Module):
    pass

if __name__ == '__main__':
    x = torch.LongTensor(np.random.randint(0, 20, (10, 4))).cuda()
    y = np.random.randint(0, 2, (10, 1)).cuda()
    print(x)
    print(y)
    model = RNN(20, 4, x.shape[1], 2)
    pred = model(x)
    print(pred)