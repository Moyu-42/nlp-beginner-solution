import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.nn.functional as F

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, classes, pretrained=None, type_='RNN', bidirectional='False', num_of_layers=3, dropout=0.3):
        super(TextRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.classes = classes
        self.pretrained = pretrained
        self.type_ = type_
        self.bidirectional = bidirectional
        self.num_of_layers = num_of_layers

        if pretrained != None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, _weight=pretrained)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        
        if self.type_ == 'RNN':
            self.rnn = nn.RNN(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_of_layers, batch_first=True, bidirectional=self.bidirectional, dropout=0.3)
        elif self.type_ == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_of_layers, batch_first=True, bidirectional=self.bidirectional, dropout=0.3)
        self.linear = nn.Linear(self.hidden_size, self.classes)

    def forward(self, x):
        batch_size, seq = x.shape
        embedding = self.embedding(x)
        if self.bidirectional:
            h0 = torch.randn(self.num_of_layers * 2, batch_size, self.hidden_size).cuda()
            c0 = torch.randn(self.num_of_layers * 2, batch_size, self.hidden_size).cuda()
        else:
            h0 = torch.randn(self.num_of_layers * 1, batch_size, self.hidden_size).cuda()
            c0 = torch.randn(self.num_of_layers * 1, batch_size, self.hidden_size).cuda()
        if self.type_ == 'RNN':
            _, hn = self.rnn(embedding, h0)
        elif self.type_ == 'LSTM':
            _, (hn, _) = self.rnn(embedding, (h0, c0))
        if self.bidirectional:
            hn = hn[0]
        y = self.linear(hn).squeeze(0)
        return y

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, classes, pretrained=None, kernel_num=100, kernel_size=[3, 4, 5], dropout=0.5):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.pretrained = pretrained
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.dropout = dropout

        if self.pretrained != None:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, _weight=pretrained)
        else:
            self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.kernel_num, (kernel_size_, embedding_size)) for kernel_size_ in self.kernel_size])
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(3 * self.kernel_num, self.classes)

    def forward(self, x):
        embedding = self.embedding(x).unsqueeze(1)
        convs = [nn.ReLU()(conv(embedding)).squeeze(3) for conv in self.convs]
        pool_out = [nn.MaxPool1d(block.size(2))(block).squeeze(2) for block in convs]
        pool_out = torch.cat(pool_out, 1)

        logits = self.linear(pool_out)

        return logits

if __name__ == '__main__':
    x = torch.LongTensor(np.random.randint(0, 20, (10, 8))).cuda()
    y = torch.tensor(np.random.randint(0, 2, (10, 1))).cuda()
    print(x)
    print(y)
    model = TextCNN(20, 6, 2).cuda()
    print(model)
    pred = model(x)
    print(pred.shape)