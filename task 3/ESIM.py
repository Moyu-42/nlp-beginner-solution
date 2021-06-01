import torch
import torch.nn as nn
import numpy as np

from torch.nn.modules.activation import Tanh

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_size, pretrained=None):
        super(EmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, _weight=pretrained)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x_embed = self.embedding(x)
        x_output = self.dropout(x_embed)
        return x_output

class InputEncodingLayer(nn.Module):
    # get \bar{a} and \bar{b}
    def __init__(self, input_size, hidden_size):
        super(InputEncodingLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)

    def forward(self, x):
        x_output, _ = self.lstm(x)
        return x_output

class LocalInferenceLayer(nn.Module):
    def __init__(self):
        super(LocalInferenceLayer, self).__init__()
    
    def forward(self, p, h):
        # print(p.shape, h.shape)
        e = torch.matmul(p, h.transpose(1, 2))
        p_w = nn.Softmax(dim=2)(e)
        h_w = nn.Softmax(dim=1)(e)
        # get \tilde{a} and \tilde{b}
        p_ = torch.matmul(p_w, h)
        h_ = torch.matmul(h_w.transpose(1, 2), p)
        # print(p_.shape, h_.shape)
        # get m_a and m_b
        m_p = torch.cat((p, p_, (p - p_), (p * p_)), dim=-1)
        m_h = torch.cat((h, h_, (h - h_), (h * h_)), dim=-1)
        # print(m_p.shape, m_h.shape)
        return m_p, m_h

class InferenceCompositionLayer(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(InferenceCompositionLayer, self).__init__()
        self.F = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=output_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
    
    def forward(self, x):
        # composition layer
        x_F = self.dropout(nn.ReLU()(self.F(x)))
        x_output, _ = self.lstm(x_F)
        # pooling
        # get average pooling
        v_ave = x_output.sum(1) / x_output.shape[-1]
        # print(v_ave.shape)
        # get max pooling
        v_max = torch.max(x_output, 1, keepdim=True).values.squeeze(1)
        # print(v_max.shape)
        v = torch.cat((v_ave, v_max), -1)
        return v

class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size, features):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(output_size, features),
            nn.Softmax(-1)
        )
    
    def forward(self, x):
        logits = self.mlp(x)
        return logits

class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, features, pretrained):
        super(ESIM, self).__init__()
        self.Embedding = EmbeddingLayer(vocab_size, embedding_size, pretrained)
        self.Encode = InputEncodingLayer(embedding_size, hidden_size)
        self.LocalInference = LocalInferenceLayer()
        self.InferenceComposition = InferenceCompositionLayer(hidden_size * 8, hidden_size, hidden_size)
        self.MLP = MLPLayer(hidden_size * 8, hidden_size, features)
    
    def forward(self, p, h):
        # print(p.shape, h.shape)
        p, h = self.Embedding(p), self.Embedding(h)
        # print(p.shape, h.shape)
        p, h = self.Encode(p), self.Encode(h)
        # print(p.shape, h.shape)
        m_p, m_h = self.LocalInference(p, h)
        # print(m_p.shape, m_h.shape)
        v_p, v_h = self.InferenceComposition(m_p), self.InferenceComposition(m_h)
        # print(v_p.shape, v_h.shape)
        v = torch.cat((v_p, v_h), -1)
        # print(v.shape)
        logits = self.MLP(v)
        return logits

if __name__ == '__main__':
    premise = torch.LongTensor(np.random.randint(0, 15, (10, 5)))
    hypothesis = torch.LongTensor(np.random.randint(0, 15, (10, 8)))
    label = np.random.randint(0, 3, (10, 1))
    model = ESIM(30, 15, 20, 3)
    print(model)
    print(model(premise, hypothesis))
