import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import utils
from models import TextRNN, TextCNN
from tqdm import tqdm, trange

if __name__ == '__main__':
    data = pd.read_csv('../data/train.tsv', sep='\t')
    # TODO: Word Embedding
    X = data['Phrase'].values
    Y = data['Sentiment'].values
    classes = len(np.unique(Y))
    X, word2vec, word2id, id2word = utils.make_data(X)
    print(word2vec.shape)
    train_X, train_Y, test_X, test_Y, dev_X, dev_Y = utils.split(X, Y)
    # TODO: split and mini-batch use DataLoader
    train_loader = Data.DataLoader(utils.MyDataSet(train_X, train_Y), 64, True)
    test_loader = Data.DataLoader(utils.MyDataSet(test_X, test_Y), 64, True)
    dev_loader = Data.DataLoader(utils.MyDataSet(dev_X, dev_Y), 64, True)
    # TODO: RNN or CNN train
    # model = TextRNN(vocab_size=len(word2id), embedding_size=200, hidden_size=64, classes=classes, pretrained=word2vec, type_='LSTM', bidirectional=True, num_of_layers=1, dropout=0).cuda()
    model = TextCNN(vocab_size=len(word2id), embedding_size=200, classes=classes, pretrained=word2vec).cuda()
    print(model)
    learning_rate = 0.01
    epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    tqdm_method = trange(epochs)
    for epoch in tqdm_method:
        tqdm_method.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
        Loss = 0
        Acc = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.cuda()
            y = y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            Loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        total = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.cuda()
            y = y.cuda()
            total += len(y)
            pred = model(x)
            pred = torch.max(pred.data, 1)[1]
            Acc += torch.sum(pred == y)
        tqdm_method.set_postfix_str("Loss: {:.4f} Acc: {:.4f}".format(Loss, Acc / total))

    # TODO: predict
    pred_data = pd.read_csv('../data/test.tsv', sep='\t')
    pred_data_X = utils.get_id(pred_data['Phrase'].values, word2id, id2word).cuda()
    pred_loader = Data.DataLoader(utils.MyDataSet(pred_data_X, pred_data_X), 128, False)
    pred_Y = []
    for i, (x, y) in enumerate(pred_loader):
        x = x.cuda()
        pred = model(x)
        pred = list(torch.max(pred.data, 1)[1].cpu().data.numpy())
        pred_Y += pred
    ans = pd.DataFrame({'PhraseId': pred_data['PhraseId'].values, 'Sentiment': pred_Y})
    ans.to_csv('task2.csv', index=False)
    # TODO: optimize
