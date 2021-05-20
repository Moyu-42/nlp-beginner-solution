import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import utils
from models import RNN, CNN
from tqdm import tqdm, trange

if __name__ == '__main__':
    data = pd.read_csv('../data/train.tsv', sep='\t')
    # TODO: Word Embedding
    X = data['Phrase'].values
    Y = data['Sentiment'].values
    classes = len(np.unique(Y))
    X, word2vec, word2id, id2word = utils.make_data(X)
    train_X, train_Y, test_X, test_Y, dev_X, dev_Y = utils.split(X, Y)
    # TODO: split and mini-batch use DataLoader
    train_loader = Data.DataLoader(utils.MyDataSet(train_X, train_Y), 128, True)
    test_loader = Data.DataLoader(utils.MyDataSet(test_X, test_Y), 128, True)
    dev_loader = Data.DataLoader(utils.MyDataSet(dev_X, dev_Y), 128, True)
    # TODO: RNN or CNN train
    model = RNN(len(word2id), 50, 64, classes, word2vec).cuda()
    print(model)
    learning_rate = 0.01
    epochs = 30
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

    # TODO: optimize
