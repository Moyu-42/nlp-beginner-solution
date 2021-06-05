import pandas as pd
import numpy as np
import utils
from ESIM import ESIM
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

if __name__ == '__main__':
    train_path = "../data/snli_1.0/snli_1.0_train.txt"
    test_path = "../data/snli_1.0/snli_1.0_test.txt"
    train_set, word2id, word2vec = utils.get_DataSet(train_path, dict, True)
    test_set, _, _ = utils.get_DataSet(test_path, word2id)
    train_loader = DataLoader(train_set, 32, True)
    test_loader = DataLoader(test_set, 32, True)

    model = ESIM(word2vec.shape[0], word2vec.shape[1], 300, 3, word2vec).cuda()
    print(model)
    learning_rate = 0.0004
    epochs = 200
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    tqdm_method = trange(epochs)
    for epoch in tqdm_method:
        tqdm_method.set_description("Epoch [{}/{}]".format(epoch + 1, epochs))
        Loss = 0
        Acc = 0
        model.train()
        for i, (p, h, y) in enumerate(train_loader):
            p = p.cuda()
            h = h.cuda()
            y = y.cuda()
            pred = model(p, h)
            loss = criterion(pred, y)
            Loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        total = 0
        for i, (p, h, y) in enumerate(test_loader):
            p = p.cuda()
            h = h.cuda()
            y = y.cuda()
            total += len(y)
            pred = model(p, h)
            pred = torch.max(pred.data, 1)[1]
            Acc += torch.sum(pred == y)
        tqdm_method.set_postfix_str("Loss: {:.4f} Acc: {:.4f}".format(Loss, Acc / total))
