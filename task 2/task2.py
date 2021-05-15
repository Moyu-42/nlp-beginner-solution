import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data as Data
import utils

if __name__ == '__main__':
    data = pd.read_csv('../data/train.tsv', sep='\t')
    # TODO: Word Embedding
    X = data['Phrase'].values
    Y = data['Sentiment'].values
    train_X, train_Y, test_X, test_Y, dev_X, dev_Y, word2vec = utils.make_data(X, Y)
    # TODO: split and mini-batch use DataLoader
    train_loader = Data.DataLoader(utils.MyDataSet(train_X, train_Y), 128, True)
    test_loader = Data.DataLoader(utils.MyDataSet(test_X, test_Y), 128, True)
    dev_loader = Data.DataLoader(utils.MyDataSet(dev_X, dev_Y), 128, True)
    # TODO: RNN or CNN train

    # TODO: predict

    # TODO: optimize
