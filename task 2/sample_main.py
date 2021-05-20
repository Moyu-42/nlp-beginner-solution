"""
Created on 2020/4/30 8:33
@author: phil
"""
from torch import optim
import torch
from sample_model import TextRNN, TextCNN
from sample_hand import make_dataloader
from models import RNN
import numpy as np

if __name__ == "__main__":
    model_names = ["LSTM", "RNN", "RNN"]  # 彩蛋：按过拟合难度排序，由难到易
    learning_rate = 0.001
    epoch_num = 500
    num_of_class = 5
    load_data_by_torchtext = False

    train_iter, val_iter, word_vectors, X_lang = make_dataloader(batch_size=100, debug=False)

    for model_name in model_names[-1:]:
        # model = RNN(vocab_size=len(word_vectors), embedding_size=50, hidden_size=128, classes=num_of_class, pretrained=word_vectors).cuda()
        if model_name == "RNN":
            model = TextRNN(vocab_size=len(word_vectors), embedding_dim=50, hidden_size=128, num_of_class=num_of_class, weights=word_vectors)
        elif model_name == "CNN":
            model = TextCNN(vocab_size=len(word_vectors), embedding_dim=50, num_of_class=num_of_class, embedding_vectors=word_vectors)
        elif model_name == "LSTM":
            model = TextRNN(vocab_size=len(word_vectors), embedding_dim=50, hidden_size=128, num_of_class=num_of_class, weights=word_vectors, rnn_type="LSTM")
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fun = torch.nn.CrossEntropyLoss()

        for epoch in range(epoch_num):
            model.train()  # 包含dropout或者BN的模型需要指定
            Loss = 0
            for i, batch in enumerate(train_iter):
                if load_data_by_torchtext:
                    x, y = batch.sent.t(), batch.label
                else:
                    x, y, lens = batch
                x = x
                y = y
                logits = model(x)
                # print(y.shape, logits.shape)
                optimizer.zero_grad()
                loss = loss_fun(logits, y)
                Loss += loss
                loss.backward()
                optimizer.step()

            # with torch.no_grad():
            model.eval()
            # train_accs = []
            # for i, batch in enumerate(train_iter):
            #     if load_data_by_torchtext:
            #         x, y = batch.sent.t(), batch.label
            #     else:
            #         x, y, lens = batch
            #     print(logits.shape)
            #     y_pre = torch.max(logits, 1)[1]
            #     print(y.shape, y_pre.shape)
            #     acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
            #     train_accs.append(acc)
            # train_acc = np.array(train_accs).mean()

            val_accs = []
            for i, batch in enumerate(val_iter):
                if load_data_by_torchtext:
                    x, y = batch.sent.t(), batch.label
                else:
                    x, y, lens = batch
                logits = model(x)
                _, y_pre = torch.max(logits, -1)
                acc = torch.mean((torch.tensor(y_pre == y, dtype=torch.float)))
                val_accs.append(acc)
            val_acc = np.array(val_accs).mean()
            print("epoch %d, loss: %.2f val acc:%.2f" % (epoch, Loss, val_acc))