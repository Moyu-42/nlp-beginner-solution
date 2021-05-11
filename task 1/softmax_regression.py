import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt

class SoftmaxRegression():
    def __init__(self, features, classes, alpha=0.01, l1=0.0, l2=0.0):
        self.W = np.random.rand(features, classes) # (features, classes)
        self.features = features
        self.classes = classes
        self.alpha = alpha
        # TODO: regularization
        self.l1 = l1
        self.l2 = l2
     
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def one_hot(self, x):
        one_hot_vec = np.zeros((x.shape[0], self.classes))
        one_hot_vec[np.arange(x.shape[0]), x.T] = 1
        return one_hot_vec
    
    def fit(self, train_X, train_Y):
        train_Y = self.one_hot(train_Y)
        m = train_X.shape[0]

        z = train_X.dot(self.W)
        y_hat = self.softmax(z)
        # CrossEntropy Loss
        loss = -1.0 / m * np.sum(train_Y * np.log(y_hat))
        dW = -1.0 / m * train_X.transpose().dot(train_Y - y_hat)

        if self.l1:
            loss += self.l1 / (2 * m) * np.sum(self.W)
            dW += self.l1 / (2 * m) * np.sign(self.W)
        elif self.l2:
            loss += self.l2 / (2 * m) * np.sum(np.square(self.W))
            dW += self.l2 / m * self.W
        self.W = self.W - self.alpha * dW
        return loss
    
    def predict(self, data_X):
        y_pred = np.argmax(data_X.dot(self.W), axis=1)
        return y_pred


if __name__ == '__main__':
    model = SoftmaxRegression(10, 5, alpha=0.01)
    x = np.random.randn(30, 10)
    y = np.random.randint(0, 5, (30, 1))
    train_X = x[: 20]
    train_Y = y[: 20]
    test_X = x[20: ]
    test_Y = y[20: ]

    loss = []
    epochs = 100
    for epoch in range(epochs + 1):
        Loss = model.fit(train_X, train_Y)
        loss.append(Loss)
        pred_Y = model.predict(test_X).reshape(-1, 1)
        acc = (pred_Y == test_Y).sum() / len(test_Y)
        print("Epoch [{}/{}] Loss: {:.4f} Acc: {:.4f}".format(epoch, epochs, Loss, acc))

    plt.figure(dpi=128, figsize=(6, 4))
    plt.plot([i for i in range(len(loss))], loss)
    plt.show()
