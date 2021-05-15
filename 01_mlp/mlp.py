from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, './handson20210511')
import test_mlp 

X, Y = fetch_openml('mnist_784', version=1, data_home="./data/", return_X_y=True)

X = X / 255.
Y = Y.astype("int")

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=2)
train_y = np.eye(10)[train_y].astype(np.int32)
test_y = np.eye(10)[test_y].astype(np.int32)
train_n = train_x.shape[0]
test_n = test_x.shape[0]

class Sigmoid:
    def __init__(self):
        self.y = None
        
    def __call__(self, x):
        y = 1/(1+np.exp(-x))  # 順伝播計算
        self.y = y
        return y
    
    def backward(self):
        return self.y * (1-self.y)  # 逆伝播計算
    
class ReLU:
    def __init__(self):
        self.x = None
        
    def __call__(self, x):
        self.x = x
        return x * (x > 0)   # 順伝播計算
    
    def backward(self):
        return 1 * (self.x > 0)   # 逆伝播計算
    
class Softmax:
    def __init__(self):
        self.y = None
        
    def __call__(self, x):
        exp_x = np.exp(x-x.max(axis=1, keepdims=True))  # ここで exp(x - x_max) を計算しよう
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)  # exp_x を用いて softmax を計算しよう
        self.y = y
        return y
    
class Linear:
    def __init__(self, in_dim, out_dim, activation):
        self.W = np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.activation = activation()
        self.delta = None
        self.x = None
        self.dW = None
        self.db = None

    def __call__(self, x):
        # 順伝播計算
        self.x = x
        u = np.dot(x, self.W) + self.b  # self.W, self.b, x を用いて u を計算しよう
        self.z = self.activation(u)
        return self.z
    
    def backward(self, dout):
        # 誤差計算
        self.delta = dout * self.activation.backward()  # dout と活性化関数の逆伝播 (self.activation.backward()) を用いて delta を計算しよう
        dout = np.dot(self.delta, self.W.T)  # self.delta, self.W を用いて 出力 o を計算しよう
        
        # 勾配計算
        self.dW = np.dot(self.x.T, self.delta)   # dW を計算しよう
        self.db = np.dot(np.ones(len(self.x)), self.delta)  # db を計算しよう
        
        return dout

class MLP():
    def __init__(self, layers):
        self.layers = layers
        
    def train(self, x, t, lr):     
        # 1. 順伝播
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)  # 順伝播計算を順番に行い， 出力 y を計算しよう
        
        # 2. 損失関数の計算
        self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        
        # 3. 誤差逆伝播
        # 3.1. 最終層
        # 3.1.1. 最終層の誤差・勾配計算
        batchsize = len(self.layers[-1].x)
        delta = (self.y - t) / batchsize
        self.layers[-1].delta = delta
        self.layers[-1].dW = np.dot(self.layers[-1].x.T, self.layers[-1].delta)
        self.layers[-1].db = np.dot(np.ones(batchsize), self.layers[-1].delta)
        dout = np.dot(self.layers[-1].delta, self.layers[-1].W.T)
        
        # 3.1.2. 最終層のパラメータ更新
        self.layers[-1].W -= lr * self.layers[-1].dW # self.layers[-1].dW を用いて最終層の重みを更新しよう
        self.layers[-1].b -= lr * self.layers[-1].db  # self.layers[-1].db を用いて最終層のバイアスを更新しよう
        
        # 3.2. 中間層
        for layer in self.layers[-2::-1]:
            # 3.2.1. 中間層の誤差・勾配計算
            dout = layer.backward(dout)   # 逆伝播計算を順番に実行しよう
            
            # 3.2.2. パラメータの更新
            layer.W -= lr * layer.dW  # 各層の重みを更新
            layer.b -= lr * layer.db  # 各層のバイアスを更新
            
        return self.loss

    def test(self, x, t):
        # 性能をテストデータで調べるために用いる
        # よって，誤差逆伝播は不要
        # 順伝播 (train関数と同様)
        self.y = x
        for layer in self.layers:
            self.y = layer(self.y)
        self.loss = np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        return self.loss

model = MLP([Linear(784, 1000, ReLU),
                      Linear(1000, 1000, ReLU),
                      Linear(1000, 10, Softmax)])

n_epoch = 40
batchsize = 8
lr = 0.005 # 0.5

for epoch in range(n_epoch):
    print('epoch %d | ' % epoch, end="")
    
    # 訓練
    sum_loss = 0
    pred_y = []
    perm = np.random.permutation(train_n)
    
    for i in range(0, train_n, batchsize):
        x = train_x[perm[i: i+batchsize]]
        t = train_y[perm[i: i+batchsize]]
        sum_loss += model.train(x, t, lr) * len(x)
        # model.y には， (N, 10)の形で，画像が0~9の各数字のどれに分類されるかの事後確率が入っている
        # そこで，最も大きい値をもつインデックスを取得することで，識別結果を得ることができる
        pred_y.extend(np.argmax(model.y, axis=1))
    
    loss = sum_loss / train_n
    
    # accuracy : 予測結果を1-hot表現に変換し，正解との要素積の和を取ることで，正解数を計算できる．
    accuracy = np.sum(np.eye(10)[pred_y] * train_y[perm]) / train_n
    print('Train loss %.3f, accuracy %.4f | ' %(loss, accuracy), end="")
    
    
    # テスト
    sum_loss = 0
    pred_y = []
    
    for i in range(0, test_n, batchsize):
        x = test_x[i: i+batchsize]
        t = test_y[i: i+batchsize]
        
        sum_loss += model.test(x, t) * len(x)
        pred_y.extend(np.argmax(model.y, axis=1))

    loss = sum_loss / test_n
    accuracy = np.sum(np.eye(10)[pred_y] * test_y) / test_n
    print('Test loss %.3f, accuracy %.4f' %(loss, accuracy))