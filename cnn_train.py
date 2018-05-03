# coding: utf-8
import time

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pandas as pd
from chainer import Chain, Variable
from chainer import cuda
from chainer import optimizers
from sklearn.model_selection import train_test_split


class CNN(Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 20, 5)
            self.conv2 = L.Convolution2D(None, 50, 5)
            self.l1 = L.Linear(None, 500)
            self.l2 = L.Linear(None, 2)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.dropout(F.relu(self.l1(h)), 0.2)
        h = self.l2(h)
        return h


if __name__ == '__main__':
    batchsize = 64
    n_epoch = 50

    df_train = pd.read_csv('./input/raw_data/train_data.csv')
    X = df_train.iloc[:, 2:].astype(np.float32).values
    y = (df_train.iloc[:, 1].astype(np.int32) - 2).values

    X /= 255.0

    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    N = y_train.size
    N_test = y_test.size

    # 画像を (nsample, channel, height, width) の4次元テンソルに変換
    X_train = X_train.reshape((len(X_train), 1, 32, 96))
    X_test = X_test.reshape((len(X_test), 1, 32, 96))

    model = CNN()

    gpu_flag = -1

    if gpu_flag >= 0:
        cuda.get_device_from_id(gpu_flag).use()
        model.to_gpu()
    xp = np if gpu_flag < 0 else cuda.cupy

    optimizer = optimizers.MomentumSGD(lr=0.01)
    optimizer.setup(model)
    # 訓練ループ
    start_time = time.clock()
    for epoch in range(1, n_epoch + 1):
        print "epoch: %d" % epoch,

        perm = np.random.permutation(N)
        sum_loss = 0
        for i in range(0, N, batchsize):
            x_batch = Variable(xp.asarray(X_train[perm[i:i + batchsize]]))
            y_batch = Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

            model.cleargrads()

            loss = F.softmax_cross_entropy(model(x_batch), y_batch)

            loss.backward()

            optimizer.update()
            model.cleargrads()

            sum_loss += float(loss.array) * len(x_batch)

        print "train mean loss: %f" % (sum_loss / N),

        with chainer.using_config('train', True):
            sum_accuracy = 0
            for i in range(0, N_test, batchsize):
                x_batch = xp.asarray(X_test[i:i + batchsize])
                y_batch = xp.asarray(y_test[i:i + batchsize])

                acc = F.accuracy(model(x_batch), y_batch, )
                sum_accuracy += float(acc.data) * len(y_batch)

        print "test accuracy: %f" % (sum_accuracy / N_test)

    end_time = time.clock()
    print end_time - start_time

    from chainer.serializers import save_npz

    # CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
    model.to_cpu()
    save_npz("output/cnn_model.npz", model)
