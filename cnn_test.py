# coding: utf-8
import csv
import time

import chainer
import chainer.functions as F
import numpy as np
from chainer import cuda
from chainer import optimizers
from sklearn.cross_validation import train_test_split

batchsize = 100
n_epoch = 100

dname = "raw_data"

f = open(dname + "/train_data.csv", "rb")
list = []
dataReader = csv.reader(f)

for row in dataReader:
    list.append(row)

f.close()

np_train = np.array(list)

X = np_train[:, 2:].astype(np.float32)[:]
y = np_train[:, 1].astype(np.int32)[:]

y[y == 2] = 0
y[y == 3] = 1

gpu_flag = -1

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

# ピクセルの値を0.0-1.0に正規化
X /= X.max()

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

N = y_train.size
N_test = y_test.size

# 画像を (nsample, channel, height, width) の4次元テンソルに変換
# MNISTはチャンネル数が1なのでreshapeだけでOK
X_train = X_train.reshape((len(X_train), 1, 32, 96))
X_test = X_test.reshape((len(X_test), 1, 32, 96))
# plt.imshow(X_train[1][0], cmap=pylab.cm.gray_r, interpolation='nearest')
# plt.show()

model = chainer.FunctionSet(conv1=F.Convolution2D(1, 20, 5),
                            # 入力1枚、出力20枚、フィルタサイズ5ピクセル
                            conv2=F.Convolution2D(20, 50, 5),
                            # 入力20枚、出力50枚、フィルタサイズ5ピクセル
                            l1=F.Linear(5250, 500),  # 入力800ユニット、出力500ユニット
                            l2=F.Linear(500, 2))  # 入力500ユニット、出力10ユニット

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()


def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(model.conv1(x)), 2)
    h = F.max_pooling_2d(F.relu(model.conv2(h)), 2)
    h = F.dropout(F.relu(model.l1(h)), train=train)
    y = model.l2(h)
    if train:
        return F.softmax_cross_entropy(y, t)
    else:
        return F.accuracy(y, t)


optimizer = optimizers.Adam()
optimizer.setup(model)

fp1 = open("accuracy.txt", "w")
fp2 = open("loss.txt", "w")

fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")

# 訓練ループ
start_time = time.clock()
for epoch in range(1, n_epoch + 1):
    print "epoch: %d" % epoch

    perm = np.random.permutation(N)
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = xp.asarray(X_train[perm[i:i + batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(y_batch)

    print "train mean loss: %f" % (sum_loss / N)
    fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
    fp2.flush()

    sum_accuracy = 0
    for i in range(0, N_test, batchsize):
        x_batch = xp.asarray(X_test[i:i + batchsize])
        y_batch = xp.asarray(y_test[i:i + batchsize])

        acc = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(acc.data) * len(y_batch)

    print "test accuracy: %f" % (sum_accuracy / N_test)
    fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp1.flush()

end_time = time.clock()
print end_time - start_time

fp1.close()
fp2.close()

import cPickle

# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
model.to_cpu()
cPickle.dump(model, open("model.pkl", "wb"), -1)
