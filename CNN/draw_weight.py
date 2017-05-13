#coding: utf-8
import cPickle
import matplotlib.pyplot as plt
model = cPickle.load(open("model.pkl", "rb"))

# 1つめのConvolution層の重みを可視化
print model.conv1.W.shape

n1, n2, h, w = model.conv1.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(n1):
    ax = fig.add_subplot(2, 2, i + 1, xticks=[], yticks=[])
    ax.imshow(model.conv1.W[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()