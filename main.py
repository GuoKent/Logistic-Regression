import numpy as np
import csv
import matplotlib.pyplot as plt


def load_data(filename):
    fr = csv.reader(open(filename))
    for line in fr:
        x.append(line[0:2])
        y.append(float(line[-1]))
    return x, y


def sigmod(z):
    return 1.0 / (1 + np.exp(-z))


def granddescent(x, y, alpha=0.0001):
    w = np.array([0, 0, 0])  # 初始化w的值，w1、w2、b
    dE = np.array([10, 10, 10])  # 含有三个数的向量，分别是w1,w2,b的梯度，这里任取初始值
    while abs(np.linalg.norm(dE)) > 0.01:
        z = np.dot(w, x.T)
        dE_k = dE
        dE = np.dot(sigmod(z) - y, x)  # dE = np.array([dE1, dE2, dEb])
        w_k = w  # 记录当前迭代时w的值
        w = w - alpha * dE  # 第k+1次迭代的w值
        if abs(np.linalg.norm(w_k - w)) < 0.01:  # 如果||w1-w2||<0.01，停止迭代
            return w
    return w


x = []
y = []
# load_data("F:/University/大二上/机器学习/Assignment1/datamat/15_train.csv")
load_data("3_train.csv")
for i in range(len(x)):  # 将读取的数据x传化为float型
    x[i][0] = float(x[i][0])
    x[i][1] = float(x[i][1])
    x[i].append(1)  # 在x后面添1，便于与b相乘
x = np.array(x)
# print(x)
# print(y)

w = granddescent(x, y)
w1, w2, b = w
print(w)  # 求出最后拟合结果的w = [w1,w2,b]
print('w1=%f' % w1)
print('w2=%f' % w2)
print('b=%f' % b)
z = np.dot(w, x.T)

#  PR/ROC曲线
probability = []
D = {}
TP, FP, TN, FN = 0, 0, 0, 0
P = []  # 查准率P
R = []  # 查全率R
TPR = []  # 真正例率
FPR = []  # 假正例率
predict = [1] * len(z)
for i in range(len(z)):
    probability.append(sigmod(z[i]))  # 用一个列表储存概率
    D.update({probability[i]: y[i]})  # 储存各概率对应的位次
D_sort = sorted(D, reverse=True)  # 对概率降序排序，返回一个列表
# print(D[D_sort[0]])
# print(predict[0])
for i in range(len(z)):
    if predict[i] == 1 and y[i] == 1:
        TP += 1
    elif predict[i] == 0 and y[i] == 1:
        FN += 1
    elif predict[i] == 1 and y[i] == 0:
        FP += 1
    else:
        TN += 1
for i in range(len(z)):
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)
    P.append(p)
    R.append(r)
    TPR.append(tpr)
    FPR.append(fpr)
    predict[len(z) - i - 1] = 0
    for j in range(len(z)):
        if predict[j] == 1 and D[D_sort[j]] == 1:
            TP += 1
        elif predict[j] == 0 and D[D_sort[j]] == 1:
            FN += 1
        elif predict[j] == 1 and D[D_sort[j]] == 0:
            FP += 1
        else:
            TN += 1
# print(P)
# print(R)


# 画出图像
x1 = []
x2 = []
plt.subplot(1, 3, 1)
for i in range(len(x)):
    x1.append(x[i][0])
    x2.append(x[i][1])
    if y[i] == 0:
        plt.plot(x1[i], x2[i], 'ro', label='label')
    else:
        plt.plot(x1[i], x2[i], 'xb', label='label')
x1_line = np.arange(20, 100)
x2_line = -w1 / w2 * x1_line - b / w2
plt.title('Logistic')
plt.plot(x1_line, x2_line)

plt.subplot(1, 3, 2)
plt.title('P-R Curve')
plt.plot(P, R)

plt.subplot(1, 3, 3)
plt.title('ROC Curve')
plt.plot(FPR, TPR)
plt.show()
