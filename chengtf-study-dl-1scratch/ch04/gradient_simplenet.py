# coding: utf-8
import sys, os
# sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
print("当前工作目录:", os.getcwd())
print("sys.path 路径:", sys.path)

# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
# 自动获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)  # 插入到最前面，优先查找
print("加入后的 sys.path:", sys.path)

import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()
# print(net.W)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
