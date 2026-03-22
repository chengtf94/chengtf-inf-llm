# coding: utf-8
import sys, os
print("当前工作目录:", os.getcwd())
print("sys.path 路径:", sys.path)

# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
# 自动获取当前脚本所在目录的父目录（即项目根目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)  # 插入到最前面，优先查找
print("加入后的 sys.path:", sys.path)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
