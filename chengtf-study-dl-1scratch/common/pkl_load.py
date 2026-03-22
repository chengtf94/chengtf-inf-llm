import pickle

# 替换成你的 .pkl 文件路径
pkl_path = "../dataset/mnist.pkl"

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 查看数据结构
print("数据类型:", type(data))  # 一般是 dict
print("键名:", data.keys())     # 比如 MNIST 会有 'train_img', 'train_label' 等

# 查看具体数据
print("训练集图片形状:", data['train_img'].shape)
print("训练集标签形状:", data['train_label'].shape)