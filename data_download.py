import os
from torchvision import datasets

data_dir = 'data/'
os.makedirs(data_dir,exist_ok=True)

# 下载数据集，pytorch封装好了API
    
train_dataset = datasets.FashionMNIST(
    data_dir,   # 指定读取的目录
    train=True, # 训练集合
    download=True # 如果数据集不存在，就进行下载
)
test_dataset = datasets.FashionMNIST(
    data_dir,  # 指定读取的目录
    train=False, # 测试集合
    download=True # 如果数据集不存在，就进行下载
)   