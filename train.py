import os
import warnings
warnings.filterwarnings('ignore')
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import torchvision.transforms as transforms
from models import LeNet , VGG9
import torch.optim as optim
from tqdm import tqdm
import torchvision
import numpy as np


# 执行训练的卡，cuda表示用显卡训练，cpu表示用cpu训练
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 图片增广+转换成tensor
apply_transform = transforms.Compose([
    transforms.ToTensor(), # 将图片封装成tensor便于pytorch处理
    transforms.Resize(32), # 对图片进行裁剪，比如一张图片是（28，28），裁剪后，变成（32，32）
    transforms.Normalize((0.5,), (0.5,)), # 对图片的像素值进行 减去均值（0.5），除去方差（0.5）
])


def seed_everything(seed) :
    # 固定训练过程的随机数
    random.seed(seed)
    os.environ['PYTHONHASHSEED' ] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_train(seed,model_type):

    seed_everything(seed) # 固定训练过程的随机数

    os.makedirs('results/',exist_ok=True) # 创建目录，如果已经存在也ok
    # 设置参数
    EPOCH=100                                   # 训练的迭代数量
    LEARNING_RATE=1e-2                          # 网络的学习率
    BATCH_SIZE=128                              # 一起训练多少图片

    # 加载数据集
    train = torchvision.datasets.FashionMNIST(
        root='data/', train=True, download=True, transform=apply_transform)
    trainloader = DataLoader(
        train,                  # 训练集
        batch_size=BATCH_SIZE,  # 一次训练多少张图片
        shuffle=True,           # 是否对其进行打乱，随机从中抽取图片
        num_workers=2           # cpu用多少个worker进程进行数据加载，一般越多越快，但是越多越吃资源
    )
    test = torchvision.datasets.FashionMNIST(
        root='data/', train=False, download=True, transform=apply_transform)
    testloader = DataLoader(
        test,                   # 测试集，用于验证模型的训练情况
        batch_size=BATCH_SIZE,
        shuffle=False,          # 不打乱数据
        num_workers=2
    )

    print(f'Train Set Number {len(train)} Test Set Number {len(test)}')

    # 指定网络模型，要么用LeNet，要么用VGG9
    model = LeNet() if model_type.lower() == 'lenet' else VGG9() 
    model.to(DEVICE)                # 把模型放到对应的设备上进行训练
    criterion = nn.CrossEntropyLoss()       # 指定模型的损失函数，本方案是多分类所以用交叉熵
    # 指定优化器，这里用随机提督下降法
    optimizer = optim.SGD(  
        model.parameters(),         # 指定要“模型参数”进行训练
        lr=LEARNING_RATE,           # 模型的学习率
        momentum=0.9,               # 
        weight_decay=0.0001         # 对模型参数的限制，越大，则越限制模型参数的更新步长
    )

    # 余弦学习率衰减
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,                      # 指定优化器
        T_max=EPOCH * len(trainloader) # 学习率逐渐衰减为0
    )
    # 开始训练
    print('Start Train')
    train_loss_list , test_loss_list = [] , [] # 记录训练损失、测试损失
    accuracy_list = []  # 记录模型识别准确率
    best_acc = float('-inf')
    for epoch in range(EPOCH):                                      # for训练，一共轮数EPOCH
        progress_bar = tqdm(trainloader)                            # 加载进度条
        train_loss=0.                                               # 记录一轮内训练损失的变化
        model.train()                                               # 让模型进行训练状态
        for _,(image,target) in enumerate(progress_bar):            # 对训练集进行for循环，一个epoch会走完一次完整的训练集
            progress_bar.set_description(f'Ep {epoch} Training...')
            image , target = image.to(DEVICE) , target.to(DEVICE)   # 将image、target两个张量放到指定的设备上执行
            target = target.view(-1)                                # 对target进行维度转换
            output = model(image)                                   # 把image喂入模型得到对应的模型的预测输出
            loss = criterion(output,target)                         # 计算输出和真值之间的损失差距
            ## backward -> step -> zero_grad 这三步是固定公式
            loss.backward()                                         # 把损失值进行梯度反向传播给模型
            optimizer.step()                                        # 优化器对模型参数进行更新
            optimizer.zero_grad()                                   # 优化器对模型参数进行梯度清零
            train_loss += loss.detach().item()                      # 记录模型损失值
            lr_scheduler.step()                                     # 对学习率迭代器进行更新，使学习率越来越小
        train_loss /= len(trainloader)                              # 训练损失取均值
        
        # model eval
        model.eval()                                                # 模型进行验证状态
        test_loss = 0.
        acc , cnt = 0, 0
        with torch.no_grad():                                       # 验证阶段要关闭梯度记录，可以减少内存占用，加快计算速度
            progress_bar = tqdm(testloader)                         # 对验证集进行循环
            for image,target in progress_bar:
                progress_bar.set_description(f'Ep {epoch} Testing...')
                image , target = image.to(DEVICE) , target.to(DEVICE)   # 指定设备
                output = model(image)                                   # 模型输出
                loss = criterion(output,target.view(-1))                # 计算预测值和真值之间的损失差距
                _, preds = torch.max(output, 1)                         # 去除每个case中预测最大的那个标签
                preds = preds.view(-1)                                  # 进行维度转换
                acc += preds.eq(target.view(-1)).sum().detach().item()  # 计算预测值和真值之间相等的数量，相等意味着预测正确
                test_loss += loss.detach().item()                       
                cnt += len(preds)
        test_loss /= len(testloader)
        accuracy = acc / cnt                                            # 准确度取平均值

        # 记录训练过程
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        accuracy_list.append(accuracy)
    
        print(f'Ep {epoch} accuracy {accuracy:.6f} train_loss {train_loss:.6f} test_loss {test_loss:.6f}')

        # 保存结果最好的模型
        if accuracy >  best_acc:
            torch.save(model.state_dict(),f'results/{model_type}_{seed}_best.pth')  # 保存结果最好的模型
            best_acc = accuracy
    # 输出过程文件
    with open(f'results/{model_type}_{seed}_record.json','w') as file:
        json.dump({
            'train_loss':train_loss_list,
            'test_loss':test_loss_list,
            'accuracy':accuracy_list,
        },file,indent=4,ensure_ascii=False)


if __name__ == '__main__': 

    seed_train(seed=100,model_type='vgg')


    