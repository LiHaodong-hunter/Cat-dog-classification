# 机 构：中国科学院大学
# 程序员：李浩东
# 时 间：2022/9/27 20:51

# 导入库
import torch
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

# 超参数定义
BATCH_SIZE = 20  # 每次批处理的数据数量
LR = 1e-3  # 学习率
EPOCHS = 10  # 训练样本训练轮数
EPOCH = 1  # 测试样本测试轮数

# 图片预处理，加载数据集
# 使用transforms.Compose()函数设置预处理方式
transform = transforms.Compose([
    # 对加载的图像作归一化处理
    transforms.CenterCrop(224),  # 将所有图像大小调整为（224x224x3）
    transforms.ToTensor(),  # 将数据转化为Tensor对象
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 将数据进行归一化处理,各通道的均值,各通道的标准差
])

# 加载数据集
# 训练集
# 在pytorch中提供了：torchvision.datasets.ImageFolder让我们训练自己的图像,要求：先创建train和test文件夹，每个文件夹下按照类别名字存储对应的图像就可以了
trainset = datasets.ImageFolder('data/train', transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)  # 用PyTorch的数据加载工具DataLoader来加载数据
# print(trainset)
# print(trainloader)
# if __name__ == '__main__':
#     for i, (imgs, labels) in enumerate(trainloader):
#         print(i,imgs,labels)

# 测试集
testset = datasets.ImageFolder('data/val', transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
# print(testset)


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积池化层
        self.conv1 = nn.Conv2d(3, 6, 5)  # 6个3*3的卷积核，生成6个220*220的特征图
        # self.dropout1 = nn.Dropout(p=0.5)  # 丢掉50%的神经元，防止过拟合
        # self.max_pool1 = F.max_pool2d(5)  # 5*5的池化核，生成6个44*44的特征图
        # 第二个卷积池化层
        self.conv2 = nn.Conv2d(6, 12, 5)  # 12个5*5的卷积核，生成12个40*40的特征图
        # self.dropout2 = nn.Dropout(p=0.5)  # 丢掉50%的神经元，防止过拟合
        # self.max_pool2 = F.max_pool2d(5)  # 5*5的池化核，生成12个8*8的特征图
        # 第三个卷积池化层
        self.conv3 = nn.Conv2d(12, 24, 3)  # 24个3*3的卷积核，生成24个6*6的特征图
        # self.dropout3 = nn.Dropout(p=0.5)  # 丢掉50%的神经元，防止过拟合
        # self.max_pool3 = F.max_pool2d(2)  # 2*2的池化核，生成24个3*3的特征图
        # 平化
        self.flatten = nn.Flatten()  # 将4D张量展平为2D张量
        # 第一个全连接层
        self.fc1 = nn.Linear(24 * 3 * 3, 64)  # 全连接层
        # self.dropout4 = nn.Dropout(p=0.5)  # 丢掉50%的神经元，防止过拟合
        # 第二个全连接层
        self.fc2 = nn.Linear(64, 16)  # 全连接层
        # self.dropout5 = nn.Dropout(p=0.5)  # 丢掉50%的神经元，防止过拟合
        # 第三个全连接层
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        # print(x.shape)  # torch.Size([100, 3, 224, 224])
        # x = x.view(x.size(0), -1)  # 维度变换,将多维度的tensor展平成一维,此时的输出x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值
        x = F.max_pool2d(F.relu(self.conv1(x)), 5)
        self.dropout = nn.Dropout(p=0.5)
        # print(x.shape)  # torch.Size([100, 12, 44, 44])
        x = F.max_pool2d(F.relu(self.conv2(x)), 5)
        self.dropout = nn.Dropout(p=0.5)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        self.dropout = nn.Dropout(p=0.5)
        x = self.flatten(x)
        # x = x.view(x.size(0), -1)  # 维度变换,将多维度的tensor展平成一维,此时的输出x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)，x.size(0)指batchsize的值
        x = F.relu(self.fc1(x))
        self.dropout = nn.Dropout(p=0.5)
        x = F.relu(self.fc2(x))
        self.dropout = nn.Dropout(p=0.5)
        x = torch.sigmoid(self.fc3(x))
        return x


# 创建对象
CUDA = torch.cuda.is_available()
if CUDA:
    cnn = CNN().cuda()
else:
    cnn = CNN()



# 优化器和损失函数
optimizer = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-3)  # 优化器,weight_decay=1e-8就是使用了L2正则化
criterion = nn.CrossEntropyLoss()  # 损失函数


# 训练模型
def train(model, dataset,criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        train_correct = 0.0
        train_total = 0.0
        for i, (imgs, labels) in enumerate(dataset):
            output = model(imgs)  # 前向传播
            # print(output)
            # print(output.data)  # 该图片属于哪一类图片的概率
            loss = criterion(output, labels)  # 计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_predict = torch.max(output.data, 1)[1]  # 是计算模型中每个类别的最大值并返回其索引值，即该类别的标签值
            if CUDA:
                train_correct += (train_predict.cuda() == labels.cuda()).sum()
            else:
                train_correct += (train_predict == labels).sum()
            train_total += labels.size(0)
            accuracy = train_correct / train_total * 100.0
            print("Train--->Epoch:%d,  Batch:%3d,  Loss:%.8f,  train_correct:%d,  train_total:%d,  accuracy:%.6f" % (
                epoch + 1, i + 1, loss.item(), train_correct, train_total, accuracy))


# 测试模型
def test(model, dataset, criterion, optimizer, epochs):
    model.eval()
    for epoch in range(epochs):
        test_correct = 0.0
        test_total = 0.0
        loss = 0.0
        for i, (imgs, labels) in enumerate(dataset):
            with torch.no_grad():
                output = model(imgs)  # 前向传播
                loss = criterion(output, labels)  # 计算损失
                test_predict = torch.max(output.data, 1)[1]
                if CUDA:
                    test_correct += (test_predict.cuda() == labels.cuda()).sum()
                else:
                    test_correct += (test_predict == labels).sum()
                test_total += labels.size(0)
                accuracy = test_correct / test_total * 100.0
                print("Test--->Epoch:%d,  Batch:%3d,  Loss:%.8f,  test_correct:%d,  test_total:%d,  accuracy:%.6f" % (
                    epoch + 1, i + 1, loss.item(), test_correct, test_total, accuracy))


def save_param(model, path):
    torch.save(model.state_dict(), path)  # 保存网络里的参数


def load_param(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    print('-------------------------------------------训练开始-----------------------------------------------')
    train(cnn, trainloader, criterion, optimizer, EPOCHS)  # 训练
    save_param(cnn, 'model.cnn')
    load_param(cnn, 'model.cnn')
    print('-------------------------------------------测试开始-----------------------------------------------')
    test(cnn, testloader, criterion, optimizer, EPOCH)  # 测试
