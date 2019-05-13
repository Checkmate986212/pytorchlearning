import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

#import argparse
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 定义网络结构
class LeNet_teachehr(nn.Module):
    def __init__(self):
        super(LeNet_teachehr, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0)
        self.bn1 = nn.BatchNorm2d(20)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 50, 5, padding=0)
        self.bn2 = nn.BatchNorm2d(50)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 500, 4, padding=0),
            nn.BatchNorm2d(500),
            nn.ReLU(),

        )
        self.conv4 = nn.Conv2d(500, 10, 1, padding=0)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        #out = []
        x = self.conv1(x)
        #out.append(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        #out.append(x)
        x = self.conv2(x)
        #out.append(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size()[0], -1) #view类似于numpy的reshape,把一张图展平成D个特征量
        return x


EPOCH = 1   #遍历数据集次数
BATCH_SIZE = 100      #批处理尺寸(batch_size)
LR = 0.001        #学习率

# 数据预处理方式
def data_tf(x):
    data_aug = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(45),
        # transforms.ColorJitter(contrast=1),
        # transforms.RandomResizedCrop(28,interpolation=2),
        transforms.Resize(28),
        transforms.ToTensor(),
    ])
    x = data_aug(x)
    return x

#transform = transforms.ToTensor()

# 训练数据集
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=data_tf)

# 训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

# 定义测试数据集
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transforms.ToTensor())

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    )
# 定义损失函数loss function 和优化方式（采用Adam）
net = LeNet_teachehr().to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
#optimizer = optim.SGD(net.parameters(),lr=LR, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.01)

# 训练
if __name__ == "__main__":

    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                net.eval()
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('accuracy after %d epoch：%.2f%%' % (epoch + 1, (100 * correct.__index__() / total)))
    torch.save(net.state_dict(), './LeNet_teacher.pth')