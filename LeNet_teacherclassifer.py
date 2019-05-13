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
        print('conv1',x.shape)
        out = []
        x = self.conv1(x)
        out.append(x)
        x = self.bn1(x)
        x = self.maxpool1(x)
        out.append(x)
        print('conv2', x.shape)
        x = self.conv2(x)
        out.append(x)
        x = self.bn2(x)
        x = self.maxpool2(x)
        # print('before conv3', x.shape)
        # x = self.conv3(x)
        # print('after conv3', x.shape)
        # x = self.conv4(x)
        # x = x.view(x.size()[0], -1)
        return x, out

def LNT(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LeNet_teachehr()
    if pretrained:
        #model.load_state_dict(torch.load('/home/yuhan_jiang/handwriting/LeNet_zyy.pth'))
        model.load_state_dict(torch.load('./LeNet_teacher.pth'))
    return model
