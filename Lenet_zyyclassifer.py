import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
#import argparse
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 定义网络结构
class LeNet_zyy(nn.Module):
    def __init__(self):
        super(LeNet_zyy, self).__init__()
        self.conv1 = nn.Sequential(     #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, padding=2), #padding=2保证输入输出尺寸相同
            nn.BatchNorm2d(6),
            nn.Dropout(0.5),
            nn.ReLU(),      #input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),#output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),
            nn.ReLU(),      #input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  #output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            #nn.BatchNorm2d(120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            #nn.BatchNorm2d(84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        out = []
        x = self.conv1(x)
        out.append(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1) #view类似于numpy的reshape,把一张图展平成D个特征量
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x, out

def LZ(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LeNet_zyy()
    if pretrained:
        #model.load_state_dict(torch.load('/home/yuhan_jiang/handwriting/LeNet_zyy.pth'))
        model.load_state_dict(torch.load('./LeNet_zyy.pth'))
    return model