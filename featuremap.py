from Lenet_zyyclassifer import LZ
from LeNet_teacherclassifer import LNT
import matplotlib
#matplotlib.use('Agg')
import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt
from pylab import *
import sys

if sys.argv[1] == 'teacher':
    net = LNT(pretrained=True)
elif sys.argv[1] == 'zyy':
    net = LZ(pretrained=True)
#net = LNT(pretrained=True)
img_pil = Image.open('./9.JPG')         # PIL.Image.Image对象          # (H x W x C), [0, 255], RGB
images = img_pil.convert('L')

images = PIL.ImageOps.invert(images)
#images.show()

def data_tf(x):
    data_aug = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
    ])
    x = data_aug(x)
    return x


images = data_tf(images)
images = images.reshape(-1, 1, 28, 28)
print('images.shape', images.shape)
output, feature = net(images)

feature_0 = feature[0]
featuremap_1 = torch.squeeze(feature[0], dim=0)
featuremap_1_arr = featuremap_1.data.cpu().numpy()
print('featuremap_1_arr shape:', featuremap_1_arr.shape)
length_1 = featuremap_1_arr.shape[0]

fig = plt.figure()
for i in range(0,length_1):
    subplot(4,5,i+1)
    imshow(featuremap_1_arr[i], cmap='gray')
    axis('off')


show()
plt.savefig('feature.png')

