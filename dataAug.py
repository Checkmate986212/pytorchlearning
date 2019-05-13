import torch
import torchvision as tv
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
def data_tf(x):
    data_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(contrast=1),
        transforms.RandomResizedCrop(28,interpolation=2),
        transforms.Resize(28),
        transforms.ToTensor(),
    ])
    x = data_aug(x)
    return x

im = Image.open('./q.png')
plt.figure()
plt.imshow(im)
