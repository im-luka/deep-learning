from google.colab import drive
drive.mount('/content/drive')

!pip install pkbar

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        # Shortcut connection to downsample residual
        # In case the output dimensions of the residual block is not the same 
        # as it's input, have a convolutional layer downsample the layer 
        # being bought forward by approporate striding and filters
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        # Create blocks
        self.block1 = self._create_block(64, 64, stride=1)
        self.block2 = self._create_block(64, 128, stride=2)
        self.block3 = self._create_block(128, 256, stride=2)
        self.block4 = self._create_block(256, 512, stride=2)
        self.linear = nn.Linear(512, num_classes)
    
    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
	# Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time, sleep

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image
import pkbar
from torch.utils.tensorboard import SummaryWriter

cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')
if cuda:
    device = torch.device('cuda')
print('device is', device)

writer = SummaryWriter("/content/drive/MyDrive/DL/Lab08/runs/CIFAR")

batch_size = 32

transform_train = transforms.Compose([
    transforms.ColorJitter(hue=.1, saturation=.75),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_dataset = datasets.CIFAR10('cifar_data', download=True, train=True, transform=transform_train)
test_dataset = datasets.CIFAR10('cifar_data', download=True, train=False, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

model = ResNet().to(device)

loss_fn = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

epochs = 20
train_per_epoch = int(len(train_dataset) / batch_size)
for e in range(epochs):
    kbar = pkbar.Kbar(target=train_per_epoch, epoch=e, num_epochs=epochs, width=20, always_stateful=False)
    for idx, (images, labels) in enumerate(train_loader):
        
        images = images.to(device)
        optimizer.zero_grad()
        output = model(images)
        labels = labels.to(device)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.item(), (e * train_per_epoch) + idx)
        predictions = output.argmax(dim=1, keepdim=True).squeeze()
        correct = (predictions == labels).sum().item()
        accuracy = correct / len(predictions)
        kbar.update(idx, values=[("loss", loss), ("acc", accuracy)])
        writer.add_scalar('acc', accuracy, (e * train_per_epoch) + idx)

num_correct = 0
num_samples = 0
model.eval()

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device=device)
        y = y.to(device=device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(f'Dobio sam točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')

%load_ext tensorboard
%tensorboard --logdir '/content/drive/MyDrive/DL/Lab08/runs/CIFAR'