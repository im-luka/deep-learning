from google.colab import drive
drive.mount('/content/drive')

!pip install pkbar

import torch
import torch.nn as nn
import torch.nn.functional as F

class Convo_Layer(nn.Module):
    def __init__(self):
        super(Convo_Layer, self).__init__()
        # CONVOLUTIONAL
        # CONVO 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # CONVO 2
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # CONVO 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # CONVO 4
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # CONVO 5
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)

        # CONVO 6
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # LINEAR
        # LINEAR 1
        self.dropout1 = nn.Dropout(p = 0.5)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.bn7 = nn.BatchNorm1d(512)

        # LINEAR 2
        self.dropout2 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(512, 512)
        self.bn8 = nn.BatchNorm1d(512)

        # LINEAR 3
        self.dropout3 = nn.Dropout(p = 0.5)
        self.fc3 = nn.Linear(512, 13)

    def forward(self, x):
        # CONVO
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool3(x)

        # LINEAR
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn7(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


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

cars_train_path = "/content/drive/MyDrive/DL/Lab05/cars/train"
cars_test_path = "/content/drive/MyDrive/DL/Lab05/cars/test"
writer = SummaryWriter("/content/drive/MyDrive/DL/Lab05/runs/CARS_WRITING")

batch_size = 128

cars_transform_train = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ColorJitter(hue=.1, saturation=.75),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cars_transform_test = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cars_train_dataset = torchvision.datasets.ImageFolder(root=cars_train_path, transform=cars_transform_train)
cars_test_dataset = torchvision.datasets.ImageFolder(root=cars_test_path, transform=cars_transform_test)

cars_train_loader = DataLoader(cars_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
cars_test_loader = DataLoader(cars_test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

model = Convo_Layer().to(device)

loss_fn = nn.NLLLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
# optimizer = optim.RMSprop(model.parameters())

epochs = 50
train_per_epoch = int(len(cars_train_dataset) / batch_size)
for e in range(epochs):
    kbar = pkbar.Kbar(target=train_per_epoch, epoch=e, num_epochs=epochs, width=20, always_stateful=False)
    for idx, (images, labels) in enumerate(cars_train_loader):

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
    for x, y in cars_test_loader:
        x = x.to(device=device)
        y = y.to(device=device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(f'Dobio sam točnih {num_correct} od ukupno {num_samples} što čini točnost od {float(num_correct) / float(num_samples) * 100:.2f}%')


%load_ext tensorboard
%tensorboard --logdir '/content/drive/MyDrive/DL/Lab05/runs/CARS_WRITING'