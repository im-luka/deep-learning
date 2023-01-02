from google.colab import drive
drive.mount('/content/drive')

!pip install pkbar

import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time, sleep

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.optim import lr_scheduler
from PIL import Image
import pkbar
from torch.utils.tensorboard import SummaryWriter

cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')
if cuda:
    device = torch.device('cuda')

cars_train_path = "/content/drive/MyDrive/DL/Lab07 alt/cars/train"
cars_test_path = "/content/drive/MyDrive/DL/Lab07 alt/cars/test"
writer = SummaryWriter("/content/drive/MyDrive/DL/Lab07 alt/runs/CARS_WRITING")

batch_size = 8

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

model = models.resnet50(pretrained=True)

for param in model.parameters():
  param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)


epochs = 20
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
    else:
        scheduler.step()

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
%tensorboard --logdir '/content/drive/MyDrive/DL/Lab07 alt/runs/CARS_WRITING'