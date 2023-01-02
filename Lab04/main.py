from google.colab import drive
drive.mount('/content/drive')

!pip install pkbar

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

emnist_data_path = "/content/drive/MyDrive/DL/Lab04/emnist_data"
writer = SummaryWriter("/content/drive/MyDrive/DL/Lab04/runs/EMNIST")
  
batch_size = 128

transforms = torchvision.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.EMNIST(root=emnist_data_path, split='letters', download=True, train=True, transform=transforms)
test_dataset = datasets.EMNIST(root=emnist_data_path, split='letters', download=True, train=False, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)

input_size = 784
hidden_size_1 = 128
hidden_size_2 = 64
hidden_size_3 = 32
output_size = 27

model = nn.Sequential(nn.Linear(input_size, hidden_size_1),
                      nn.ReLU(),
                      nn.Linear(hidden_size_1, hidden_size_2),
                      nn.ReLU(),
                      nn.Linear(hidden_size_2, hidden_size_3),
                      nn.ReLU(),
                      nn.Linear(hidden_size_3, output_size),
                      nn.LogSoftmax(dim=1)).to(device)

loss_fn = nn.NLLLoss().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

epochs = 15
train_per_epoch = int(len(train_dataset) / batch_size)

for e in range(epochs):
    kbar = pkbar.Kbar(target=train_per_epoch, epoch=e, num_epochs=epochs, width=20, always_stateful=False)
    for idx, (images, labels) in enumerate(train_loader):

        images = images.to(device).view(images.shape[0], -1)
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


correct_count, all_count = 0, 0
for images, labels in test_loader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img.to(device))

        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu().numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number of images: ", all_count)
print("Test accuracy: ", (correct_count / all_count))