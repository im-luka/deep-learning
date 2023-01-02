import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

def show_dataset(data_loader):
  for data in data_loader:
    images, labels = data  

    grid = torchvision.utils.make_grid(images, nrow=5)
    plt.figure(figsize=(15,10))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.axis("off")
    plt.show()

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((375,500)),
    torchvision.transforms.ColorJitter(hue=.1, saturation=.75),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(7),
    torchvision.transforms.RandomGrayscale(),
    torchvision.transforms.ToTensor(),
])

dataset = torchvision.datasets.ImageFolder("models_train", transform=transforms)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=120, shuffle=True)

show_dataset(data_loader)
