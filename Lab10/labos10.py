from google.colab import drive
drive.mount('/content/drive')

!pip install pkbar

# SEMI-SUPERVISED GAN

CHANNELS = 3
NUM_CLASSES = 10
LATENT_DIM = 100
IMG_SIZE = 32
SAMPLE_INTERVAL = 400

BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 200

import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM, img_size=IMG_SIZE, channels=CHANNELS):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size=IMG_SIZE, channels=CHANNELS, num_classes=NUM_CLASSES):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, num_classes + 1), nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

# Commented out IPython magic to ensure Python compatibility.
from torchvision.transforms.transforms import Resize
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time, sleep

from torchvision.utils import save_image
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from PIL import Image
import pkbar
from torch.utils.tensorboard import SummaryWriter

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

cuda = True if torch.cuda.is_available() else False
print("device is cuda ->", cuda)

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.CIFAR10('cifar_data', download=True, train=True, transform=transforms)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

for epoch in range(EPOCHS):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake_aux_gt = Variable(LongTensor(batch_size).fill_(NUM_CLASSES), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, LATENT_DIM))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
#             % (epoch, 200, i, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % SAMPLE_INTERVAL == 0:
            save_image(gen_imgs.data[:25], "/content/drive/MyDrive/DL/Lab10/images/%d.png" % batches_done, nrow=5, normalize=True)

    else:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict()
        }, '/content/drive/MyDrive/DL/Lab10/checkpoints/chk' + str(epoch) + '.pth')

model_to_load = "/content/drive/MyDrive/DL/Lab10/checkpoints/$$$$$$$$"

cuda = True if torch.cuda.is_available() else False

generator = Generator()

if cuda:
  device = torch.device('cuda')
  generator = generator.cuda()

checkpoint = torch.load(model_to_load)
generator.load_state_dict(checkpoint['generator_state_dict'])

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

with torch.no_grad():
  for i in range(10):
    z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, LATENT_DIM))))
    gen_imgs = generator(z)
    save_image(gen_imgs.data, "/content/drive/MyDrive/DL/Lab10/images/generated%d.png" % i, nrow=23, normalize=True)