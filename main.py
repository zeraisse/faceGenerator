import os
import torch
from ganClass import CelebDataset, Generator, Discriminator
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn


# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64)),
    transforms.CenterCrop((64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_path = "./dataset/celeba"
dataset = CelebDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print(f"Number of images in dataset: {len(dataset)}")

## Loss function
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(z_dim=100)
discriminator = Discriminator()

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## log to chek if GPU is detected
if torch.cuda.is_available():
    print("Nom du GPU :", torch.cuda.get_device_name(0))

generator.to(device)
discriminator.to(device)