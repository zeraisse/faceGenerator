import torch
from ganClass import CelebDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64)),
    transforms.CenterCrop((64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_path = "./dataset/celeba"
dataset = CelebDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
print(f"Number of images in dataset: {len(dataset)}")