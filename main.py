import os
import torch
from ganClass import CelebDataset, Generator, Discriminator, FaceGenerator
from torch.utils.data import DataLoader
from torchvision import transforms


# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64)),
    transforms.CenterCrop((64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset_path = ".\dataset\img_align_celeba\img_align_celeba"
dataset = CelebDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
print(f"Number of images in dataset: {len(dataset)}")

# Initialize generator and discriminator
generator = Generator(z_dim=100)
discriminator = Discriminator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## log to chek if GPU is detected
# print("PyTorch version :", torch.__version__)
# print("CUDA version compilée avec PyTorch :", torch.version.cuda)
if torch.cuda.is_available():
    print("Nom du GPU :", torch.cuda.get_device_name(0))
else:
    print("Aucun GPU détecté, utilisation du CPU.")

generator.to(device)
discriminator.to(device)
face_gan = FaceGenerator(generator, discriminator, dataloader, device)
face_gan.training_face()