import torch
import torch.nn as nn
from torchvision.utils import save_image
import os 

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True), # Optimisation mémoire + gradients
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_channels * 64 * 64),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 3, 64, 64)
        return img
    
    def save_generated_images(self, epoch, batch_idx, z, output_dir="generated_images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with torch.no_grad():
            generated_images = self.forward(z).cpu()
            save_image(generated_images, os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}.png"), nrow=8, normalize=True)