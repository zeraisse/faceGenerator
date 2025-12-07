import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, features_g=64):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Entrée : Bruit Z (z_dim x 1 x 1)
            self._block(z_dim, features_g * 16, 4, 1, 0), # Sortie : 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1), # Sortie : 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1), # Sortie : 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # Sortie : 32x32
            
            # Couche finale pour arriver à 64x64
            nn.ConvTranspose2d(features_g * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Pour ramener les pixels entre -1 et 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels), # Stabilise l'apprentissage
            nn.ReLU(True)
        )

    def forward(self, z):
        # On redimensionne le bruit pour qu'il soit compatible avec ConvTranspose2d
        # z est [Batch, 100] -> [Batch, 100, 1, 1]
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)

    def save_generated_images(self, epoch, batch_idx, z, output_dir="generated_images"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with torch.no_grad():
            # On adapte aussi le Z ici pour la sauvegarde
            z_view = z.view(z.size(0), z.size(1), 1, 1)
            generated_images = self.model(z_view).cpu()
            save_image(generated_images, os.path.join(output_dir, f"epoch_{epoch}_batch_{batch_idx}.png"), nrow=8, normalize=True)