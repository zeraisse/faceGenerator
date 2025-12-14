import os
import torch
import torch.nn as nn
import re
import matplotlib.pyplot as plt

class FaceGenerator:
    def __init__(self, generator, discriminator, dataloader, device):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.device = device
        
        # Optimiseurs
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Scaler pour le Mixed Precision (RTX 5070 Ti Boost)
        self.scaler = torch.amp.GradScaler("cuda")
        
        self.fixed_noise = torch.randn(64, 100).to(self.device)
        self.g_losses = []
        self.d_losses = []

    def training_face(self, epochs=10, start_epoch=0):
        print(f"Entraînement Optimisé RTX 5070 Ti (BCEWithLogitsLoss)")
        
        # On remplace BCELoss par BCEWithLogitsLoss pour supporter le Mixed Precision
        criterion = nn.BCEWithLogitsLoss() 
        
        for epoch in range(start_epoch, epochs):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            num_batches = 0

            for i, imgs in enumerate(self.dataloader):
                # Transfert asynchrone vers le GPU (plus rapide)
                real_imgs = imgs.to(self.device, non_blocking=True)
                batch_size = real_imgs.size(0)

                # Labels lissés pour meilleure stabilité (Label Smoothing)
                valid = torch.tensor(0.9, device=self.device).expand(batch_size, 1)
                fake = torch.zeros(batch_size, 1, device=self.device)
                
                # --- TRAIN DISCRIMINATOR ---
                self.optimizer_D.zero_grad(set_to_none=True) # Opti mémoire

                with torch.amp.autocast("cuda"): # Active le Mixed Precision
                    # Loss Real
                    output_real = self.discriminator(real_imgs)
                    real_loss = criterion(output_real, valid)
                    
                    # Loss Fake
                    z = torch.randn(batch_size, 100, device=self.device)
                    fake_imgs = self.generator(z)
                    output_fake = self.discriminator(fake_imgs.detach())
                    fake_loss = criterion(output_fake, fake)
                    
                    d_loss = (real_loss + fake_loss) / 2

                # Backward avec Scaler
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.optimizer_D)
                
                # --- TRAIN GENERATOR ---
                self.optimizer_G.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda"):
                    # On veut tromper le discriminateur
                    output_fake_for_G = self.discriminator(fake_imgs)
                    g_loss = criterion(output_fake_for_G, valid)

                self.scaler.scale(g_loss).backward()
                self.scaler.step(self.optimizer_G)
                
                # Mise à jour du scaler
                self.scaler.update()

                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()
                num_batches += 1

                # Log moins fréquent pour ne pas ralentir la console
                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

            avg_d_loss = d_loss_epoch / num_batches
            avg_g_loss = g_loss_epoch / num_batches
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)

            # Sauvegardes
            self.generator.save_generated_images(epoch, i, self.fixed_noise)
            self.save_models(epoch)
            self.plot_losses()

    def save_models(self, epoch, output_dir="saved_models"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.generator.state_dict(), os.path.join(output_dir, f"generator_epoch_{epoch}.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(output_dir, f"discriminator_epoch_{epoch}.pth"))
        print(f"Sauvegarde Epoch {epoch}")

    def load_latest_models(self, output_dir="saved_models"):
        if not os.path.exists(output_dir): return -1
        gen_files = [f for f in os.listdir(output_dir) if f.startswith("generator_epoch_") and f.endswith(".pth")]
        if not gen_files: return -1
        
        epochs = []
        for f in gen_files:
            match = re.search(r'epoch_(\d+)\.pth$', f)
            if match: epochs.append(int(match.group(1)))
        
        if not epochs: return -1
        
        last_epoch = max(epochs)
        gen_path = os.path.join(output_dir, f"generator_epoch_{last_epoch}.pth")
        disc_path = os.path.join(output_dir, f"discriminator_epoch_{last_epoch}.pth")
        
        self.generator.load_state_dict(torch.load(gen_path, map_location=self.device, weights_only=True))
        self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device, weights_only=True))
        print(f"Reprise Epoch {last_epoch}")
        return last_epoch

    def plot_losses(self, output_dir="training_plots"):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator')
        plt.plot(self.g_losses, label='Generator')
        plt.title('Training Losses')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "losses.png"))
        plt.close()