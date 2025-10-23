import os
import torch
import torch.nn as nn
import re


class FaceGenerator:
    def __init__(self, generator, discriminator, dataloader, device):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.device = device
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(64, 100).to(self.device)
        self.g_losses = []
        self.d_losses = []


    def training_face(self, epochs=10, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            d_loss_epoch = 0.0
            g_loss_epoch = 0.0
            num_batches = 0

            for i, imgs in enumerate(self.dataloader):
                real_imgs = imgs.to(self.device)
                batch_size = real_imgs.size(0)
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)

                random_noise = torch.randn(batch_size, 100, device=self.device)
                fake_imgs = self.generator(random_noise)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_loss = nn.BCELoss()(self.discriminator(real_imgs), valid)
                fake_loss = nn.BCELoss()(self.discriminator(fake_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                # Train Generator
                self.optimizer_G.zero_grad()
                g_loss = nn.BCELoss()(self.discriminator(fake_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()

                d_loss_epoch += d_loss.item()
                g_loss_epoch += g_loss.item()
                num_batches += 1

                if i % 50 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            avg_d_loss = d_loss_epoch / num_batches
            avg_g_loss = g_loss_epoch / num_batches
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)

            self.generator.save_generated_images(epoch, i, self.fixed_noise)
            # Save model after each epoch for backup
            self.save_models(epoch)
            self.plot_losses()
            
    def save_models(self, epoch, output_dir="saved_models"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.generator.state_dict(), os.path.join(output_dir, f"generator_epoch_{epoch}.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(output_dir, f"discriminator_epoch_{epoch}.pth"))
        print(f"Modèles sauvegardés dans {output_dir} à l'époque {epoch}")
    

    def load_latest_models(self, output_dir="saved_models"):
        """
        Charge les poids les plus récents du générateur et du discriminateur.
        Retourne le numéro de la dernière époque sauvegardée, ou -1 si aucun modèle trouvé.
        """
        if not os.path.exists(output_dir):
            print(f"Dossier {output_dir} non trouvé. Entraînement depuis zéro.")
            return -1

        # Liste tous les fichiers generator_epoch_*.pth
        gen_files = [f for f in os.listdir(output_dir) if f.startswith("generator_epoch_") and f.endswith(".pth")]
        if not gen_files:
            print("Aucun modèle sauvegardé trouvé. Entraînement depuis zéro.")
            return -1

        # Extrait les numéros d'époque
        epochs = []
        for f in gen_files:
            match = re.search(r'epoch_(\d+)\.pth$', f)
            if match:
                epochs.append(int(match.group(1)))

        if not epochs:
            print("Impossible de parser les noms de fichiers. Entraînement depuis zéro.")
            return -1

        last_epoch = max(epochs)
        gen_path = os.path.join(output_dir, f"generator_epoch_{last_epoch}.pth")
        disc_path = os.path.join(output_dir, f"discriminator_epoch_{last_epoch}.pth")

        # Charger les poids
        self.generator.load_state_dict(torch.load(gen_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))
        print(f"Modèles chargés à partir de l'époque {last_epoch}")
        return last_epoch


    def plot_losses(self, output_dir="training_plots"):
        import matplotlib.pyplot as plt  

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(10, 6))
        plt.plot(self.d_losses, label='Discriminator Loss', marker='o')
        plt.plot(self.g_losses, label='Generator Loss', marker='o')
        plt.title('GAN Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "losses.png")
        plt.savefig(plot_path)
        print(f"Courbe des pertes sauvegardée dans {plot_path}")
        plt.close()


                