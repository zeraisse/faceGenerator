import torch
import torch.nn as nn
class FaceGenerator:
    def __init__(self, generator, discriminator, dataloader, device):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.device = device
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.fixed_noise = torch.randn(64, 100).to(self.device)


    def training_face(self, epochs=5):
        for epoch in range(epochs):
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

                if i % 50 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            self.generator.save_generated_images(epoch, i, self.fixed_noise)
    

                