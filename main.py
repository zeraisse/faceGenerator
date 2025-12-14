import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from ganClass.CelebDataset import CelebDataset
from ganClass.Generator import Generator
from ganClass.Discriminator import Discriminator
from ganClass.FaceGenerator import FaceGenerator

# 🚀 OPTIMISATION 1 : Active l'auto-tuner de Nvidia
# La carte va tester plusieurs algos de convolution au début pour trouver le plus rapide
torch.backends.cudnn.benchmark = True

def main():
    # --- 1. CONFIGURATION MATÉRIELLE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n" + "="*50)
    print(f"🔥 PROCESSEUR : {device}")
    if device.type == 'cuda':
        print(f"🎮 CARTE GRAPHIQUE : {torch.cuda.get_device_name(0)}")
        print(f"⚡ ARCHITECTURE : {torch.cuda.get_device_capability(0)}")
    print("="*50 + "\n")

    # --- 2. HYPERPARAMÈTRES (Mode "Force Brute") ---
    lr = 0.0002
    batch_size = 256 
    image_size = 64
    z_dim = 100
    epochs = 20

    # --- 3. PRÉPARATION DES DONNÉES ---
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_path = r".\dataset\img_align_celeba\img_align_celeba"
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERREUR : Le dossier {dataset_path} n'existe pas !")
        return

    print("📂 Chargement du Dataset...")
    dataset = CelebDataset(root_dir=dataset_path, transform=transform)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        # num_workers=4 permet au CPU de préparer les images en avance pendant que le GPU bosse
        num_workers=4,   
        # pin_memory=True accélère le transfert RAM -> VRAM
        pin_memory=True  
    )
    print(f"Dataset chargé : {len(dataset)} images.")

    # --- 4. INITIALISATION DES MODÈLES ---
    generator = Generator(z_dim).to(device)
    # Note : Le discriminateur ne doit plus avoir de Sigmoid à la fin (voir Discriminator.py modifié)
    discriminator = Discriminator().to(device)

    # Création du manager d'entraînement
    face_gan = FaceGenerator(generator, discriminator, dataloader, device)
    
    print("Vérification des sauvegardes...")
    last_epoch = face_gan.load_latest_models()
    
    if last_epoch != -1:
        start_epoch = last_epoch + 1
        print(f"⏩ Reprise de l'entraînement à l'époque {start_epoch}")
    else:
        start_epoch = 0
        print(f"✨ Démarrage d'un nouvel entraînement")

    # --- 6. LANCEMENT DU MOTEUR ---
    face_gan.training_face(epochs=epochs, start_epoch=start_epoch)
    print("\nEntraînement terminé avec succès !")

if __name__ == "__main__":
    main()