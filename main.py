import os
import torch
import torch.nn as nn
import multiprocessing
from ganClass import CelebDataset, Generator, Discriminator, FaceGenerator
from torch.utils.data import DataLoader
from torchvision import transforms
# --- OPTIMISATION 1 : Utilisation des Tensor Cores (Série 50 - Architecture Blackwell) ---
# Permet des calculs matriciels beaucoup plus rapides en TF32 sans perte de qualité visible.
torch.set_float32_matmul_precision('high')

# --- DEFINITION DES PARAMETRES ---
# IMAGE_SIZE : Doit correspondre à l'architecture de tes Generator/Discriminator (ici adaptés pour 64x64)
# BATCH_SIZE : Avec 16 Go de VRAM, on peut monter à 512 pour un apprentissage très stable.
# EPOCHS : 300 époques permettront d'atteindre une très haute qualité avec un DCGAN.
IMAGE_SIZE = 64
BATCH_SIZE = 512
TOTAL_EPOCHS = 20
Z_DIM = 100

# Fonction d'initialisation des poids (Spécifique pour stabiliser les DCGAN)
# Initialise les couches de Convolution avec une distribution Normale (moyenne 0, std 0.02)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# --- PROTECTION MULTIPROCESSING WINDOWS ---
# Indispensable sous Windows pour utiliser num_workers > 0 sans crash
if __name__ == '__main__':
    
    # 1. Préparation des transformations d'images
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 2. Chargement du Dataset
    dataset_path = r".\dataset\img_align_celeba\img_align_celeba"
    
    if not os.path.exists(dataset_path):
        print(f"ERREUR CRITIQUE : Le dossier {dataset_path} n'existe pas !")
        print("Vérifie le chemin de ton dataset.")
        exit()

    dataset = CelebDataset(root_dir=dataset_path, transform=transform)

    # 3. Configuration du DataLoader (Optimisé pour Ryzen)
    # num_workers : Utilise les cœurs du CPU pour pré-charger les images pendant que le GPU bosse
    # pin_memory : Accélère le transfert RAM -> VRAM
    num_workers = min(4, multiprocessing.cpu_count()) 

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True
    )

    print(f"--- Configuration ---")
    print(f"Images : {len(dataset)}")
    print(f"Batch Size : {BATCH_SIZE}")
    print(f"Workers CPU : {num_workers}")
    print(f"Image Size : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"---------------------")

    # 4. Détection du Matériel (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Détecté : {gpu_name}")
        print(f"VRAM disponible : {vram:.2f} GB")
    else:
        print("ATTENTION : Aucun GPU détecté ! L'entraînement sera extrêmement lent sur CPU.")

    # 5. Initialisation des Modèles
    generator = Generator(z_dim=Z_DIM) 
    discriminator = Discriminator()

    # Envoi sur le GPU
    generator.to(device)
    discriminator.to(device)

    # 6. Gestion de la reprise d'entraînement (Checkpoint)
    face_gan = FaceGenerator(generator, discriminator, dataloader, device)
    last_epoch = face_gan.load_latest_models()
    
    # Logique d'initialisation des poids :
    # On applique weights_init SEULEMENT si on commence de zéro (start_epoch == 0).
    # Si on charge un modèle existant, les poids chargés écrasent l'initialisation, donc pas besoin.
    start_epoch = last_epoch + 1 if last_epoch >= 0 else 0
    
    if start_epoch == 0:
        print("Initialisation des poids pour DCGAN (Normal Distribution)...")
        generator.apply(weights_init)
        discriminator.apply(weights_init)
    else:
        print(f"Reprise de l'entraînement à l'époque {start_epoch}")

    # 7. Lancement de la boucle d'entraînement
    print(f"Début de l'entraînement pour {TOTAL_EPOCHS} époques...")
    try:
        face_gan.training_face(epochs=TOTAL_EPOCHS, start_epoch=start_epoch)
        print("Entraînement terminé avec succès !")
    except KeyboardInterrupt:
        print("\nArrêt manuel de l'entraînement (Ctrl+C).")
