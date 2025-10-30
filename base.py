# pip install virtualenv       : installer virtualenv
# virtualenv nomEnvironnemnent : créer un environnement virtuel 
# venv/Scripts/activate        : activer l'environnement créer 

# python base.py
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict

# Transformations pour normaliser les images et les redimensionner
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionner les images
    transforms.ToTensor(),          # Convertir en tenseur PyTorch
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Normalisation
                         std=[0.5, 0.5, 0.5])
])

# Charger les données avec ImageFolder
dataset = datasets.ImageFolder(root="data_img", transform=transform)



# Créer un DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Obtenir la correspondance des classes
print("Classes:", dataset.classes)
print("Index des classes:", dataset.class_to_idx)


# ========================================================


# Obtenir la correspondance des classes
print("Classes:", dataset.classes)
print("Index des classes:", dataset.class_to_idx)

# Initialiser un dictionnaire pour stocker les images et labels
class_images = defaultdict(list)
class_labels = defaultdict(list)

# Limiter à 100 images par classe
max_images_per_class = 10

# Parcourir les lots d'images
for images, labels in data_loader:
    for image, label in zip(images, labels):
        class_images[label.item()].append(image)
        class_labels[label.item()].append(label.item())

        # Arrêter dès qu'on a 100 images pour chaque classe
        if all(len(class_images[class_idx]) >= max_images_per_class for class_idx in class_images):
            break
    else:
        continue  # Si break n'a pas été appelé, continuer à parcourir le batch
    break  # Sortir de la boucle principale quand on a atteint 100 images pour chaque classe

# Limiter à 10 images pour chaque classe
for class_idx in class_images:
    class_images[class_idx] = class_images[class_idx][:max_images_per_class]
    class_labels[class_idx] = class_labels[class_idx][:max_images_per_class]

# Afficher la forme des images et labels pour vérifier
print(f"Nombre d'images par classe :")
for class_idx, images in class_images.items():
    print(f"Classe {dataset.classes[class_idx]} : {len(images)} images")




# ========================================================




# Exemple : visualiser un batch
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5  # dénormalisation
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Obtenir un batch
dataiter = iter(data_loader)
images, labels = next(dataiter)

# Montrer les images
imshow(torchvision.utils.make_grid(images))
print('Labels:', labels)

# ================================================================

import os
from translate import translate

root = "data_img"

for folder in os.listdir(root):
    old_path = os.path.join(root, folder)
    if not os.path.isdir(old_path):
        continue

    # Traduction vers l'anglais (si possible)
    new_name = translate.get(folder)
    if new_name:
        new_path = os.path.join(root, new_name)
        
        # Évite de renommer si le nom est identique ou déjà existant
        if not os.path.exists(new_path) and new_name != folder:
            os.rename(old_path, new_path)
            print(f"Renamed: {folder} → {new_name}")
        else:
            print(f" Skipped: {folder} → {new_name} (already exists or same name)")
    else:
        print(f" No translation found for: {folder}")

