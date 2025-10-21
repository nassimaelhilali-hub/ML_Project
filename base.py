# pip install virtualenv       : installer virtualenv
# virtualenv nomEnvironnemnent : créer un environnement virtuel 
# venv/Scripts/activate        : activer l'environnement créer 


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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
