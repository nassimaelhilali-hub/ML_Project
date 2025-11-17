###========================================================================
# Importations et Configuration
###========================================================================

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Configuration de l'appareil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de l'appareil: {device}")


###========================================================================
# Transformation de Base (Normalisation)
###========================================================================

# Transformation de base : Redimensionnement, ToTensor, Normalisation
BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Charger les données avec ImageFolder et la transformation de base
dataset = datasets.ImageFolder(root="data_img", transform=BASE_TRANSFORM)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

print("Classes:", dataset.classes)
print("Index des classes:", dataset.class_to_idx)


###========================================================================
# Sélection d'un Échantillon (Sous-échantillonnage)
###========================================================================

# Initialiser un dictionnaire pour stocker les images et labels
class_images = defaultdict(list)
class_labels = defaultdict(list)

# Limite
max_images_per_class = 10 

print(f"Collecte de {max_images_per_class} images par classe...")

# Parcourir les lots d'images et collecter les tenseurs
for images, labels in data_loader:
    for image, label in zip(images, labels):
        class_images[label.item()].append(image)
        class_labels[label.item()].append(label.item())

        # Condition d'arrêt
        if all(len(class_images[class_idx]) >= max_images_per_class for class_idx in dataset.class_to_idx):
            break
    else:
        continue
    break


# Vérification finale du nombre d'images
for class_idx in class_images:
    class_images[class_idx] = class_images[class_idx][:max_images_per_class]
    class_labels[class_idx] = class_labels[class_idx][:max_images_per_class]

print("Nombre d'images par classe après sous-échantillonnage :")
for class_idx, images in class_images.items():
    print(f"Classe {dataset.classes[class_idx]} : {len(images)} images")


# Définition des Datasets pour l'Augmentation
###========================================================================
# Classes de Dataset pour la Manipulation de Tenseurs
###========================================================================

class SubsetTensorDataset(Dataset):
    """Encapsule les tenseurs d'images collectés et normalisés."""
    def __init__(self, class_images, class_labels):
        self.images = []
        self.labels = []
        for label, images in class_images.items():
            self.images.extend(images)
            self.labels.extend([label] * len(images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class AugmentedSubset(Dataset):
    """Applique une transformation spécifique (e.g., RandomErasing) à un Subset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            # Application de la transformation (Random Erasing) sur le tenseur
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


###========================================================================
# Division et Application de Random Erasing
###========================================================================

# Définition de Random Erasing (Suppression de zone)
# Il ne fait que la suppression de zone, car les autres étapes sont déjà faites.
RANDOM_ERASING_TRANSFORM = transforms.RandomErasing(
    p=0.5,           # 50% de chance d'appliquer la suppression
    scale=(0.02, 0.33), # Taille de la zone
    ratio=(0.3, 3.3),   # Ratio H/L
    value='random'   # Remplir la zone avec des valeurs aléatoires
)

# Créer le dataset de base avec les tenseurs collectés
base_dataset = SubsetTensorDataset(class_images, class_labels)

# Division en Train (80%) et Validation (20%)
train_size = int(0.8 * len(base_dataset))
valid_size = len(base_dataset) - train_size
train_indices, valid_indices = random_split(base_dataset, [train_size, valid_size])

# Création des datasets finaux
# train_dataset reçoit RandomErasing
train_dataset = AugmentedSubset(train_indices, transform=RANDOM_ERASING_TRANSFORM)
# valid_dataset ne reçoit aucune transformation additionnelle
valid_dataset = AugmentedSubset(valid_indices, transform=None)

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False) # Pas besoin de mélanger la validation

print(f"Taille du jeu d'entraînement (avec Random Erasing): {len(train_dataset)}")
print(f"Taille du jeu de validation: {len(valid_dataset)}")


###========================================================================
# Visualisation de l'Augmentation
###========================================================================

def imshow(img, title=None):
    """Dé-normalise et affiche une image (tenseur PyTorch)"""
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    
    # Dé-normalisation
    img = img.numpy().transpose((1, 2, 0)) # C, H, W -> H, W, C
    img = std * img + mean
    img = np.clip(img, 0, 1) 
    
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')

# --- Visualisation ---
image_index = 0 # Première image du train_dataset
image_aug, label_idx = train_dataset[image_index] 
image_orig, _ = valid_dataset[image_index] # L'image de base (sans augmentation)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

plt.sca(axes[0])
imshow(image_orig.cpu(), title=f"Original - Classe {label_idx}")

plt.sca(axes[1])
imshow(image_aug.cpu(), title=f"Augmentée (Random Erasing) - Classe {label_idx}")

plt.tight_layout()
plt.show()


# Modèle CNN et Fonctions d'Entraînement/Test
###========================================================================
# Définition du Modèle 
###========================================================================

class CNN_Animals(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Taille d'entrée pour la FC: 128x128 -> (pool x3) -> 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

model = CNN_Animals(num_classes=len(dataset.classes)).to(device)
print(model)


###========================================================================
# Boucles d'Entraînement et de Test
###========================================================================

def train_loop(loader, model, loss_fn, optimizer, log=True):
    model.train()
    total_loss = 0
    size = len(loader.dataset)

    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # Forward
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if log and batch % 10 == 0:
            current = (batch + 1) * X.size(0)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / len(loader)
    print(f"🔹 Moyenne de la loss sur l'époque : {avg_loss:.4f}")
    return avg_loss

def test_loop(loader, model, loss_fn, log=True):
    model.eval()
    size = len(loader.dataset)
    num_batches = len(loader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    if log:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, loss: {test_loss:>8f} \n")
    return test_loss, 100*correct

###========================================================================
# Entraînement Final et Tracé des Résultats
###========================================================================

epochs = 50 
loss_test = []
loss_train = []
acc_test = []

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for t in range(epochs):
    print(f"\n===== Époque {t+1} =====")
    # Entraînement avec data augmentation
    loss_train.append(train_loop(train_loader, model, loss_fn, optimizer))
    # Évaluation sans data augmentation
    l, a = test_loop(valid_loader, model, loss_fn)
    loss_test.append(l)
    acc_test.append(a)
print("Done!")

# Tracé des courbes
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_train, label='Train Loss')
plt.plot(loss_test, label='Validation Loss')
plt.title('Courbe de Perte')
plt.xlabel('Époque')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc_test, label='Validation Accuracy', color='green')
plt.title("Courbe de Précision")
plt.xlabel('Époque')
plt.ylabel('Précision (%)')
plt.legend()
plt.show()



###========================================================================