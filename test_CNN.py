# === 0. Imports ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

# === 1. Préparation des données ===
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionner les images
    transforms.ToTensor(),          # Convertir en tenseur PyTorch
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation
])

# Charger les données avec ImageFolder
dataset = datasets.ImageFolder(root="data_img", transform=transform)

# Créer un DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Vérifier les classes
print("Classes:", dataset.classes)
print("Index des classes:", dataset.class_to_idx)

# === 1.b Visualisation d’un batch d’images ===
def imshow(img):
    img = img / 2 + 0.5  # dénormalisation
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

dataiter = iter(data_loader)
images, labels = next(dataiter)
imshow(utils.make_grid(images))
print("Labels:", labels)

# === 2. Définition du modèle CNN ===
class CNN_Animals(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # --- Couches convolutionnelles ---
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # --- Couches fully connected ---
        # Après 3 convolutions + 3 poolings : 128x128 → 16x16
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # aplatir
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# === 3. Initialisation du modèle ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Animals(num_classes=10).to(device)
print(model)

# === 4. Définition de la fonction de perte et de l’optimiseur ===
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 5. Fonction d'entraînement ===
def train_loop(dataloader, model, loss_fn, optimizer, log=True):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # forward
        pred = model(X)
        loss = loss_fn(pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if log and batch % 10 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / len(dataloader)
    print(f"🔹 Moyenne de la loss sur l'époque : {avg_loss:.4f}")
    return avg_loss

# === 6. Lancer l'entraînement (ex : 3 époques) ===
for epoch in range(3):
    print(f"\n===== Époque {epoch+1} =====")
    train_loop(data_loader, model, loss_fn, optimizer)
