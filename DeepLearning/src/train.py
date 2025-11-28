import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from model import ChessNet

# --- CONFIGURATION ---
DATA_FILE = "../data/train_data.pt"
MODEL_SAVE_PATH = "../models/chess_model.pth"
BATCH_SIZE = 512       # Augmente si tu as beaucoup de VRAM
EPOCHS = 10            # Nombre de passages complets
LEARNING_RATE = 0.001

def train():
    # 1. Setup Device (GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    # 2. Chargement des données
    if not os.path.exists(DATA_FILE):
        print("Dataset introuvable. Lancez preprocess.py d'abord.")
        return

    print("Chargement du dataset en mémoire...")
    data = torch.load(DATA_FILE)
    inputs = data['inputs']
    labels = data['labels']
    
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Données chargées : {len(dataset)} positions.")

    # 3. Initialisation Modèle
    model = ChessNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Boucle d'entraînement
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()       # Reset gradients
            outputs = model(x)          # Prédiction
            loss = criterion(outputs, y)# Calcul erreur
            loss.backward()             # Backprop
            optimizer.step()            # Correction poids

            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Fin Epoch {epoch+1} : Loss Moyenne = {avg_loss:.4f} ---")
        
        # Sauvegarde intermédiaire
        os.makedirs("../models", exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Entraînement terminé !")

if __name__ == "__main__":
    train()