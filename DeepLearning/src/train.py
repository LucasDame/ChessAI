import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt # Pour les graphiques
from model import ChessNet

# --- CONFIGURATION ---
TRAIN_FILE = "../data/train_data.pt"
VAL_FILE = "../data/val_data.pt"
MODEL_SAVE_PATH = "../models/chess_model.pth"
GRAPH_SAVE_PATH = "../models/training_loss.png"

BATCH_SIZE = 512
EPOCHS = 20        # Un peu plus d'époques pour voir la courbe
LEARNING_RATE = 0.001

def load_dataset(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} introuvable.")
    data = torch.load(filepath)
    return TensorDataset(data['inputs'], data['labels'])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    # 1. Chargement des DEUX datasets
    print("Chargement Train & Val...")
    train_dataset = load_dataset(TRAIN_FILE)
    val_dataset = load_dataset(VAL_FILE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ChessNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Historique pour le graphique
    history = {'train_loss': [], 'val_loss': []}

    print("Début de l'entraînement...")
    
    for epoch in range(EPOCHS):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # --- PHASE DE VALIDATION (TEST) ---
        model.eval() # Désactive Dropout & BatchNorm stats update
        val_running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad(): # Pas de gradient ici (économise mémoire/temps)
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_running_loss += loss.item()
                
                # Calcul de précision (optionnel mais utile)
                _, predicted = torch.max(outputs, 1)
                total_preds += y.size(0)
                correct_preds += (predicted == y).sum().item()

        avg_val_loss = val_running_loss / len(val_loader)
        accuracy = 100 * correct_preds / total_preds

        # Stockage
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}%")
        
        # Sauvegarde du modèle à chaque époque
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # --- GÉNÉRATION DU GRAPHIQUE ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Courbe d\'apprentissage (Loss)')
    plt.xlabel('Époques')
    plt.ylabel('Loss (Entropie Croisée)')
    plt.legend()
    plt.grid(True)
    
    os.makedirs("../models", exist_ok=True)
    plt.savefig(GRAPH_SAVE_PATH)
    print(f"Graphique sauvegardé : {GRAPH_SAVE_PATH}")
    print("Entraînement terminé !")

if __name__ == "__main__":
    train()