import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
from model import ChessNet

# --- CONFIGURATION ---
TRAIN_FILE = "../data/train_data.pt"
VAL_FILE = "../data/val_data.pt"
MODEL_SAVE_PATH = "../models/chess_model.pth"
GRAPH_SAVE_PATH = "../models/training_loss.png"

BATCH_SIZE = 512
EPOCHS = 50        # On met beaucoup d'époques, l'Early Stopping arrêtera avant !
LEARNING_RATE = 0.001
PATIENCE = 3       # Arrêter si pas d'amélioration pendant 5 époques

def load_dataset(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} introuvable.")
    data = torch.load(filepath)
    return TensorDataset(data['inputs'], data['labels'])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    # 1. Chargement
    print("Chargement Train & Val...")
    try:
        train_dataset = load_dataset(TRAIN_FILE)
        val_dataset = load_dataset(VAL_FILE)
    except FileNotFoundError:
        print("Erreur: Lancez d'abord preprocess.py !")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ChessNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Historique
    history = {'train_loss': [], 'val_loss': []}
    
    # Variables pour Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Début de l'entraînement (Max Epochs: {EPOCHS}, Patience: {PATIENCE})...")
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
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
        
        # --- VALIDATION ---
        model.eval()
        val_running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_preds += y.size(0)
                correct_preds += (predicted == y).sum().item()

        avg_val_loss = val_running_loss / len(val_loader)
        accuracy = 100 * correct_preds / total_preds

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.2f}%")

        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_val_loss:
            # C'est un record ! On sauvegarde ce modèle précis.
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   >>> Nouveau record ! Modèle sauvegardé.")
        else:
            # Pas d'amélioration
            patience_counter += 1
            print(f"   >>> Pas d'amélioration ({patience_counter}/{PATIENCE})")
            
            if patience_counter >= PATIENCE:
                print(f"⛔ ARRÊT PRÉCOCE (Early Stopping) déclenché à l'époque {epoch+1}.")
                print("Le modèle commençait à surapprendre (Overfitting).")
                break

    # --- GRAPHIQUE ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss', linestyle='--')
    
    # Marquer le point d'arrêt idéal
    best_epoch = len(history['val_loss']) - patience_counter - 1
    if patience_counter >= PATIENCE: # Si on s'est arrêté tôt
         plt.axvline(x=best_epoch, color='g', linestyle=':', label='Meilleur Modèle')

    plt.title('Courbe d\'apprentissage (Loss)')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(GRAPH_SAVE_PATH)
    print(f"Graphique sauvegardé : {GRAPH_SAVE_PATH}")
    print("Entraînement terminé !")

if __name__ == "__main__":
    train()