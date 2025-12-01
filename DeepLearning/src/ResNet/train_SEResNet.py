import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import glob
import random
import gc
import matplotlib.pyplot as plt
from model_ResNet import ChessNet
import time

def format_time(seconds):
    """Convertit des secondes en format h:m:s"""
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"

# --- CONFIGURATION AVANCÉE ---
DATA_DIR = "../../data/processed"
MODEL_SAVE_PATH = "../../models/chess_model_seresnet.pth" # Nouveau nom pour ne pas écraser l'ancien
GRAPH_SAVE_PATH = "../../models/training_loss_seresnet.png"

# Hyperparamètres Modèle
NUM_RES_BLOCKS = 10      # Nombre de blocs (Essaie 10 pour commencer, puis 20)
NUM_CHANNELS = 128       # Largeur du réseau (64, 128, ou 256)
USE_SE = True            # Activer Squeeze-and-Excitation ?

# Hyperparamètres Entraînement
BATCH_SIZE = 512         # Baisse à 256 si tu as une erreur "Out of Memory" GPU
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 3

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Configuration ResNet ---")
    print(f"Device: {device}")
    print(f"Blocs: {NUM_RES_BLOCKS} | Canaux: {NUM_CHANNELS} | SE: {USE_SE}")
    print(f"----------------------------")

    # 1. Repérage des fichiers
    train_files = glob.glob(os.path.join(DATA_DIR, "train_part_*.pt"))
    val_files = glob.glob(os.path.join(DATA_DIR, "val_part_*.pt"))
    
    if not train_files:
        print(f"Erreur : Aucun fichier trouvé dans {DATA_DIR}. Lancez preprocess.py.")
        return

    # 2. Initialisation du Nouveau Modèle
    model = ChessNet(num_res_blocks=NUM_RES_BLOCKS, 
                     num_channels=NUM_CHANNELS, 
                     use_se=USE_SE).to(device)
                     
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print("Début de l'entraînement...")

    global_start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")

        epoch_start_time = time.time()
        
        # --- TRAIN ---
        model.train()
        total_train_loss = 0.0
        total_batches = 0
        
        random.shuffle(train_files) # Mélange les fichiers

        for i, f_path in enumerate(train_files):
            print(f"   [Train] Fichier {i+1}/{len(train_files)} : {os.path.basename(f_path)}")
            
            try:
                data = torch.load(f_path)
                dataset = TensorDataset(data['inputs'], data['labels_move'])
                
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
                
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
                
                total_batches += len(loader)
                
                # Nettoyage RAM
                del data, dataset, loader, x, y, outputs, loss
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Erreur lecture fichier {f_path}: {e}")

        avg_train_loss = total_train_loss / max(1, total_batches)

        # --- VALIDATION ---
        model.eval()
        total_val_loss = 0.0
        total_val_batches = 0
        correct_preds = 0
        total_samples = 0

        print("   [Val] Evaluation...")
        with torch.no_grad():
            for f_path in val_files:
                data = torch.load(f_path)
                dataset = TensorDataset(data['inputs'], data['labels_move'])
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
                
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    total_val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total_samples += y.size(0)
                    correct_preds += (predicted == y).sum().item()
                
                total_val_batches += len(loader)
                del data, dataset, loader, x, y, outputs
                gc.collect()

        avg_val_loss = total_val_loss / max(1, total_val_batches)
        accuracy = 100 * correct_preds / max(1, total_samples)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"   >>> Fin Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Sauvegarde
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   >>> 🏆 Modèle sauvegardé !")
        else:
            patience_counter += 1
            print(f"   >>> Pas d'amélioration ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("⛔ Early Stopping.")
                break
        
        epoch_duration = time.time() - epoch_start_time
        print(f"   ⏱️ Durée Epoch : {format_time(epoch_duration)}")
    
    global_duration = time.time() - global_start_time
    print(f"\n✅ Entraînement terminé en : {format_time(global_duration)}")

    # Graphique
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val', linestyle='--')
    plt.title('Loss ResNet')
    plt.legend()
    plt.savefig(GRAPH_SAVE_PATH)
    print("Terminé.")

if __name__ == "__main__":
    train()