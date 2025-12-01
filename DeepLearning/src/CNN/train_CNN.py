import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import glob
import random
import gc # Garbage Collector pour forcer le nettoyage de la RAM
import matplotlib.pyplot as plt
from model_CNN import ChessNet

# --- CONFIGURATION ---
DATA_DIR = "../../data/processed"
MODEL_SAVE_PATH = "../../models/chess_model.pth"
GRAPH_SAVE_PATH = "../../models/training_loss.png"

BATCH_SIZE = 512
EPOCHS = 20 
LEARNING_RATE = 0.001
PATIENCE = 3 # On réduit un peu la patience car 1 époque est très longue maintenant

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")

    # 1. Repérage des fichiers
    train_files = glob.glob(os.path.join(DATA_DIR, "train_part_*.pt"))
    val_files = glob.glob(os.path.join(DATA_DIR, "val_part_*.pt"))
    
    if not train_files:
        print("Erreur : Aucun fichier de données trouvé.")
        return

    print(f"Dataset : {len(train_files)} fichiers d'entraînement détectés.")

    # 2. Initialisation
    model = ChessNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print("Début du Streaming...")

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        
        # --- PHASE TRAIN (Fichier par Fichier) ---
        model.train()
        total_train_loss = 0.0
        total_batches = 0
        
        # On mélange l'ordre des fichiers à chaque époque pour varier
        random.shuffle(train_files)

        for i, f_path in enumerate(train_files):
            print(f"   [Train] Chargement fichier {i+1}/{len(train_files)} : {os.path.basename(f_path)}")
            
            # Chargement local (RAM monte)
            data = torch.load(f_path)
            dataset = TensorDataset(data['inputs'], data['labels_move'])

            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # Entraînement sur ce morceau
            file_loss = 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                file_loss += loss.item()
            
            total_train_loss += file_loss
            total_batches += len(loader)

            # Nettoyage immédiat (RAM descend)
            del data, dataset, loader, x, y, outputs, loss
            gc.collect() 
            torch.cuda.empty_cache() # Vide le cache GPU aussi

        avg_train_loss = total_train_loss / total_batches

        # --- PHASE VALIDATION (Fichier par Fichier) ---
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
                
                # Nettoyage
                del data, dataset, loader, x, y, outputs
                gc.collect()

        avg_val_loss = total_val_loss / total_val_batches
        accuracy = 100 * correct_preds / total_samples

        # --- LOGS & SAVE ---
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        print(f"   >>> Fin Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            os.makedirs("../models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   >>> Modèle sauvegardé (Nouveau Record) 🏆")
        else:
            patience_counter += 1
            print(f"   >>> Pas d'amélioration ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("⛔ Early Stopping.")
                break

    # Graphique
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val', linestyle='--')
    plt.title('Loss')
    plt.legend()
    plt.savefig(GRAPH_SAVE_PATH)
    print("Terminé.")

if __name__ == "__main__":
    train()