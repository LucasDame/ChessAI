import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import glob
import random
import gc
import matplotlib.pyplot as plt
from evaluator_nn import load_evaluator, Evaluator
import time

def format_time(seconds):
    """Convertit des secondes en format h:m:s"""
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"

# --- CONFIGURATION ---
DATA_DIR = "../../data/processed"
MODEL_SAVE_PATH = "../../models/chess_model_evaluator.pth"
GRAPH_SAVE_PATH = "../../models/training_loss_evaluator.png"

BATCH_SIZE = 2048
EPOCHS = 25 
LEARNING_RATE = 0.001
PATIENCE = 5

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Entraînement Evaluator sur {device} ---")
    
    train_files = glob.glob(os.path.join(DATA_DIR, "train_part_*.pt"))
    val_files = glob.glob(os.path.join(DATA_DIR, "val_part_*.pt"))
    
    if not train_files:
        print("Erreur: Lancez le nouveau preprocess.py !")
        return

    model = Evaluator(model_path=None).to(device)
    
    criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'loss': [], 'val_loss': []}

    best_val_loss = float('inf')
    patience_counter = 0

    global_start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        epoch_start_time = time.time()
        total_loss = 0.0

        try:
            data = torch.load(random.choice(train_files))
            dataset = TensorDataset(data['boards'], data['values'])
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            for x, y in dataloader:
                model.train()
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                outputs = model(x).squeeze()
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)

            avg_train_loss = total_loss / len(dataloader.dataset)
            history['loss'].append(avg_train_loss)
            del data, dataset, dataloader, x, y, outputs, loss
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Erreur pendant l'entraînement: {e}")
            continue

        model.eval()
        val_loss = 0.0
        val_batches = 0

        print("   [Validation] Évaluation sur les fichiers de validation...")
        with torch.no_grad():
            for f_path in val_files:
                try:
                    data = torch.load(f_path)
                    dataset = TensorDataset(data['boards'], data['values'])
                    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
                    for x, y in dataloader:
                        x, y = x.to(device), y.to(device)
                        outputs = model(x).squeeze()
                        loss = criterion(outputs, y)
                        val_loss += loss.item() * x.size(0)
                    val_batches += len(dataloader.dataset)
                    del data, dataset, dataloader, x, y, outputs, loss
                    gc.collect()
                    torch.cuda.empty_cache()
                except Exception as e:
                    pass
        
        avg_val_loss = val_loss / max(1, val_batches)
        history['val_loss'].append(avg_val_loss)

        # Logs et Sauvegarde du modèle si amélioration
        epoch_time = format_time(time.time() - epoch_start_time)
        print(f"   [Résultats] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Temps: {epoch_time}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   [Modèle Sauvegardé] Nouveau meilleur modèle avec Val Loss: {best_val_loss:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   [Patience] Pas d'amélioration ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("   [Arrêt Précoce] Pas d'amélioration depuis plusieurs époques.")
                break
        
    total_training_time = format_time(time.time() - global_start_time)
    print(f"\n=== Entraînement Terminé en {total_training_time} ===")

    # Sauvegarde du graphique de perte
    os.makedirs(os.path.dirname(GRAPH_SAVE_PATH), exist_ok=True)
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.title('Courbe de Perte pendant l\'Entraînement')
    plt.legend()
    plt.savefig(GRAPH_SAVE_PATH)
    plt.close()

if __name__ == "__main__":
    train()