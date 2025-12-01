import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import glob
import random
import gc
import matplotlib.pyplot as plt
from model_AlphaZero import ChessNet
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
MODEL_SAVE_PATH = "../../models/chess_model_AlphaZero.pth"
GRAPH_SAVE_PATH = "../../models/training_loss_AlphaZero.png"

BATCH_SIZE = 2048
EPOCHS = 25
LEARNING_RATE = 0.001
PATIENCE = 5

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Entraînement Double Tête (Policy + Value) sur {device} ---")
    
    train_files = glob.glob(os.path.join(DATA_DIR, "train_part_*.pt"))
    val_files = glob.glob(os.path.join(DATA_DIR, "val_part_*.pt"))
    
    if not train_files:
        print("Erreur: Lancez le nouveau preprocess.py !")
        return

    model = ChessNet(num_res_blocks=10, use_se=True).to(device)
    
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'loss': [], 'val_loss': []}

    best_val_loss = float('inf')
    patience_counter = 0

    global_start_time = time.time()

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        epoch_start_time = time.time()

        # --- PHASE TRAINING ---
        model.train()
        total_loss = 0.0
        batches = 0
        random.shuffle(train_files)

        # UTILISATION DE ENUMERATE POUR LE SUIVI
        for i, f_path in enumerate(train_files):
            # Affiche : [Train] Fichier 1/19 : train_part_0.pt
            print(f"   [Train] Fichier {i+1}/{len(train_files)} : {os.path.basename(f_path)}")
            
            try:
                data = torch.load(f_path)
                dataset = TensorDataset(data['inputs'], data['labels_move'], data['labels_val'])
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
                
                for x, y_move, y_val in loader:
                    x, y_move, y_val = x.to(device), y_move.to(device), y_val.to(device)
                    
                    optimizer.zero_grad()
                    pred_policy, pred_value = model(x)
                    
                    loss_p = criterion_policy(pred_policy, y_move)
                    loss_v = criterion_value(pred_value, y_val)
                    
                    loss = loss_p + 0.01 * loss_v 
                    
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                batches += len(loader)
                del data, dataset, loader, x, y_move, y_val, pred_policy, pred_value, loss
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"Erreur fichier {f_path}: {e}")

        avg_loss = total_loss / max(1, batches)
        history['loss'].append(avg_loss)
        
        # --- PHASE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        print("   [Val] Validation en cours...")
        with torch.no_grad():
            for f_path in val_files:
                try:
                    data = torch.load(f_path)
                    dataset = TensorDataset(data['inputs'], data['labels_move'], data['labels_val'])
                    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
                    for x, y_m, y_v in loader:
                        x, y_m, y_v = x.to(device), y_m.to(device), y_v.to(device)
                        p_pol, p_val = model(x)
                        l_p = criterion_policy(p_pol, y_m)
                        l_v = criterion_value(p_val, y_v)
                        val_loss += (l_p + 0.01 * l_v).item()
                    val_batches += len(loader)
                    del data, dataset, loader; gc.collect()
                except: pass
        
        avg_val = val_loss / max(1, val_batches)
        history['val_loss'].append(avg_val)
        
        # --- LOGS ET SAUVEGARDE ---
        epoch_duration = time.time() - epoch_start_time
        print(f"   >>> Fin Epoch | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val:.4f}")
        print(f"   ⏱️ Durée: {format_time(epoch_duration)}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("   🏆 Nouveau Record ! Modèle sauvegardé.")
        else:
            patience_counter += 1
            print(f"   ⚠️ Pas d'amélioration ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("⛔ ARRÊT PRÉCOCE (Early Stopping).")
                break
    
    global_duration = time.time() - global_start_time
    print(f"\n✅ Entraînement terminé en : {format_time(global_duration)}")

    os.makedirs(os.path.dirname(GRAPH_SAVE_PATH), exist_ok=True)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('AlphaZero Loss')
    plt.legend()
    plt.savefig(GRAPH_SAVE_PATH)

if __name__ == "__main__":
    train()