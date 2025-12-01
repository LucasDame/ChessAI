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

# --- CONFIGURATION ---
DATA_DIR = "../data/processed_dual" # Le nouveau dossier
MODEL_SAVE_PATH = "../models/chess_model_AlphaZero.pth"
GRAPH_SAVE_PATH = "../models/training_loss_AlphaZero.png"

BATCH_SIZE = 512
EPOCHS = 15
LEARNING_RATE = 0.001

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Entraînement Double Tête (Policy + Value) ---")
    
    train_files = glob.glob(os.path.join(DATA_DIR, "train_part_*.pt"))
    val_files = glob.glob(os.path.join(DATA_DIR, "val_part_*.pt"))
    
    if not train_files:
        print("Erreur: Lancez le nouveau preprocess.py !")
        return

    model = ChessNet(num_res_blocks=10, use_se=True).to(device)
    
    # Deux critères de perte
    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss() # Mean Squared Error pour la valeur (-1 à 1)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'loss': [], 'val_loss': []}

    for epoch in range(EPOCHS):
        print(f"\n=== EPOCH {epoch+1}/{EPOCHS} ===")
        model.train()
        total_loss = 0.0
        batches = 0
        random.shuffle(train_files)

        for f_path in train_files:
            data = torch.load(f_path)
            # On charge 3 éléments maintenant
            dataset = TensorDataset(data['inputs'], data['labels_move'], data['labels_val'])
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            for x, y_move, y_val in loader:
                x, y_move, y_val = x.to(device), y_move.to(device), y_val.to(device)
                
                optimizer.zero_grad()
                
                # Le modèle retourne 2 choses
                pred_policy, pred_value = model(x)
                
                # Calcul des 2 pertes
                loss_p = criterion_policy(pred_policy, y_move)
                loss_v = criterion_value(pred_value, y_val)
                
                # Perte totale (On donne un petit poids à la value pour équilibrer)
                loss = loss_p + 0.01 * loss_v 
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            batches += len(loader)
            del data, dataset, loader; gc.collect(); torch.cuda.empty_cache()

        avg_loss = total_loss / batches
        history['loss'].append(avg_loss)
        print(f"   >>> Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f}")
        
        # Validation rapide
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for f_path in val_files:
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
        
        avg_val = val_loss / val_batches
        history['val_loss'].append(avg_val)
        print(f"   >>> Val Loss: {avg_val:.4f}")
        
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Graphique
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.legend()
    plt.savefig(GRAPH_SAVE_PATH)

if __name__ == "__main__":
    train()