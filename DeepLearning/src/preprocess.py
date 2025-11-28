import chess.pgn
import torch
import os
import glob
import gc # Garbage Collector pour libérer la RAM
from dataset import board_to_tensor, move_to_index

# --- CONFIGURATION ---
RAW_DATA_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed" # Nouveau dossier pour les chunks
CHUNK_SIZE = 500000              # Sauvegarde tous les 500k coups
MAX_TOTAL_POSITIONS = 10000000    # Ton objectif ambitieux
VAL_RATIO = 0.1

def create_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pgn_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pgn"))
    if not pgn_files:
        print(f"Erreur : Aucun fichier .pgn trouvé dans {RAW_DATA_DIR}")
        return

    # Buffers temporaires
    train_buffer = []
    val_buffer = []
    
    total_extracted = 0
    chunk_id = 0

    print(f"Début de l'extraction (Mode Chunking). Objectif : {MAX_TOTAL_POSITIONS}")

    for pgn_path in pgn_files:
        if total_extracted >= MAX_TOTAL_POSITIONS: break
        
        print(f"\nLecture de : {os.path.basename(pgn_path)}")
        pgn = open(pgn_path)
        
        while True:
            if total_extracted >= MAX_TOTAL_POSITIONS: break
            
            try:
                game = chess.pgn.read_game(pgn)
            except: continue
            if game is None: break

            # --- FILTRES (ELO + TEMPS) ---
            headers = game.headers
            try:
                # 1. ELO > 2200
                if int(headers.get("WhiteElo", 0)) < 2200 or int(headers.get("BlackElo", 0)) < 2200:
                    continue
                
                # 2. TEMPS (Anti-Bullet)
                # On veut au moins 180 secondes (3 minutes)
                time_control = headers.get("TimeControl", "")
                if not time_control or "?" in time_control: continue
                
                if "+" in time_control:
                    base = int(time_control.split("+")[0])
                else:
                    base = int(time_control)
                
                if base < 180: continue # C'est du Bullet/UltraBullet
            except: continue

            # --- EXTRACTION ---
            board = game.board()
            for move in game.mainline_moves():
                tensor = board_to_tensor(board)
                move_idx = move_to_index(move)
                
                # Répartition aléatoire immédiate (90% train, 10% val)
                # On utilise un hash simple pour la déterministe ou random
                if hash(str(board)) % 10 == 0: # Environ 10%
                    val_buffer.append((tensor, move_idx))
                else:
                    train_buffer.append((tensor, move_idx))
                
                board.push(move)
                total_extracted += 1

                # --- SAUVEGARDE INCREMENTALE (CHUNKING) ---
                # Si le buffer Train est plein, on sauvegarde et on vide
                if len(train_buffer) >= CHUNK_SIZE:
                    save_chunk(train_buffer, "train", chunk_id)
                    train_buffer = [] # Libère la RAM
                    gc.collect()      # Force le nettoyage
                    chunk_id += 1
                    print(f"   > Chunk {chunk_id-1} sauvegardé. Total: {total_extracted}")

    # Sauvegarde des restes
    if train_buffer: save_chunk(train_buffer, "train", chunk_id)
    if val_buffer: save_chunk(val_buffer, "val", 0) # On met tout le val dans un seul fichier souvent
    
    print("\nTerminé !")

def save_chunk(data, prefix, idx):
    filename = os.path.join(OUTPUT_DIR, f"{prefix}_part_{idx}.pt")
    inputs = torch.stack([s[0] for s in data])
    labels = torch.tensor([s[1] for s in data], dtype=torch.long)
    torch.save({'inputs': inputs, 'labels': labels}, filename)

if __name__ == "__main__":
    create_datasets()