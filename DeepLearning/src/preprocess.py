import chess.pgn
import torch
import os
import random
from dataset import board_to_tensor, move_to_index

# --- CONFIGURATION ---
PGN_FILE = "../data/data.pgn"
TRAIN_OUTPUT = "../data/train_data.pt"
VAL_OUTPUT = "../data/val_data.pt"
MAX_POSITIONS = 1000000  # On augmente pour le "gros" dataset
VAL_RATIO = 0.1          # 10% des données pour le test

def create_datasets():
    if not os.path.exists(PGN_FILE):
        print(f"Erreur : {PGN_FILE} introuvable.")
        return

    print(f"Lecture de {PGN_FILE}...")
    pgn = open(PGN_FILE)
    
    # On stocke tout dans une liste temporaire
    data_samples = []
    count = 0
    
    while count < MAX_POSITIONS:
        try:
            game = chess.pgn.read_game(pgn)
        except: continue
        if game is None: break

        # Filtre Elo (> 2000)
        try:
            elo = int(game.headers.get("WhiteElo", 0))
            if elo < 2000: continue
        except: continue

        board = game.board()
        for move in game.mainline_moves():
            tensor = board_to_tensor(board)
            move_idx = move_to_index(move)
            
            # On stocke le tuple (input, label)
            data_samples.append((tensor, move_idx))
            
            board.push(move)
            count += 1
            if count % 50000 == 0: print(f"Positions extraites : {count}")
            if count >= MAX_POSITIONS: break

    print(f"Total positions : {len(data_samples)}")
    print("Mélange des données (Shuffle)...")
    random.shuffle(data_samples) # Très important !

    # Séparation Train / Val
    split_idx = int(len(data_samples) * (1 - VAL_RATIO))
    train_samples = data_samples[:split_idx]
    val_samples = data_samples[split_idx:]

    print(f"Train set: {len(train_samples)} | Val set: {len(val_samples)}")

    def save_split(samples, filename):
        # Séparation des inputs et labels pour le stockage tensoriel optimisé
        inputs = torch.stack([s[0] for s in samples])
        labels = torch.tensor([s[1] for s in samples], dtype=torch.long)
        torch.save({'inputs': inputs, 'labels': labels}, filename)
        print(f"Sauvegardé : {filename}")

    print("Sauvegarde Train...")
    save_split(train_samples, TRAIN_OUTPUT)
    
    print("Sauvegarde Val...")
    save_split(val_samples, VAL_OUTPUT)

if __name__ == "__main__":
    create_datasets()