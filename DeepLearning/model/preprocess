import chess.pgn
import torch
import os
from dataset import board_to_tensor, move_to_index

# --- CONFIGURATION ---
PGN_FILE = "../data/data.pgn"      # Mets ton fichier PGN ici
OUTPUT_FILE = "../data/train_data.pt"
MAX_POSITIONS = 500000             # Nombre de positions à extraire

def create_dataset():
    if not os.path.exists(PGN_FILE):
        print(f"Erreur : Fichier {PGN_FILE} introuvable.")
        return

    print(f"Lecture de {PGN_FILE}...")
    pgn = open(PGN_FILE)
    
    inputs = []
    labels = []
    count = 0
    
    while count < MAX_POSITIONS:
        try:
            game = chess.pgn.read_game(pgn)
        except: continue
        if game is None: break

        # Filtre Elo (Optionnel, ici on prend > 2000)
        try:
            elo = int(game.headers.get("WhiteElo", 0))
            if elo < 2000: continue
        except: continue

        board = game.board()
        for move in game.mainline_moves():
            # Input : Plateau avant le coup
            inputs.append(board_to_tensor(board))
            # Label : Le coup joué
            labels.append(move_to_index(move))
            
            board.push(move)
            count += 1
            if count % 10000 == 0: print(f"Positions extraites : {count}")
            if count >= MAX_POSITIONS: break

    print("Conversion en Tenseurs finaux...")
    # Stack transforme une liste de tenseurs en un seul gros tenseur
    data = {
        'inputs': torch.stack(inputs),
        'labels': torch.tensor(labels, dtype=torch.long)
    }
    
    os.makedirs("../data", exist_ok=True)
    torch.save(data, OUTPUT_FILE)
    print(f"Dataset sauvegardé dans {OUTPUT_FILE} ({count} positions)")

if __name__ == "__main__":
    create_dataset()