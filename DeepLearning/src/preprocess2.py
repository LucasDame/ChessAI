import chess.pgn
import torch
import os
import glob
import gc
from dataset import board_to_tensor, move_to_index

# --- CONFIGURATION ---
RAW_DATA_DIR = "../data/raw"
OUTPUT_DIR = "../data/processed" # Nouveau dossier pour distinguer
CHUNK_SIZE = 500000 
MAX_TOTAL_POSITIONS = 10000000 

def get_game_result_value(game):
    """Retourne 1.0 si Blanc gagne, -1.0 si Noir gagne, 0.0 si Nul"""
    res = game.headers.get("Result", "*")
    if res == "1-0": return 1.0
    elif res == "0-1": return -1.0
    elif res == "1/2-1/2": return 0.0
    return None # Partie non finie ou inconnue

def create_datasets():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pgn_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pgn"))
    
    train_buffer = []
    val_buffer = []
    total_extracted = 0
    chunk_id = 0

    for pgn_path in pgn_files:
        if total_extracted >= MAX_TOTAL_POSITIONS: break
        print(f"Lecture : {os.path.basename(pgn_path)}")
        pgn = open(pgn_path)
        
        while True:
            if total_extracted >= MAX_TOTAL_POSITIONS: break
            try: game = chess.pgn.read_game(pgn)
            except: continue
            if game is None: break

            # Filtres (ELO + Temps) - Identique à avant
            try:
                if int(game.headers.get("WhiteElo", 0)) < 2200: continue
                tc = game.headers.get("TimeControl", "")
                if not tc or (int(tc.split("+")[0]) if "+" in tc else int(tc)) < 180: continue
            except: continue

            # --- NOUVEAU : Récupération du résultat ---
            result_value = get_game_result_value(game)
            if result_value is None: continue # On ignore les parties non terminées

            if result_value != 0.0:
                # On regarde le dernier coup du PGN
                try:
                    last_move = game.end().move
                    board_end = game.end().board()
                    
                    # Si ce n'est PAS un échec et mat, on ignore la partie !
                    # (Donc on vire les abandons et les pertes au temps)
                    if not board_end.is_checkmate():
                        continue 
                except:
                    continue

            board = game.board()
            for move in game.mainline_moves():
                tensor = board_to_tensor(board)
                move_idx = move_to_index(move)
                
                # --- CALCUL DE LA VALEUR RELATIVE ---
                # Si c'est aux Blancs de jouer et que Blanc a gagné (1.0) -> Valeur = 1.0
                # Si c'est aux Noirs de jouer et que Blanc a gagné (1.0) -> Valeur = -1.0 (Noir va perdre)
                turn_multiplier = 1.0 if board.turn == chess.WHITE else -1.0
                current_value = result_value * turn_multiplier
                
                # On stocke le trio : (Input, Move Label, Value Label)
                sample = (tensor, move_idx, current_value)
                
                if hash(str(board)) % 10 == 0: val_buffer.append(sample)
                else: train_buffer.append(sample)
                
                board.push(move)
                total_extracted += 1

                if len(train_buffer) >= CHUNK_SIZE:
                    save_chunk(train_buffer, "train", chunk_id)
                    train_buffer = []
                    gc.collect()
                    chunk_id += 1
                    print(f"   > Chunk {chunk_id} sauvegardé. Total: {total_extracted}")

    if train_buffer: save_chunk(train_buffer, "train", chunk_id)
    if val_buffer: save_chunk(val_buffer, "val", 0)
    print("Terminé.")

def save_chunk(data, prefix, idx):
    filename = os.path.join(OUTPUT_DIR, f"{prefix}_part_{idx}.pt")
    inputs = torch.stack([s[0] for s in data])
    labels_move = torch.tensor([s[1] for s in data], dtype=torch.long)
    labels_val = torch.tensor([s[2] for s in data], dtype=torch.float32).unsqueeze(1) # Shape (N, 1)
    
    torch.save({
        'inputs': inputs, 
        'labels_move': labels_move,
        'labels_val': labels_val
    }, filename)

if __name__ == "__main__":
    create_datasets()