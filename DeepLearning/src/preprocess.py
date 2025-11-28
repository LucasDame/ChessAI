import chess.pgn
import torch
import os
import random
import glob # Pour trouver tous les fichiers
from dataset import board_to_tensor, move_to_index

# --- CONFIGURATION ---
RAW_DATA_DIR = "../data/raw"       # Dossier contenant tes multiples .pgn
TRAIN_OUTPUT = "../data/train_data.pt"
VAL_OUTPUT = "../data/val_data.pt"

# Limite globale pour ne pas exploser la RAM
# 2 Millions de positions = environ 6-8 Go de RAM nécessaire lors du traitement
MAX_TOTAL_POSITIONS = 2000000  
VAL_RATIO = 0.1

def create_datasets():
    # 1. Trouver tous les fichiers PGN
    # On cherche les fichiers qui finissent par .pgn dans le dossier raw
    pgn_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.pgn"))
    
    if not pgn_files:
        print(f"Erreur : Aucun fichier .pgn trouvé dans {RAW_DATA_DIR}")
        print("Créez le dossier 'raw' et mettez vos fichiers dedans.")
        return

    print(f"Fichiers trouvés ({len(pgn_files)}) :")
    for f in pgn_files: print(f" - {os.path.basename(f)}")

    data_samples = []
    total_count = 0
    
    # 2. Boucle sur chaque fichier
    for pgn_path in pgn_files:
        if total_count >= MAX_TOTAL_POSITIONS:
            break
            
        print(f"\n--- Traitement de {os.path.basename(pgn_path)} ---")
        pgn = open(pgn_path)
        file_count = 0
        
        while True:
            # Sécurité globale
            if total_count >= MAX_TOTAL_POSITIONS:
                print("Limite globale atteinte !")
                break

            try:
                game = chess.pgn.read_game(pgn)
            except: continue
            
            # ... (Lecture du jeu juste avant) ...
            
            if game is None: 
                break

            # --- FILTRES DE QUALITÉ ---
            headers = game.headers

            # 1. Filtre ELO (Niveau des joueurs)
            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
                # On veut que les DEUX joueurs soient forts
                if white_elo < 2200 or black_elo < 2200: 
                    continue 
            except: continue

            # 2. Filtre TEMPS (Exclure le Bullet)
            # Format TimeControl : "300+0" (300 secondes + 0 incrément)
            time_control = headers.get("TimeControl", "")
            
            # On ignore les parties sans temps défini
            if not time_control or "?" in time_control:
                continue

            try:
                if "+" in time_control:
                    base_time, increment = time_control.split("+")
                    base_time = int(base_time)
                else:
                    base_time = int(time_control)

                # RÈGLE : On rejette tout ce qui est en dessous de 180 secondes (3 minutes)
                # Bullet = < 180s. Blitz = 180s à 480s. Rapid/Classical > 480s.
                if base_time < 180: 
                    continue 
            except:
                continue # Si le format est bizarre, on ignore par sécurité

            # ... (La suite avec board_to_tensor reste inchangée) ...

            board = game.board()
            for move in game.mainline_moves():
                # On capture
                tensor = board_to_tensor(board)
                move_idx = move_to_index(move)
                
                # On stocke (Attention, cela consomme de la RAM)
                data_samples.append((tensor, move_idx))
                
                board.push(move)
                total_count += 1
                file_count += 1
                
                if total_count % 50000 == 0: 
                    print(f"   > Total accumulé : {total_count}")
                    
                if total_count >= MAX_TOTAL_POSITIONS: break
        
        print(f"Fin du fichier. {file_count} positions extraites.")

    # 3. Mélange et Sauvegarde
    print(f"\nExtraction terminée. Total final : {total_count} positions.")
    
    if total_count == 0:
        print("Aucune position extraite ! Vérifiez vos filtres ELO ou vos fichiers.")
        return

    print("Mélange des données (Shuffle)...")
    random.shuffle(data_samples)

    # Séparation
    split_idx = int(len(data_samples) * (1 - VAL_RATIO))
    train_samples = data_samples[:split_idx]
    val_samples = data_samples[split_idx:]

    print(f"Train set: {len(train_samples)} | Val set: {len(val_samples)}")

    def save_split(samples, filename):
        if not samples: return
        print(f"Conversion Tenseurs pour {filename} (Patience, ça peut prendre du temps)...")
        # stack est l'opération lourde en RAM
        inputs = torch.stack([s[0] for s in samples])
        labels = torch.tensor([s[1] for s in samples], dtype=torch.long)
        torch.save({'inputs': inputs, 'labels': labels}, filename)
        print(f"Sauvegardé : {filename}")

    save_split(train_samples, TRAIN_OUTPUT)
    save_split(val_samples, VAL_OUTPUT)
    print("\nTout est prêt pour l'entraînement !")

if __name__ == "__main__":
    create_datasets()