import torch
import torch.nn as nn
import chess
import numpy as np
import os
import sys

# Gestion des chemins pour importer dataset.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from dataset import board_to_tensor

# --- CONFIGURATION ---
# Nom du fichier à charger (Mets ici le numéro de ta meilleure génération)
# Exemple : "genetic_best_gen_50.pth"
MODEL_FILENAME = "genetic_best_gen_50.pth" 
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", MODEL_FILENAME)
DEVICE = torch.device("cpu") # Le TinyNet est si petit que le CPU est souvent plus rapide (pas d'overhead transfert GPU)
DEPTH_INFERENCE = 3 # On peut monter à 3 ou 4 car le réseau est léger

# --- 1. L'ARCHITECTURE (Doit être IDENTIQUE à genetic_trainer.py) ---
class TinyChessNet(nn.Module):
    def __init__(self):
        super(TinyChessNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.net(x) * 10 

# --- 2. CHARGEMENT DU MODÈLE ---
def load_model():
    print(f"[GENETIC] Recherche du modèle : {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        # Si le fichier 50 n'existe pas, on essaie de trouver le plus récent
        import glob
        list_of_files = glob.glob(os.path.join(CURRENT_DIR, "..", "models", "genetic_best_gen_*.pth"))
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"[GENETIC] Modèle spécifié introuvable, chargement du plus récent : {os.path.basename(latest_file)}")
            path_to_load = latest_file
        else:
            print("[ERREUR] Aucun modèle génétique trouvé !")
            return None
    else:
        path_to_load = MODEL_PATH

    model = TinyChessNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(path_to_load, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        print(f"[ERREUR CHARGEMENT] {e}")
        return None

GENETIC_MODEL = load_model()

# --- 3. MOTEUR ALPHA-BETA (Optimisé pour le jeu) ---
def evaluate_position(model, board):
    with torch.no_grad():
        data = board_to_tensor(board)
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float().unsqueeze(0).to(DEVICE)
        else:
            tensor = data.float().unsqueeze(0).to(DEVICE)
        return model(tensor).item()

def alpha_beta(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_position(GENETIC_MODEL, board)

    moves = list(board.legal_moves)
    # Tri des coups pour optimiser l'élagage (Captures en premier)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    if maximizing_player:
        max_eval = -float('inf')
        for move in moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

# --- 4. FONCTION PRINCIPALE ---
def get_genetic_move(fen_string: str) -> str:
    if GENETIC_MODEL is None: return ""
    
    board = chess.Board(fen_string)
    best_move = None
    
    alpha = -float('inf')
    beta = float('inf')
    
    # L'IA veut toujours maximiser son score relatif
    # (Le réseau a appris : +10 = Bon pour Blanc, -10 = Bon pour Noir ?)
    # ATTENTION : Dans genetic_trainer, on entraînait les Blancs.
    # Si le réseau est symétrique (input tensor gère le tour), ça va.
    # Sinon, il faut inverser le score si c'est aux noirs.
    # Supposons que le réseau donne un score absolu (positif = avantage Blanc).
    
    is_white_turn = (board.turn == chess.WHITE)
    
    # Si c'est aux blancs, on veut MAXIMISER le score.
    # Si c'est aux noirs, on veut MINIMISER le score.
    target_max = is_white_turn 
    
    best_val = -float('inf') if target_max else float('inf')
    
    moves = list(board.legal_moves)
    # Tri simple pour la racine aussi
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    for move in moves:
        board.push(move)
        # Appel récursif
        val = alpha_beta(board, DEPTH_INFERENCE - 1, alpha, beta, not target_max)
        board.pop()

        if target_max: # Blancs
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, val)
        else: # Noirs
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, val)
            
    if best_move:
        return best_move.uci()
    
    # Fallback
    return list(board.legal_moves)[0].uci()