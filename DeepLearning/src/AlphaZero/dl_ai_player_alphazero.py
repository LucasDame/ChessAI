import torch
import chess
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "AlphaZero"))

# Import du modèle ALPHAZERO
from AlphaZero.model_AlphaZero import ChessNet 
from dataset import board_to_tensor, index_to_move

MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", "chess_model_AlphaZero.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"[AlphaZero] Chargement depuis : {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERREUR] Modèle introuvable.")
        return None

    # Mêmes paramètres que train_AlphaZero.py
    model = ChessNet(num_res_blocks=10, num_channels=128, use_se=True).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"[SUCCÈS] AlphaZero chargé sur {DEVICE}.")
        return model
    except Exception as e:
        print(f"[ERREUR CHARGEMENT] {e}")
        return None

CHESS_MODEL = load_model()

def get_alphazero_move(fen_string: str) -> str:
    if CHESS_MODEL is None: return ""
    board = chess.Board(fen_string)
    input_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # AlphaZero renvoie DEUX valeurs : (Policy, Value)
        policy, value = CHESS_MODEL(input_tensor)
        
    # On affiche l'estimation de victoire (entre -1 et 1) pour le debug
    win_prob = value.item()
    print(f"[AlphaZero] Estimation position : {win_prob:.3f}")

    # On choisit le coup basé uniquement sur la Policy
    best_index = torch.argmax(policy).item()
    best_move_obj = index_to_move(best_index)

    if best_move_obj in board.legal_moves:
        return board.san(best_move_obj)
    
    # Fallback
    top_indices = torch.topk(policy, 20).indices.squeeze().tolist()
    for idx in top_indices:
        move = index_to_move(idx)
        if move in board.legal_moves: return board.san(move)
            
    return board.san(list(board.legal_moves)[0])