import torch
import chess
import os
import sys

# Ajout dynamique du chemin pour trouver les modules
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "ResNet")) # Adapter si besoin

# Import du modèle RESNET
from ResNet.Model_ResNet import ChessNet  
from dataset import board_to_tensor, index_to_move

# --- CONFIGURATION ---
# Chemin vers le modèle entraîné ResNet
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", "chess_model_resnet.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print(f"[ResNet] Chargement depuis : {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERREUR] Modèle introuvable.")
        return None

    # ATTENTION : Il faut les mêmes paramètres qu'à l'entraînement !
    # Exemple : num_res_blocks=10, use_se=True
    model = ChessNet(num_res_blocks=10, num_channels=128, use_se=True).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"[SUCCÈS] ResNet chargé sur {DEVICE}.")
        return model
    except Exception as e:
        print(f"[ERREUR CHARGEMENT] {e}")
        return None

CHESS_MODEL = load_model()

def get_resnet_move(fen_string: str) -> str:
    if CHESS_MODEL is None: return ""
    board = chess.Board(fen_string)
    input_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = CHESS_MODEL(input_tensor) # ResNet sort juste la Policy
        
    best_index = torch.argmax(output).item()
    best_move_obj = index_to_move(best_index)

    if best_move_obj in board.legal_moves:
        return board.san(best_move_obj)
    
    # Fallback (Coup légal si le meilleur est illégal)
    top_indices = torch.topk(output, 20).indices.squeeze().tolist()
    for idx in top_indices:
        move = index_to_move(idx)
        if move in board.legal_moves: return board.san(move)
            
    return board.san(list(board.legal_moves)[0])