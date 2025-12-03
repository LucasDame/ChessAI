import torch
import chess
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# --- CORRECTION IMPORT ---
try:
    from ResNet.model_ResNet import ChessNet
except ImportError:
    from src.ResNet.model_ResNet import ChessNet
    
from dataset import board_to_tensor, index_to_move

MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", "chess_model_seresnet.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    if not os.path.exists(MODEL_PATH): return None
    # Paramètres identiques à l'entraînement !
    model = ChessNet(num_res_blocks=10, num_channels=128, use_se=True).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    except: return None

CHESS_MODEL = load_model()

def get_resnet_move(fen_string: str) -> str:
    if CHESS_MODEL is None: return ""
    board = chess.Board(fen_string)
    input_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = CHESS_MODEL(input_tensor)
    
    best_index = torch.argmax(output).item()
    best_move_obj = index_to_move(best_index)

    if best_move_obj in board.legal_moves:
        return board.uci(best_move_obj)
    
    top_indices = torch.topk(output, 20).indices.squeeze().tolist()
    for idx in top_indices:
        move = index_to_move(idx)
        if move in board.legal_moves: return board.uci(move)
            
    return board.uci(list(board.legal_moves)[0])