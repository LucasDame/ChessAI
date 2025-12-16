import torch
import torch.nn as nn
import chess
import numpy as np
import sys
import os

# Gestion des imports pour trouver dataset.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset import board_to_tensor

# --- 1. ARCHITECTURE LÉGÈRE ---
class TinyChessNet(nn.Module):
    def __init__(self):
        super(TinyChessNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17 * 8 * 8, 128), # Un peu plus large pour mieux apprendre
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x) * 10

# --- 2. MOTEUR D'INFÉRENCE ALPHA-BETA ---
def evaluate_board(model, board):
    with torch.no_grad():
        data = board_to_tensor(board)
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float().unsqueeze(0)
        else:
            tensor = data.float().unsqueeze(0)
        return model(tensor).item()

def alpha_beta(board, depth, alpha, beta, maximizing, model):
    if depth == 0 or board.is_game_over():
        return evaluate_board(model, board)

    moves = list(board.legal_moves)
    # Tri simple pour optimiser l'élagage (captures d'abord)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    if maximizing:
        max_eval = -float('inf')
        for move in moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha: break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha: break
        return min_eval

def get_best_move(model, board, depth=2):
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    
    # L'IA joue toujours pour maximiser SON score vu par le réseau
    # Si le réseau est bien entraîné, il doit renvoyer +10 pour "Bon pour Blanc" 
    # et -10 pour "Bon pour Noir".
    # Donc si c'est aux Blancs, on maximise. Si c'est aux Noirs, on minimise.
    is_white = (board.turn == chess.WHITE)
    best_val = -float('inf') if is_white else float('inf')
    
    moves = list(board.legal_moves)
    # Mélange pour éviter la répétition, puis tri tactique
    import random
    random.shuffle(moves)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    for move in moves:
        board.push(move)
        val = alpha_beta(board, depth - 1, alpha, beta, not is_white, model)
        board.pop()

        if is_white:
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, val)
        else:
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, val)
            
    return best_move if best_move else moves[0]