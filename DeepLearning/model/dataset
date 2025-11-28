import chess
import numpy as np
import torch

def board_to_tensor(board):
    """Convertit un board python-chess en tenseur (17, 8, 8)"""
    tensor = np.zeros((17, 8, 8), dtype=np.float32)
    
    # 1. Les Pièces (0-11)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            plane = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            row, col = divmod(square, 8)
            tensor[plane][row][col] = 1.0

    # 2. Métadonnées (12-16)
    if board.turn == chess.BLACK: tensor[12, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[16, :, :] = 1.0

    return torch.from_numpy(tensor)

def move_to_index(move):
    """Encode un coup (ex: a1a2) en un index unique (0-4095)"""
    return move.from_square * 64 + move.to_square

def index_to_move(idx):
    """Decode un index en objet Move"""
    return chess.Move(idx // 64, idx % 64)