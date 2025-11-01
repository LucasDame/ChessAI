from Piece import Piece

class King(Piece):

    def __init__(self, color, name = "king"):
        super().__init__(color, name)
    
    def possible_moves(self, board):
        moves = []
        x, y = self.get_position()
        king_moves = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        for dx, dy in king_moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target_piece = board.get_piece(new_x, new_y)
                if target_piece is None or target_piece.get_color() != self.get_color():
                    moves.append((new_x, new_y))
        
        return moves