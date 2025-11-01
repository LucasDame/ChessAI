from Piece import Piece

class Knight(Piece):

    def __init__(self, color, name = "knight"):
        super().__init__(color, name)

    def possible_moves(self, board):
        moves = []
        x, y = self.get_position()
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        
        for dx, dy in knight_moves:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < 8 and 0 <= new_y < 8:
                target_piece = board.get_piece(new_x, new_y)
                if target_piece is None or target_piece.get__color() != self.get__color():
                    moves.append((new_x, new_y))
        
        return moves