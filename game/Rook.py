from Piece import Piece

class Rook(Piece):

    def __init__(self, color, name = "rook"):
        super().__init__(color, name)
    
    def possible_moves(self, board):
        moves = []
        row, col = self.get_position()
        
        # Vertical and horizontal moves
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                if board.is_empty(r, c):
                    moves.append((r, c))
                else:
                    if board.get_piece(r, c).get_color() != self.get_color():
                        moves.append((r, c))
                    break
                r += dr
                c += dc
        
        return moves