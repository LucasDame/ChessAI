from Piece import Piece

class Queen(Piece):

    def __init__(self, color, name = "queen"):
        super().__init__(color, name)
    
    def possible_moves(self, board):
        moves = []
        x, y = self.get_position()
        
        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            while 0 <= new_x < 8 and 0 <= new_y < 8:
                target_piece = board.get_piece(new_x, new_y)
                if target_piece is None:
                    moves.append((new_x, new_y))
                elif target_piece.get_color() != self.get_color():
                    moves.append((new_x, new_y))
                    break
                else:
                    break
                new_x += dx
                new_y += dy
        
        return moves