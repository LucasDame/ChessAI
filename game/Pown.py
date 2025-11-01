from Piece import Piece

class Pown(Piece):

    def __init__(self, color, name="pown"):
        super().__init__(color, name)
    
    def possible_moves(self, board):
        moves = []
        direction = 1 if self.__color == "white" else -1
        start_row = 6 if self.__color == "white" else 1
        current_row, current_col = self.position

        # Move forward one square
        if board.is_empty(current_row + direction, current_col):
            moves.append((current_row + direction, current_col))
            # Move forward two squares from starting position
            if current_row == start_row and board.is_empty(current_row + 2 * direction, current_col):
                moves.append((current_row + 2 * direction, current_col))

        # Capture diagonally
        for col_offset in [-1, 1]:
            new_col = current_col + col_offset
            if board.is_opponent_piece(current_row + direction, new_col, self.__color):
                moves.append((current_row + direction, new_col))

        return moves

