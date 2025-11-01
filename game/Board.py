from Pown import Pown
from Rook import Rook
from Bishop import Bishop
from Knight import Knight
from King import King
from Queen import Queen

class Board():

    def __init__(self):
        self.__grid = [[None for _ in range(8)] for _ in range(8)]
        self.setup_pieces()

    def getGrid(self):
        return self.__grid

    def setup_pieces(self):
        # Setup pawns
        for col in range(8):
            self.__grid[1][col] = Pown("black")
            self.__grid[6][col] = Pown("white")
        
        # Setup rooks
        self.__grid[0][0] = Rook("black")
        self.__grid[0][7] = Rook("black")
        self.__grid[7][0] = Rook("white")
        self.__grid[7][7] = Rook("white")
        
        # Setup knights
        self.__grid[0][1] = Knight("black")
        self.__grid[0][6] = Knight("black")
        self.__grid[7][1] = Knight("white")
        self.__grid[7][6] = Knight("white")
        
        # Setup bishops
        self.__grid[0][2] = Bishop("black")
        self.__grid[0][5] = Bishop("black")
        self.__grid[7][2] = Bishop("white")
        self.__grid[7][5] = Bishop("white")
        
        # Setup queens
        self.__grid[0][3] = Queen("black")
        self.__grid[7][3] = Queen("white")
        
        # Setup kings
        self.__grid[0][4] = King("black")
        self.__grid[7][4] = King("white")

    def is_empty(self, row, col):
        if 0 <= row < 8 and 0 <= col < 8:
            return self.__grid[row][col] is None
        return False

    def is_opponent_piece(self, row, col, color):
        if 0 <= row < 8 and 0 <= col < 8:
            piece = self.__grid[row][col]
            return piece is not None and piece.get_color() != color
        return False
    
    def move_piece(self, from_row, from_col, to_row, to_col):
        piece = self.__grid[from_row][from_col]
        if piece:
            self.__grid[to_row][to_col] = piece
            self.__grid[from_row][from_col] = None
    
    def __str__(self):
        
        def piece_symbol(cell):
            if cell is None:
                return "__"
            name = getattr(cell, "name", None)
            if not name:
                name = cell.__class__.__name__
            name = name.lower()

            mapping = {
                "pown": "P", "pawn": "P", "pion": "P",
                "rook": "R", "tour": "R",
                "knight": "N", "horse": "N", "cheval": "N",
                "bishop": "B", "fou": "B",
                "queen": "Q", "dame": "Q",
                "king": "K", "roi": "K"
            }
            sym = mapping.get(name, name[0].upper() if name else "?")
            color = cell.get_color()
            if color == "white":
                return "w" + sym
            elif color == "black":
                return "b" + sym
            else:
                return " " + sym

        lines = []
        # build board lines with row numbers at left (8..1)
        for r in range(8):
            row = [piece_symbol(self.__grid[r][c]) for c in range(8)]
            lines.append(f"{8 - r} | " + " ".join(row))

        # separator and file letters below, aligned with the board
        prefix = " " * len("8 | ")
        # compute content width dynamically to match board string
        content_width = len(" ".join([piece_symbol(self.__grid[0][c]) for c in range(8)]))
        separator = prefix + "-" * content_width

        files = [chr(ord('a') + i) for i in range(8)]
        # each file label centered in a 3-char slot (matches "XX " pattern of board)
        bottom = prefix + "".join(f" {ch} " for ch in files)

        return "\n".join(lines + [separator, bottom])
