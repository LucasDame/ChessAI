from abc import ABC, abstractmethod

class Piece(ABC):

    def __init__(self, color, name):
        self.__color = color
        self.__name = name
        self.__pos = [None, None]

    def get_color(self):
        return self.__color

    def get_name(self):
        return self.__name
    
    def get_position(self):
        return self.__pos
    
    def set_position(self, x, y):
        self.pos = [x, y]
    
    def set_position(self, row, col):
        self.position = (row, col)
    
    @abstractmethod
    def possible_moves(self, board):
        pass