from Board import Board


def main():
    chess_board = Board()
    chess_board.move_piece(6, 0, 5, 0)
    print(chess_board)
    

    

if __name__ == "__main__":
    main()