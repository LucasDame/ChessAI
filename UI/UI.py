import pygame
import subprocess
import sys
import threading
import queue

# --- CONFIGURATION ---
WIDTH, HEIGHT = 600, 600
DIMENSION = 8  # 8x8 cases
SQ_SIZE = HEIGHT // DIMENSION
FPS = 15

# Chemins
# Note: On n'utilise plus ENGINE_PATH pour l'exécution directe ici, 
# car on force l'utilisation de WSL.
ENGINE_PATH = r"C:/Users/msluc/OneDrive/Projets Info/ChessAI/ChessC/engine.exe"
LINUX_ENGINE_PATH = "/mnt/c/Users/msluc/OneDrive/Projets Info/ChessAI/ChessC/engine"

# Couleurs
COLOR_LIGHT = (234, 235, 200) # Beige
COLOR_DARK = (119, 149, 86)   # Vert
COLOR_HIGHLIGHT = (255, 255, 0, 100) # Jaune transparent

# Mapping des pièces pour l'affichage (Texte de secours)
PIECE_SYMBOLS = {
    'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
    'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k'
}

class ChessEngineProcess:
    """Gère la communication avec le programme C via WSL"""
    def __init__(self):
        
        # --- MODIFICATION CRUCIALE POUR WSL ---
        # On construit la commande sous forme de liste : ["wsl", "chemin_linux"]
        # Cela dit à Windows : "Lance WSL, et demande-lui d'exécuter ce fichier"
        self.command = ["wsl", LINUX_ENGINE_PATH]
        
        try:
            # Lance le moteur C via WSL
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0, 
            )
            self.q = queue.Queue()
            
            # Thread pour lire la sortie du C sans bloquer l'interface
            self.listener = threading.Thread(target=self.read_output)
            self.listener.daemon = True
            self.listener.start()
            
            print(f"[Python] Commande lancée : {' '.join(self.command)}")
            
        except FileNotFoundError:
            print(f"[Python] ERREUR : Impossible de lancer WSL ou le fichier n'existe pas.")
            print(f"Commande tentée : {self.command}")
            self.process = None

    def read_output(self):
        """Lit ce que le moteur C 'print' dans le terminal"""
        # Si le processus n'a pas démarré, on arrête tout de suite
        if not self.process:
            return

        while True:
            try:
                line = self.process.stdout.readline()
                if line:
                    self.q.put(line.strip())
                else:
                    # Si line est vide, le processus est peut-être mort
                    break
            except Exception as e:
                print(f"Erreur lecture : {e}")
                break

    def send_command(self, cmd):
        """Envoie une commande (ex: 'e2e4') au moteur C"""
        if self.process:
            print(f"[Python -> C] : {cmd}") 
            try:
                data = (cmd + "\n").encode('utf-8')
                self.process.stdin.write(data)
                self.process.stdin.flush()
            except BrokenPipeError:
                print("[Python] Erreur : Le moteur C s'est arrêté.")

    def get_messages(self):
        """Récupère les messages en attente"""
        msgs = []
        while not self.q.empty():
            msgs.append(self.q.get())
        return msgs

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Interface Moteur Bitboard (via WSL)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 32, bold=True)
        
        self.images = {}
        self.load_images()
        
        # État du jeu (Représentation interne simplifiée pour Python)
        self.board = [
            ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"],
            ["bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["--", "--", "--", "--", "--", "--", "--", "--"],
            ["wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"],
            ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        ]
        
        self.selected_sq = () # (row, col)
        self.player_clicks = [] # [(row, col), (row, col)]
        
        # Connexion au moteur (Pas besoin d'argument path car hardcodé pour WSL)
        self.engine = ChessEngineProcess()

    def load_images(self):
        pieces = ['wP', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bP', 'bR', 'bN', 'bB', 'bK', 'bQ']
        for piece in pieces:
            try:
                # Essaie de charger l'image
                self.images[piece] = pygame.image.load(f"assets/{piece}.png")
                self.images[piece] = pygame.transform.scale(self.images[piece], (SQ_SIZE, SQ_SIZE))
            except:
                pass

    def draw_board(self):
        colors = [COLOR_LIGHT, COLOR_DARK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[((r + c) % 2)]
                pygame.draw.rect(self.screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
                
                # Surlignage de la sélection
                if self.selected_sq == (r, c):
                    s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                    s.set_alpha(100)
                    s.fill((0, 0, 255))
                    self.screen.blit(s, (c*SQ_SIZE, r*SQ_SIZE))

    def draw_pieces(self):
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                piece = self.board[r][c]
                if piece != "--":
                    if piece in self.images:
                        self.screen.blit(self.images[piece], pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
                    else:
                        # Fallback texte si pas d'image
                        color = (0, 0, 0) if piece[0] == 'w' else (50, 50, 50)
                        text = self.font.render(PIECE_SYMBOLS[piece], True, color)
                        text_rect = text.get_rect(center=(c*SQ_SIZE + SQ_SIZE//2, r*SQ_SIZE + SQ_SIZE//2))
                        self.screen.blit(text, text_rect)

    def col_row_to_algebraic(self, col, row):
        """Convertit (col 0, row 7) -> 'a1'"""
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        rank = 8 - row 
        return f"{files[col]}{rank}"

    def update_board_from_move(self, start, end):
        """Met à jour le plateau Python VISUELLEMENT"""
        startRow, startCol = start
        endRow, endCol = end
        
        piece = self.board[startRow][startCol]
        self.board[endRow][endCol] = piece
        self.board[startRow][startCol] = "--"

    def run(self):
        running = True
        while running:
            # 1. Lire les messages du moteur C
            if self.engine:
                messages = self.engine.get_messages()
                for msg in messages:
                    print(f"[C > Python] : {msg}")
                    # TODO: Ajouter ici la logique si le coup est Illégal pour annuler le mouvement visuel

            # 2. Gestion Evénements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    location = pygame.mouse.get_pos()
                    col = location[0] // SQ_SIZE
                    row = location[1] // SQ_SIZE
                    
                    if self.selected_sq == (row, col):
                        self.selected_sq = ()
                        self.player_clicks = []
                    else:
                        self.selected_sq = (row, col)
                        self.player_clicks.append(self.selected_sq)
                    
                    if len(self.player_clicks) == 2:
                        start = self.player_clicks[0]
                        end = self.player_clicks[1]
                        
                        move_str = self.col_row_to_algebraic(start[1], start[0]) + \
                                   self.col_row_to_algebraic(end[1], end[0])
                        
                        if self.engine:
                            self.engine.send_command(move_str)
                        
                        self.update_board_from_move(start, end)
                        
                        self.selected_sq = ()
                        self.player_clicks = []

            # 3. Dessin
            self.draw_board()
            self.draw_pieces()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    g = Game()
    g.run()