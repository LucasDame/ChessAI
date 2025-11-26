import pygame
import subprocess
import sys
import threading
import queue
import socket
import time
import os

# --- CONFIGURATION ---
WIDTH, HEIGHT = 600, 600
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
FPS = 15

# Réseau
HOST = "127.0.0.1"
PORT = 12345

# --- CHEMINS (CORRECTION ROBUSTE) ---
# BASE_DIR est le dossier où se trouve ce fichier UI.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# On suppose que le dossier 'assets' est à côté de UI.py
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Le chemin vers l'exécutable Linux (vu depuis WSL)
# ATTENTION : Il faut pointer vers le FICHIER exécutable, pas le dossier.
# Si tu as compilé avec "gcc -o engine", le fichier s'appelle "engine".
LINUX_ENGINE_PATH = "/mnt/c/Users/msluc/OneDrive/Projets Info/ChessAI/ChessC/API"

# Couleurs
COLOR_LIGHT = (234, 235, 200) 
COLOR_DARK = (119, 149, 86)   
COLOR_HIGHLIGHT = (255, 255, 0, 100) 

PIECE_SYMBOLS = {
    'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
    'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k'
}

# Ajout de constantes pour la connexion
MAX_RETRIES = 10
RETRY_DELAY = 0.5  # secondes

class ChessAPIClient:
    def __init__(self):
        self.sock = None
        self.process = None
        self.q = queue.Queue()
        self.running = True

        print("[Python] Lancement du serveur C via WSL...")
        self.launch_server()
        # On attend un peu, mais le connect_to_server gère les retries maintenant
        time.sleep(4) 
        self.connect_to_server()

    def launch_server(self):
        cmd = ["wsl", LINUX_ENGINE_PATH]
        try:
            self.process = subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE 
            )
        except FileNotFoundError:
            print(f"[Erreur] Impossible de lancer : {cmd}")

    def connect_to_server(self):
        """Tente de se connecter au port TCP avec des réessais."""
        print(f"[Python] Tentative de connexion à {HOST}:{PORT}...")
        for attempt in range(MAX_RETRIES):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((HOST, PORT))
                print(f"[Python] Connecté au moteur après {attempt + 1} essai(s).")
                
                self.listener = threading.Thread(target=self.receive_data)
                self.listener.daemon = True
                self.listener.start()
                return 

            except ConnectionRefusedError:
                print(f"[Python] Essai {attempt + 1}/{MAX_RETRIES} échoué. Le serveur démarre...")
                if self.sock: self.sock.close()
                time.sleep(RETRY_DELAY)

        print("[Erreur Critique] Impossible de se connecter au moteur C.")
        sys.exit(1)

    def receive_data(self):
        while self.running and self.sock:
            try:
                data = self.sock.recv(4096) # Augmenté un peu pour être sûr d'avoir tout le board
                if not data: break
                
                # Le serveur peut envoyer plusieurs messages collés
                # On décode tout
                full_msg = data.decode('utf-8').strip()
                
                # On peut recevoir "board:XXX\nillegal..."
                # On split par ligne pour traiter chaque message
                lines = full_msg.split('\n') # Ou split par le caractère nul si tu l'utilises
                
                for line in lines:
                    line = line.strip()
                    if line: self.q.put(line)
                    
            except: break

    def send_command(self, cmd):
        if self.sock:
            try:
                print(f"[Python -> API] : {cmd}")
                self.sock.sendall(cmd.encode('utf-8'))
            except: pass

    def get_messages(self):
        msgs = []
        while not self.q.empty(): msgs.append(self.q.get())
        return msgs

    def close(self):
        self.running = False
        if self.sock: self.sock.close()
        if self.process: self.process.terminate()

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess Client (Source de Vérité: C)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 32, bold=True)
        
        self.images = {}
        self.load_images()
        
        # Le plateau est initialisé vide, on attendra que le C nous donne l'état
        # Ou on l'init standard pour le premier affichage
        self.board = [["--" for _ in range(8)] for _ in range(8)]
        self.init_standard_board()
        
        self.selected_sq = () 
        self.player_clicks = [] 
        self.client = ChessAPIClient()

    def init_standard_board(self):
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

    def load_images(self):
        pieces = ['wP', 'wR', 'wN', 'wB', 'wK', 'wQ', 'bP', 'bR', 'bN', 'bB', 'bK', 'bQ']
        print(f"[Info] Recherche des images dans : {ASSETS_DIR}")
        
        for piece in pieces:
            # CORRECTION : On utilise ASSETS_DIR calculé proprement
            path = os.path.join(ASSETS_DIR, f"{piece}.png")
            try:
                self.images[piece] = pygame.image.load(path)
                self.images[piece] = pygame.transform.scale(self.images[piece], (SQ_SIZE, SQ_SIZE))
            except Exception as e:
                if piece == 'wP': # Log juste une fois pour pas spammer
                    print(f"[Attention] Image non trouvée : {path}")

    # --- LA FONCTION QUI MANQUAIT EST ICI ---
    def set_board_from_string(self, board_str):
        """Met à jour tout le plateau à partir de la chaîne de 64 chars reçue du C"""
        # Mapping des chars C vers les codes pièces Python
        char_to_piece = {
            'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
            'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK',
            '-': '--'
        }
        
        # On s'assure de ne pas dépasser 64 chars
        limit = min(len(board_str), 64)
        
        for i in range(limit):
            char = board_str[i]
            row = i // 8
            col = i % 8
            if char in char_to_piece:
                self.board[row][col] = char_to_piece[char]
    # -----------------------------------------

    def draw_board(self):
        colors = [COLOR_LIGHT, COLOR_DARK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[((r + c) % 2)]
                pygame.draw.rect(self.screen, color, pygame.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
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
                        color = (0, 0, 0) if piece[0] == 'w' else (50, 50, 50)
                        text = self.font.render(PIECE_SYMBOLS.get(piece, "?"), True, color)
                        text_rect = text.get_rect(center=(c*SQ_SIZE + SQ_SIZE//2, r*SQ_SIZE + SQ_SIZE//2))
                        self.screen.blit(text, text_rect)

    def get_promotion_choice(self, color):
        """Affiche un menu pour choisir la promotion et retourne 'q', 'r', 'b', ou 'n'"""
        # On dessine un rectangle au centre
        w, h = 300, 100
        rect_x = (WIDTH - w) // 2
        rect_y = (HEIGHT - h) // 2
        
        # Couleurs des boutons
        pygame.draw.rect(self.screen, (50, 50, 50), (rect_x, rect_y, w, h))
        pygame.draw.rect(self.screen, (200, 200, 200), (rect_x, rect_y, w, h), 3)
        
        options = ['q', 'r', 'b', 'n'] # Ordre des boutons
        # Mapping pour afficher les images correspondantes
        piece_codes = {
            'q': 'wQ' if color == 'w' else 'bQ',
            'r': 'wR' if color == 'w' else 'bR',
            'b': 'wB' if color == 'w' else 'bB',
            'n': 'wN' if color == 'w' else 'bN'
        }
        
        btn_width = w // 4
        
        # --- Boucle d'attente locale (bloque le jeu jusqu'au choix) ---
        waiting = True
        choice = 'q' # Par défaut
        
        while waiting:
            for i, opt in enumerate(options):
                # Dessin des zones
                bx = rect_x + i * btn_width
                pygame.draw.rect(self.screen, (100, 100, 100), (bx, rect_y, btn_width, h), 1)
                
                # Dessin de la pièce
                p_code = piece_codes[opt]
                if p_code in self.images:
                    img = self.images[p_code]
                    # On centre l'image dans le bouton
                    img_rect = img.get_rect(center=(bx + btn_width//2, rect_y + h//2))
                    self.screen.blit(img, img_rect)
            
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    # Est-ce qu'on a cliqué dans le menu ?
                    if rect_y <= my <= rect_y + h and rect_x <= mx <= rect_x + w:
                        # Quel bouton ?
                        idx = (mx - rect_x) // btn_width
                        if 0 <= idx < 4:
                            choice = options[idx]
                            waiting = False
                            
        return choice

    def col_row_to_algebraic(self, col, row):
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        rank = 8 - row 
        return f"{files[col]}{rank}"

    def run(self):
        running = True
        game_over = False
        
        while running:
            # 1. Réponses API
            messages = self.client.get_messages()
            for msg in messages:
                print(f"[API > Python] : {msg}")
                
                # On découpe le message par espace pour voir s'il y a plusieurs infos
                # Ex: "board:rnbq... game_over:checkmate"
                parts = msg.split(' ')
                
                for part in parts:
                    if part.startswith("board:"):
                        raw_board = part.split(":")[1]
                        if len(raw_board) == 64:
                            self.set_board_from_string(raw_board)
                            print("-> Plateau mis à jour.")
                    
                    elif part == "game_over:checkmate":
                        print("!!! ECHEC ET MAT !!!")
                        pygame.display.set_caption("ECHEC ET MAT - Partie Terminée")
                        game_over = True
                        
                    elif part == "game_over:stalemate":
                        print("!!! PAT (Match Nul) !!!")
                        pygame.display.set_caption("PAT - Partie Terminée")
                        game_over = True
                        
                    elif part == "illegal_move_king_check":
                        print("-> Coup illégal (Roi en échec)")

            # 2. Evénements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.client.send_command("quit")
                    running = False
                
                # On interdit de jouer si c'est Game Over
                elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
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
                        
                        # 1. Conversion coordonnées
                        move_str = self.col_row_to_algebraic(start[1], start[0]) + \
                                   self.col_row_to_algebraic(end[1], end[0])
                        
                        # 2. DÉTECTION PROMOTION
                        # On regarde quelle pièce bouge
                        piece_moving = self.board[start[0]][start[1]]
                        
                        # Si c'est un Pion et qu'il va sur la ligne 0 (Blanc) ou 7 (Noir)
                        if piece_moving[1] == 'P':
                            if (piece_moving[0] == 'w' and end[0] == 0) or \
                               (piece_moving[0] == 'b' and end[0] == 7):
                                
                                # Appel du menu graphique (Bloquant)
                                promo_char = self.get_promotion_choice(piece_moving[0])
                                move_str += promo_char # ex: "a7a8" + "n"
                        
                        # 3. Envoi au moteur
                        self.client.send_command(move_str)
                        
                        self.selected_sq = ()
                        self.player_clicks = []

            self.draw_board()
            self.draw_pieces()
            
            # Petit effet visuel si Game Over
            if game_over:
                s = pygame.Surface((WIDTH, HEIGHT))
                s.set_alpha(100) # Transparence
                s.fill((255, 0, 0)) # Rouge
                self.screen.blit(s, (0,0))
                
            pygame.display.flip()
            self.clock.tick(FPS)

        self.client.close()
        pygame.quit()

if __name__ == "__main__":
    g = Game()
    g.run()