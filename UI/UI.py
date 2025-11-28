import pygame
import subprocess
import sys
import threading
import queue
import socket
import time
import os

# =============================================================================
#                               CONFIGURATION
# =============================================================================

BOARD_WIDTH = 600
HEIGHT = 600
PANEL_WIDTH = 250 
WIDTH = BOARD_WIDTH + PANEL_WIDTH

DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
FPS = 15

# Réseau
HOST = "127.0.0.1"
PORT = 12345

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# !!! VÉRIFIEZ BIEN CE CHEMIN !!!
# Si compilé avec "gcc main.c -o engine", c'est "engine" (ou "engine.exe" sur Windows sans WSL)
LINUX_ENGINE_PATH = "/mnt/c/Users/msluc/OneDrive/Projets Info/ChessAI/ChessEngine/API_negamax"

# Couleurs Plateau
COLOR_LIGHT = (234, 235, 200) 
COLOR_DARK = (119, 149, 86)   
COLOR_HIGHLIGHT = (255, 255, 0, 100) # Bleu sélection
COLOR_LAST_MOVE = (255, 170, 0, 150) # Orange dernier coup

# Couleurs UI
COLOR_PANEL = (40, 40, 40)
COLOR_TEXT = (220, 220, 220)
COLOR_BTN = (70, 70, 70)
COLOR_BTN_HOVER = (100, 100, 100)

PIECE_SYMBOLS = {
    'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
    'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k'
}

MAX_RETRIES = 10
RETRY_DELAY = 0.5 

# =============================================================================
#                               CLIENT RESEAU
# =============================================================================

class ChessAPIClient:
    def __init__(self):
        self.sock = None
        self.process = None
        self.q = queue.Queue()
        self.running = True
        self.launch_server()
        # Petit temps pour laisser le serveur C démarrer
        time.sleep(0.5)
        self.connect_to_server()

    def launch_server(self):
        cmd = ["wsl", LINUX_ENGINE_PATH]
        try:
            self.process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
        except FileNotFoundError:
            print(f"[Erreur] Impossible de lancer : {cmd}")

    def connect_to_server(self):
        print(f"[Python] Connexion à {HOST}:{PORT}...")
        for attempt in range(MAX_RETRIES):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((HOST, PORT))
                print(f"[Python] Connecté au moteur !")
                self.listener = threading.Thread(target=self.receive_data)
                self.listener.daemon = True
                self.listener.start()
                return 
            except ConnectionRefusedError:
                if self.sock: self.sock.close()
                time.sleep(RETRY_DELAY)
        print("Impossible de se connecter au moteur C.")
        sys.exit(1)

    def receive_data(self):
        while self.running and self.sock:
            try:
                data = self.sock.recv(4096)
                if not data: break
                full_msg = data.decode('utf-8').strip()
                # Gérer les messages collés
                lines = full_msg.split('\n')
                for line in lines:
                    if line.strip(): self.q.put(line.strip())
            except: break

    def send_command(self, cmd):
        if self.sock:
            try:
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

# =============================================================================
#                               INTERFACE GRAPHIQUE
# =============================================================================

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI Project")
        self.clock = pygame.time.Clock()
        
        # Polices
        self.font = pygame.font.SysFont("Arial", 20) 
        self.font_big = pygame.font.SysFont("Arial", 32, bold=True)
        
        self.images = {}
        self.load_images()
        
        # Plateau vide au départ
        self.board = [["--" for _ in range(8)] for _ in range(8)]
        self.init_standard_board()
        
        # État du jeu
        self.selected_sq = () 
        self.player_clicks = [] 
        self.move_log = []      # Historique textuel
        self.last_move = []     # Coordonnées dernier coup (pour surbrillance)
        
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
        for piece in pieces:
            path = os.path.join(ASSETS_DIR, f"{piece}.png")
            try:
                self.images[piece] = pygame.image.load(path)
                self.images[piece] = pygame.transform.scale(self.images[piece], (SQ_SIZE, SQ_SIZE))
            except: pass

    def draw_text_centered(self, text, y_offset, font_size=40, color=(0, 0, 0)):
        font = pygame.font.SysFont("Arial", font_size, bold=True)
        text_surface = font.render(text, True, color)
        rect = text_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2 + y_offset))
        self.screen.blit(text_surface, rect)

    def show_start_screen(self):
        intro = True
        while intro:
            self.screen.fill(COLOR_LIGHT)
            self.draw_text_centered("CHESS AI PROJECT", -150, 60, (50, 50, 50))
            
            # Boutons centrés (attention WIDTH inclut le panneau, on centre sur l'écran total)
            cx = WIDTH // 2
            
            btn_pvp = pygame.Rect(cx - 150, HEIGHT//2 - 80, 300, 50)
            btn_white = pygame.Rect(cx - 150, HEIGHT//2, 300, 50)
            btn_black = pygame.Rect(cx - 150, HEIGHT//2 + 80, 300, 50)
            
            mx, my = pygame.mouse.get_pos()
            
            for btn, text in [(btn_pvp, "Humain vs Humain"), (btn_white, "IA (Blancs)"), (btn_black, "IA (Noirs)")]:
                col = (150, 150, 150) if btn.collidepoint((mx, my)) else (100, 100, 100)
                pygame.draw.rect(self.screen, col, btn)
                txt_surf = self.font_big.render(text, True, (255,255,255))
                txt_rect = txt_surf.get_rect(center=btn.center)
                self.screen.blit(txt_surf, txt_rect)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.client.close(); pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_pvp.collidepoint(event.pos): return 'pvp', 'w'
                    elif btn_white.collidepoint(event.pos): return 'pve', 'w'
                    elif btn_black.collidepoint(event.pos): return 'pve', 'b'

    def get_promotion_choice(self, color):
        w, h = 300, 100
        rect_x, rect_y = (BOARD_WIDTH - w) // 2, (HEIGHT - h) // 2
        pygame.draw.rect(self.screen, (50, 50, 50), (rect_x, rect_y, w, h))
        options = ['q', 'r', 'b', 'n']
        piece_codes = {'q': 'Q', 'r': 'R', 'b': 'B', 'n': 'N'}
        btn_width = w // 4
        waiting = True; choice = 'q'
        
        while waiting:
            for i, opt in enumerate(options):
                bx = rect_x + i * btn_width
                p_code = (color + piece_codes[opt])
                if p_code in self.images:
                    img = self.images[p_code]
                    img_rect = img.get_rect(center=(bx + btn_width//2, rect_y + h//2))
                    self.screen.blit(img, img_rect)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if rect_y <= my <= rect_y + h and rect_x <= mx <= rect_x + w:
                        idx = (mx - rect_x) // btn_width
                        if 0 <= idx < 4: choice = options[idx]; waiting = False
        return choice

    def set_board_from_string(self, board_str):
        char_to_piece = {'P':'wP','N':'wN','B':'wB','R':'wR','Q':'wQ','K':'wK','p':'bP','n':'bN','b':'bB','r':'bR','q':'bQ','k':'bK','-':'--'}
        limit = min(len(board_str), 64)
        for i in range(limit):
            row, col = i // 8, i % 8
            if board_str[i] in char_to_piece: self.board[row][col] = char_to_piece[board_str[i]]

    def parse_move_to_indices(self, move_str):
        """Convertit 'e2e4' en [(6,4), (4,4)]"""
        if len(move_str) < 4: return []
        col_start = ord(move_str[0]) - ord('a')
        col_end = ord(move_str[2]) - ord('a')
        row_start = 8 - int(move_str[1])
        row_end = 8 - int(move_str[3])
        return [(row_start, col_start), (row_end, col_end)]

    def col_row_to_algebraic(self, col, row):
        return f"{chr(ord('a')+col)}{8-row}"

    def draw_sidebar(self, turn):
        # Fond
        pygame.draw.rect(self.screen, COLOR_PANEL, (BOARD_WIDTH, 0, PANEL_WIDTH, HEIGHT))
        
        # Titre
        title = self.font_big.render("Historique", True, COLOR_TEXT)
        self.screen.blit(title, (BOARD_WIDTH + 20, 20))
        
        # Liste des coups (Affiche les 20 derniers)
        y = 70
        start_index = max(0, len(self.move_log) - 20)
        
        for i, move in enumerate(self.move_log[start_index:]): 
            num = start_index + i + 1
            # Blancs/Noirs
            color = (255, 255, 255) if num % 2 != 0 else (150, 150, 150)
            
            txt = f"{num}. {move}"
            text_surf = self.font.render(txt, True, color)
            self.screen.blit(text_surf, (BOARD_WIDTH + 20, y))
            y += 25

        # Bouton UNDO
        btn_undo = pygame.Rect(BOARD_WIDTH + 25, HEIGHT - 80, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        col = COLOR_BTN_HOVER if btn_undo.collidepoint(mouse_pos) else COLOR_BTN
        pygame.draw.rect(self.screen, col, btn_undo)
        
        undo_txt = self.font_big.render("ANNULER", True, (255, 255, 255))
        txt_rect = undo_txt.get_rect(center=btn_undo.center)
        self.screen.blit(undo_txt, txt_rect)
        
        return btn_undo

    def draw_board(self, orientation='w'):
        colors = [COLOR_LIGHT, COLOR_DARK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[((r + c) % 2)]
                draw_r, draw_c = (r, c) if orientation == 'w' else (7-r, 7-c)
                pygame.draw.rect(self.screen, color, pygame.Rect(draw_c*SQ_SIZE, draw_r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
                
                # Surbrillance dernier coup
                if self.last_move:
                    if (r, c) in self.last_move:
                        s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                        s.set_alpha(150)
                        s.fill(COLOR_LAST_MOVE)
                        self.screen.blit(s, (draw_c*SQ_SIZE, draw_r*SQ_SIZE))

                # Surbrillance sélection
                if self.selected_sq == (r, c):
                    s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                    s.set_alpha(100)
                    s.fill(COLOR_HIGHLIGHT)
                    self.screen.blit(s, (draw_c*SQ_SIZE, draw_r*SQ_SIZE))

    def draw_pieces(self, orientation='w'):
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                piece = self.board[r][c]
                if piece != "--":
                    draw_r, draw_c = (r, c) if orientation == 'w' else (7-r, 7-c)
                    if piece in self.images:
                        self.screen.blit(self.images[piece], pygame.Rect(draw_c*SQ_SIZE, draw_r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
                    else:
                        color = (0, 0, 0) if piece[0] == 'w' else (50, 50, 50)
                        text = self.font.render(PIECE_SYMBOLS.get(piece, "?"), True, color)
                        text_rect = text.get_rect(center=(draw_c*SQ_SIZE + SQ_SIZE//2, draw_r*SQ_SIZE + SQ_SIZE//2))
                        self.screen.blit(text, text_rect)

    def run(self):
        game_mode, player_color = self.show_start_screen()
        
        self.screen.fill((0,0,0)) 
        self.draw_board(player_color)
        pygame.display.flip()

        running = True
        turn = 'w' 
        ai_thinking = False
        
        while running:
            messages = self.client.get_messages()
            for msg in messages:
                print(f"[API] {msg}")
                parts = msg.split(' ')
                for part in parts:
                    if part.startswith("board:"):
                        raw_board = part.split(":")[1]
                        if len(raw_board) == 64:
                            self.set_board_from_string(raw_board)
                            turn = 'b' if turn == 'w' else 'w'
                            ai_thinking = False
                            
                    elif part.startswith("bestmove:"):
                        move = part.split(":")[1]
                        if move != "none":
                            self.move_log.append(move)
                            self.last_move = self.parse_move_to_indices(move)
                            
                    elif "game_over" in part:
                        reason = part.split(":")[1]
                        if reason == "draw_repetition":
                            print("!!! MATCH NUL (3 Répétitions) !!!")
                        else:
                            print(f"!!! FIN : {reason} !!!")
                        ai_thinking = True
                    
                    # ### CORRECTION 1 : GESTION DES COUPS ILLÉGAUX ###
                    elif "illegal" in part:
                        print("-> Coup annulé (Illégal)")
                        # On retire le coup optimiste qu'on avait ajouté
                        if self.move_log: 
                            self.move_log.pop()
                        
                        # On remet la surbrillance sur le VRAI dernier coup valide
                        if self.move_log:
                            self.last_move = self.parse_move_to_indices(self.move_log[-1])
                        else:
                            self.last_move = []
                    # ################################################

            # IA Automatique
            if game_mode == 'pve' and turn != player_color and not ai_thinking:
                ai_thinking = True
                pygame.time.delay(100)
                self.client.send_command("go")

            # Dessin
            self.draw_board(player_color)
            self.draw_pieces(player_color)
            btn_undo = self.draw_sidebar(turn)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.client.send_command("quit"); running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_undo.collidepoint(event.pos):
                        if not ai_thinking:
                            print("Undo demandé...")
                            undo_count = 2 if game_mode == 'pve' else 1
                            
                            # On vérifie qu'on peut annuler
                            if len(self.move_log) >= undo_count:
                                for _ in range(undo_count):
                                    # 1. On dit au moteur d'annuler
                                    self.client.send_command("undo")
                                    
                                    # 2. On retire le dernier coup de la liste visuelle
                                    if self.move_log: 
                                        self.move_log.pop()
                                    
                                    # 3. On change le tour
                                    turn = 'b' if turn == 'w' else 'w'
                                
                                # 4. Mise à jour de la surbrillance orange (Coup précédent)
                                if self.move_log:
                                    # On regarde le dernier coup restant dans la liste
                                    last_move_str = self.move_log[-1]
                                    self.last_move = self.parse_move_to_indices(last_move_str)
                                else:
                                    # Si on est revenu au début
                                    self.last_move = []
                                
                                # Petit délai pour laisser le C répondre "board:..."
                                time.sleep(0.1)
                        continue

                    # Jeu
                    if event.pos[0] > BOARD_WIDTH: continue
                    if (game_mode == 'pve' and turn != player_color) or ai_thinking: continue

                    col = event.pos[0] // SQ_SIZE
                    row = event.pos[1] // SQ_SIZE
                    if player_color == 'b': col, row = 7 - col, 7 - row
                    
                    if self.selected_sq == (row, col):
                        self.selected_sq = (); self.player_clicks = []
                    else:
                        self.selected_sq = (row, col); self.player_clicks.append(self.selected_sq)
                    
                    if len(self.player_clicks) == 2:
                        start, end = self.player_clicks[0], self.player_clicks[1]
                        move_str = self.col_row_to_algebraic(start[1], start[0]) + \
                                   self.col_row_to_algebraic(end[1], end[0])
                        
                        piece_moving = self.board[start[0]][start[1]]
                        if piece_moving[1] == 'P':
                            if (piece_moving[0] == 'w' and end[0] == 0) or \
                               (piece_moving[0] == 'b' and end[0] == 7):
                                move_str += self.get_promotion_choice(piece_moving[0])
                        
                        self.client.send_command(move_str)
                        
                        # On ajoute tout de suite (optimiste)
                        self.move_log.append(move_str)
                        self.last_move = [start, end]
                        
                        self.selected_sq = (); self.player_clicks = []

            self.clock.tick(FPS)

        self.client.close(); pygame.quit()

if __name__ == "__main__":
    g = Game()
    g.run()