import pygame
import subprocess
import sys
import threading
import queue
import socket
import time
import os
import chess # Bibliothèque python-chess pour gérer le FEN pour l'IA

# =============================================================================
#                               CONFIGURATION
# =============================================================================

# --- CHEMINS (A ADAPTER SELON TA CONFIGURATION) ---
# Chemin vers ton exécutable compilé C
# Si tu es sous Windows avec WSL, garde le /mnt/c/...
LINUX_ENGINE_PATH = "../ChessEngine/API_negamax"

# Ajout du dossier DeepLearning au path pour trouver dl_ai_player.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Remonte d'un cran (ChessProject/)
DL_SRC_DIR = os.path.join(PROJECT_ROOT, "DeepLearning", "src")
sys.path.append(DL_SRC_DIR)

# Import de ton IA
try:
    from dl_ai_player import get_dl_move
    AI_AVAILABLE = True
    print("[INFO] Module DeepLearning chargé avec succès.")
except ImportError:
    AI_AVAILABLE = False
    print("[ATTENTION] Impossible de charger dl_ai_player. Vérifiez les chemins.")

# Dimensions
BOARD_WIDTH = 600
HEIGHT = 600
PANEL_WIDTH = 250 
WIDTH = BOARD_WIDTH + PANEL_WIDTH
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
FPS = 30

# Réseau
HOST = "127.0.0.1"
PORT = 12345

# Couleurs
COLOR_LIGHT = (234, 235, 200) 
COLOR_DARK = (119, 149, 86)   
COLOR_HIGHLIGHT = (255, 255, 0, 100) 
COLOR_LAST_MOVE = (255, 170, 0, 150) 

COLOR_PANEL = (40, 40, 40)
COLOR_TEXT = (220, 220, 220)
COLOR_BTN = (70, 70, 70)
COLOR_BTN_HOVER = (100, 100, 100)

ASSETS_DIR = os.path.join(BASE_DIR, "assets")

PIECE_SYMBOLS = {
    'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
    'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k'
}

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
        time.sleep(1) # Attente démarrage serveur C
        self.connect_to_server()

    def launch_server(self):
        # On sépare le chemin pour gérer les espaces correctement si nécessaire
        # Mais avec subprocess et une liste, les espaces sont souvent gérés automatiquement.
        # Vérifions d'abord si le fichier existe côté Windows pour éviter une erreur silencieuse
        
        # Note : os.path.exists vérifie le chemin Windows (C:\...), pas le chemin WSL (/mnt/c/...)
        # On fait confiance au chemin donné.
        
        print(f"[PYTHON] Lancement de : wsl {LINUX_ENGINE_PATH}")
        cmd = ["wsl", LINUX_ENGINE_PATH]
        
        try:
            # On utilise shell=False (par défaut) pour que la liste d'arguments gère les espaces
            self.process = subprocess.Popen(cmd, 
                                          stdin=subprocess.PIPE, # Important pour éviter les conflits d'entrée
                                          stdout=subprocess.PIPE, # On pourrait lire la sortie pour débugger
                                          stderr=subprocess.PIPE)
            print("[PYTHON] Processus C démarré.")
        except FileNotFoundError:
            print(f"[ERREUR CRITIQUE] Impossible de trouver 'wsl' ou le fichier.")
        except Exception as e:
            print(f"[ERREUR LANCEMENT] {e}")

    def connect_to_server(self):
        print(f"[PYTHON] Tentative de connexion à {HOST}:{PORT}...")
        for attempt in range(10):
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((HOST, PORT))
                print(f"[PYTHON] Connecté au moteur !")
                self.listener = threading.Thread(target=self.receive_data)
                self.listener.daemon = True
                self.listener.start()
                return 
            except ConnectionRefusedError:
                if self.sock: self.sock.close()
                time.sleep(0.5)
        print("[ERREUR] Échec connexion au moteur C.")

    def receive_data(self):
        while self.running and self.sock:
            try:
                data = self.sock.recv(4096)
                if not data: break
                full_msg = data.decode('utf-8').strip()
                lines = full_msg.split('\n')
                for line in lines:
                    if line.strip(): self.q.put(line.strip())
            except: break

    def send_command(self, cmd):
        if self.sock:
            try:
                # Ajout du saut de ligne pour que le C détecte la fin de commande
                msg = cmd + "\n"
                self.sock.sendall(msg.encode('utf-8'))
            except Exception as e:
                print(f"[ERREUR ENVOI] {e}")

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
        pygame.display.set_caption("Chess AI - Deep Learning Edition")
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont("Arial", 18) 
        self.font_big = pygame.font.SysFont("Arial", 28, bold=True)
        
        self.images = {}
        self.load_images()
        
        # Plateau visuel (Tableau 8x8 de strings)
        self.visual_board = [["--" for _ in range(8)] for _ in range(8)]
        
        # Plateau Logique (python-chess) pour générer les FEN corrects pour l'IA
        self.py_board = chess.Board() 
        
        self.init_standard_visual_board()
        
        # État du jeu
        self.selected_sq = () 
        self.player_clicks = [] 
        self.move_log = []      
        self.last_move = []     
        
        self.client = ChessAPIClient()

    def init_standard_visual_board(self):
        # Initialisation manuelle pour l'affichage avant connexion
        self.visual_board = [
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
                img = pygame.image.load(path)
                self.images[piece] = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
            except: 
                print(f"[WARN] Image manquante : {piece}")

    def show_start_screen(self):
        intro = True
        while intro:
            self.screen.fill(COLOR_LIGHT)
            # Titre
            title = self.font_big.render("CHESS AI PROJECT", True, (50, 50, 50))
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 100))
            
            # Boutons
            cx = WIDTH // 2
            btn_pvp = pygame.Rect(cx - 150, HEIGHT//2 - 60, 300, 50)
            btn_white = pygame.Rect(cx - 150, HEIGHT//2, 300, 50) # Jouer Blanc vs IA
            btn_black = pygame.Rect(cx - 150, HEIGHT//2 + 60, 300, 50) # Jouer Noir vs IA
            
            mx, my = pygame.mouse.get_pos()
            
            for btn, text, mode in [
                (btn_pvp, "Humain vs Humain", ('pvp', 'w')), 
                (btn_white, "Jouer BLANCS vs IA", ('pve', 'w')), 
                (btn_black, "Jouer NOIRS vs IA", ('pve', 'b'))
            ]:
                col = (180, 180, 180) if btn.collidepoint((mx, my)) else (140, 140, 140)
                pygame.draw.rect(self.screen, col, btn)
                txt_surf = self.font_big.render(text, True, (255,255,255))
                txt_rect = txt_surf.get_rect(center=btn.center)
                self.screen.blit(txt_surf, txt_rect)

            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.client.close(); pygame.quit(); sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_pvp.collidepoint(event.pos): return ('pvp', 'w')
                    elif btn_white.collidepoint(event.pos): return ('pve', 'w')
                    elif btn_black.collidepoint(event.pos): return ('pve', 'b')

    def get_promotion_choice(self, color):
        # Simple popup pour promotion
        return 'q' # Par défaut Reine pour simplifier l'exemple

    def update_visual_board_from_string(self, board_str):
        # Met à jour le plateau visuel depuis le string envoyé par le C
        char_to_piece = {'P':'wP','N':'wN','B':'wB','R':'wR','Q':'wQ','K':'wK','p':'bP','n':'bN','b':'bB','r':'bR','q':'bQ','k':'bK','-':'--'}
        limit = min(len(board_str), 64)
        for i in range(limit):
            row, col = i // 8, i % 8
            if board_str[i] in char_to_piece: self.visual_board[row][col] = char_to_piece[board_str[i]]

    def parse_move_to_indices(self, move_str):
        if len(move_str) < 4: return []
        col_start = ord(move_str[0]) - ord('a')
        col_end = ord(move_str[2]) - ord('a')
        row_start = 8 - int(move_str[1])
        row_end = 8 - int(move_str[3])
        return [(row_start, col_start), (row_end, col_end)]

    def col_row_to_algebraic(self, col, row):
        return f"{chr(ord('a')+col)}{8-row}"

    def draw_sidebar(self, turn):
        pygame.draw.rect(self.screen, COLOR_PANEL, (BOARD_WIDTH, 0, PANEL_WIDTH, HEIGHT))
        
        # Titre
        title = self.font_big.render("Historique", True, COLOR_TEXT)
        self.screen.blit(title, (BOARD_WIDTH + 20, 20))
        
        # Liste des coups
        y = 70
        start_index = max(0, len(self.move_log) - 18)
        for i, move in enumerate(self.move_log[start_index:]): 
            color = (255, 255, 255) if (start_index + i) % 2 == 0 else (170, 170, 170)
            txt = f"{start_index + i + 1}. {move}"
            text_surf = self.font.render(txt, True, color)
            self.screen.blit(text_surf, (BOARD_WIDTH + 20, y))
            y += 25

        # Bouton UNDO
        btn_undo = pygame.Rect(BOARD_WIDTH + 25, HEIGHT - 70, 200, 50)
        mouse_pos = pygame.mouse.get_pos()
        col = COLOR_BTN_HOVER if btn_undo.collidepoint(mouse_pos) else COLOR_BTN
        pygame.draw.rect(self.screen, col, btn_undo)
        undo_txt = self.font_big.render("ANNULER", True, (255, 255, 255))
        self.screen.blit(undo_txt, undo_txt.get_rect(center=btn_undo.center))
        
        return btn_undo

    def draw_board(self, orientation='w'):
        colors = [COLOR_LIGHT, COLOR_DARK]
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                color = colors[((r + c) % 2)]
                draw_r, draw_c = (r, c) if orientation == 'w' else (7-r, 7-c)
                pygame.draw.rect(self.screen, color, pygame.Rect(draw_c*SQ_SIZE, draw_r*SQ_SIZE, SQ_SIZE, SQ_SIZE))
                
                # Highlights
                if self.last_move:
                    if (r, c) in self.last_move:
                        s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                        s.set_alpha(150); s.fill(COLOR_LAST_MOVE)
                        self.screen.blit(s, (draw_c*SQ_SIZE, draw_r*SQ_SIZE))

                if self.selected_sq == (r, c):
                    s = pygame.Surface((SQ_SIZE, SQ_SIZE))
                    s.set_alpha(100); s.fill(COLOR_HIGHLIGHT)
                    self.screen.blit(s, (draw_c*SQ_SIZE, draw_r*SQ_SIZE))

    def draw_pieces(self, orientation='w'):
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                piece = self.visual_board[r][c]
                if piece != "--":
                    draw_r, draw_c = (r, c) if orientation == 'w' else (7-r, 7-c)
                    if piece in self.images:
                        self.screen.blit(self.images[piece], pygame.Rect(draw_c*SQ_SIZE, draw_r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    # =========================================================================
    #                               BOUCLE DE JEU
    # =========================================================================
    def run(self):
        game_mode, player_color = self.show_start_screen()
        
        # Si on joue Blanc, l'IA est 'b', sinon 'w'
        ai_color = 'b' if player_color == 'w' else 'w'
        
        running = True
        turn = 'w' 
        ai_thinking = False
        
        while running:
            # 1. Gestion Réseau (Mise à jour état)
            messages = self.client.get_messages()
            for msg in messages:
                print(f"[API] {msg}")
                if "illegal" in msg:
                    # Coup refusé par le C : on annule visuellement
                    print("Coup illégal détecté !")
                    if self.move_log: 
                        cancelled_move = self.move_log.pop()
                        try: self.py_board.pop() # Annule aussi dans python-chess
                        except: pass
                    
                    if self.move_log:
                        self.last_move = self.parse_move_to_indices(self.move_log[-1])
                    else:
                        self.last_move = []
                    ai_thinking = False

                parts = msg.split(' ')
                for part in parts:
                    if part.startswith("board:"):
                        raw_board = part.split(":")[1]
                        if len(raw_board) == 64:
                            self.update_visual_board_from_string(raw_board)
                            # Changement de tour logique
                            turn = 'b' if turn == 'w' else 'w'
                            if turn == player_color: ai_thinking = False
                            
                    elif "game_over" in part:
                        print(f"FIN: {part}")
                        ai_thinking = True 

            # 2. IA DEEP LEARNING (Si c'est son tour)
            if game_mode == 'pve' and turn == ai_color and not ai_thinking:
                ai_thinking = True
                
                # On utilise python-chess pour générer le FEN parfait pour le réseau
                fen = self.py_board.fen()
                print(f"[IA] Réfléchit sur : {fen}")
                
                # --- APPEL AU MODELE ---
                if AI_AVAILABLE:
                    # L'IA nous donne un coup algébrique (ex: e7e5)
                    # On utilise threading pour ne pas figer l'interface
                    def ai_thread_func():
                        try:
                            # Petit délai pour voir l'action
                            time.sleep(0.5) 
                            ai_move_san = get_dl_move(fen) # Peut retourner SAN (e5) ou UCI (e7e5)?
                            
                            # Conversion si nécessaire
                            # get_dl_move retourne du SAN (e.g. "Nf3"), on le convertit en UCI pour le moteur C
                            move_obj = self.py_board.parse_san(ai_move_san)
                            uci_move = move_obj.uci()
                            
                            print(f"[IA] Joue : {uci_move}")
                            self.client.send_command(uci_move)
                            self.move_log.append(uci_move)
                            self.last_move = self.parse_move_to_indices(uci_move)
                            
                            # Mise à jour synchronisée du board Python
                            self.py_board.push(move_obj)
                            
                        except Exception as e:
                            print(f"[ERREUR IA] {e}")

                    t = threading.Thread(target=ai_thread_func)
                    t.start()
                else:
                    print("[ERREUR] IA non disponible.")

            # 3. Dessin
            self.screen.fill((0,0,0)) 
            self.draw_board(player_color)
            self.draw_pieces(player_color)
            btn_undo = self.draw_sidebar(turn)
            pygame.display.flip()

            # 4. Gestion Événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.client.send_command("quit"); running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # --- UNDO ---
                    if btn_undo.collidepoint(event.pos):
                        if not ai_thinking:
                            print("Undo...")
                            undo_cnt = 2 if game_mode == 'pve' else 1
                            if len(self.move_log) >= undo_cnt:
                                for _ in range(undo_cnt):
                                    self.client.send_command("undo")
                                    if self.move_log: self.move_log.pop()
                                    try: self.py_board.pop()
                                    except: pass
                                    turn = 'b' if turn == 'w' else 'w'
                                
                                if self.move_log:
                                    self.last_move = self.parse_move_to_indices(self.move_log[-1])
                                else:
                                    self.last_move = []
                        continue

                    # --- JEU HUMAIN ---
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
                        
                        # Promotion automatique Reine pour simplifier l'UI
                        piece_moving = self.visual_board[start[0]][start[1]]
                        if piece_moving[1] == 'P':
                            if (piece_moving[0] == 'w' and end[0] == 0) or \
                               (piece_moving[0] == 'b' and end[0] == 7):
                                move_str += 'q'
                        
                        # Tentative de jouer le coup sur le board Python pour vérifier la légalité basique
                        try:
                            move_obj = chess.Move.from_uci(move_str)
                            if move_obj in self.py_board.legal_moves:
                                self.py_board.push(move_obj)
                                self.client.send_command(move_str)
                                self.move_log.append(move_str)
                                self.last_move = [start, end]
                            else:
                                print(f"Coup illégal (Python Check): {move_str}")
                        except:
                            # Si c'est une promotion complexe non gérée, on envoie quand même au C
                            self.client.send_command(move_str)

                        self.selected_sq = (); self.player_clicks = []

            self.clock.tick(FPS)

        self.client.close(); pygame.quit()

if __name__ == "__main__":
    g = Game()
    g.run()