import pygame
import subprocess
import sys
import threading
import queue
import socket
import time
import os
import chess 

# =============================================================================
#                               CONFIGURATION
# =============================================================================

# --- CHEMINS (ADAPTE SELON TA CONFIG) ---
LINUX_ENGINE_PATH = "/mnt/c/Users/msluc/OneDrive/Projets Info/ChessAI/ChessEngine/API_negamax"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DL_SRC_DIR = os.path.join(PROJECT_ROOT, "DeepLearning", "src")
sys.path.append(DL_SRC_DIR)

# Dimensions & Couleurs
BOARD_WIDTH = 600
HEIGHT = 600
PANEL_WIDTH = 250 
WIDTH = BOARD_WIDTH + PANEL_WIDTH
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
FPS = 30
HOST = "127.0.0.1"
PORT = 12345

COLOR_LIGHT = (234, 235, 200) 
COLOR_DARK = (119, 149, 86)   
COLOR_HIGHLIGHT = (255, 255, 0, 100) 
COLOR_LAST_MOVE = (255, 170, 0, 150) 
COLOR_PANEL = (40, 40, 40)
COLOR_TEXT = (220, 220, 220)
COLOR_BTN = (70, 70, 70)
COLOR_BTN_HOVER = (100, 100, 100)
COLOR_DROPDOWN_BG = (255, 255, 255)
COLOR_DROPDOWN_HOVER = (200, 200, 255)

ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# =============================================================================
#                               WIDGETS
# =============================================================================

class Dropdown:
    def __init__(self, x, y, w, h, font, main_color, hover_color, options):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.main_color = main_color
        self.hover_color = hover_color
        self.options = options
        self.selected_index = 0
        self.is_open = False
        self.active_option = -1
        
    def draw(self, surface):
        pygame.draw.rect(surface, self.main_color, self.rect)
        pygame.draw.rect(surface, (0,0,0), self.rect, 2)
        msg = self.font.render(self.options[self.selected_index], 1, (0,0,0))
        surface.blit(msg, msg.get_rect(center=self.rect.center))
        
        if self.is_open:
            for i, option in enumerate(self.options):
                rect = pygame.Rect(self.rect.x, self.rect.y + (i+1)*self.rect.height, self.rect.width, self.rect.height)
                color = self.hover_color if i == self.active_option else self.main_color
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (0,0,0), rect, 1)
                msg = self.font.render(option, 1, (0,0,0))
                surface.blit(msg, msg.get_rect(center=rect.center))

    def update(self, event_list):
        mpos = pygame.mouse.get_pos()
        self.active_option = -1
        if self.is_open:
            for i in range(len(self.options)):
                rect = pygame.Rect(self.rect.x, self.rect.y + (i+1)*self.rect.height, self.rect.width, self.rect.height)
                if rect.collidepoint(mpos):
                    self.active_option = i
                    break
        for event in event_list:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.is_open:
                    if self.active_option != -1:
                        self.selected_index = self.active_option
                        self.is_open = False
                    elif not self.rect.collidepoint(mpos):
                        self.is_open = False
                    else:
                        self.is_open = not self.is_open
                else:
                    if self.rect.collidepoint(mpos):
                        self.is_open = not self.is_open

    def get_selected(self):
        return self.selected_index, self.options[self.selected_index]

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
        time.sleep(1)
        self.connect_to_server()

    def launch_server(self):
        print(f"[PYTHON] Lancement de : wsl {LINUX_ENGINE_PATH}")
        try:
            cmd = ["wsl", LINUX_ENGINE_PATH]
        except:
            cmd = [LINUX_ENGINE_PATH]
        
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("[PYTHON] Processus C démarré.")
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
        pygame.display.set_caption("Chess AI - Project")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18) 
        self.font_big = pygame.font.SysFont("Arial", 28, bold=True)
        
        self.images = {}
        self.load_images()
        self.visual_board = [["--" for _ in range(8)] for _ in range(8)]
        self.py_board = chess.Board() 
        self.init_standard_visual_board()
        
        self.selected_sq = () 
        self.player_clicks = [] 
        self.move_log = []      
        self.last_move = []     
        
        self.client = ChessAPIClient()
        
        # Fonction d'IA active (sera chargée dynamiquement)
        self.active_ai_function = None

    def init_standard_visual_board(self):
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
            except: pass

    def show_start_screen(self):
        intro = True
        
        # --- MISE A JOUR DES OPTIONS ---
        opponents = [
            "Humain vs Humain", 
            "IA C (Minimax)", 
            "IA Simple (CNN)", 
            "IA Standard (ResNet)",
            "IA Avancée (SE-ResNet)", # NOUVEAU CHOIX
            "IA Experte (AlphaZero)",
            "IA Génétique (TinyNet)"
        ]
        dd_opponent = Dropdown(WIDTH//2 - 150, 200, 300, 40, self.font, COLOR_DROPDOWN_BG, COLOR_DROPDOWN_HOVER, opponents)
        
        colors = ["Je joue les Blancs", "Je joue les Noirs"]
        dd_color = Dropdown(WIDTH//2 - 150, 300, 300, 40, self.font, COLOR_DROPDOWN_BG, COLOR_DROPDOWN_HOVER, colors)
        
        btn_play = pygame.Rect(WIDTH//2 - 100, 450, 200, 60)

        while intro:
            self.screen.fill(COLOR_LIGHT)
            title = self.font_big.render("CONFIGURATION DE LA PARTIE", True, (50, 50, 50))
            self.screen.blit(title, (WIDTH//2 - title.get_width()//2, 80))
            
            lbl_opp = self.font.render("Choisir l'adversaire :", True, (50, 50, 50))
            self.screen.blit(lbl_opp, (WIDTH//2 - 150, 175))
            
            lbl_col = self.font.render("Choisir votre couleur :", True, (50, 50, 50))
            self.screen.blit(lbl_col, (WIDTH//2 - 150, 275))

            event_list = pygame.event.get()
            for event in event_list:
                if event.type == pygame.QUIT:
                    self.client.close(); pygame.quit(); sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_play.collidepoint(event.pos):
                        idx_opp, str_opp = dd_opponent.get_selected()
                        idx_col, str_col = dd_color.get_selected()
                        
                        # --- MAPPING MIS A JOUR ---
                        mode = 'pvp'
                        if idx_opp == 1: mode = 'pve_c'
                        elif idx_opp == 2: mode = 'pve_cnn'
                        elif idx_opp == 3: mode = 'pve_resnet'
                        elif idx_opp == 4: mode = 'pve_seresnet' # Nouveau cas
                        elif idx_opp == 5: mode = 'pve_alphazero'
                        elif idx_opp == 6: mode = 'pve_genetic'
                        
                        color = 'w' if idx_col == 0 else 'b'
                        
                        self.load_ai_model(mode)
                        return mode, color

            dd_opponent.update(event_list)
            dd_color.update(event_list)

            mx, my = pygame.mouse.get_pos()
            col_play = (100, 200, 100) if btn_play.collidepoint((mx, my)) else (150, 150, 150)
            pygame.draw.rect(self.screen, col_play, btn_play)
            txt_play = self.font_big.render("JOUER", True, (255, 255, 255))
            self.screen.blit(txt_play, txt_play.get_rect(center=btn_play.center))

            dd_color.draw(self.screen)
            dd_opponent.draw(self.screen)
            pygame.display.update()

    def load_ai_model(self, mode):
        """Charge dynamiquement le bon modèle IA"""
        self.active_ai_function = None
        
        if mode == 'pve_cnn':
            print("[UI] Chargement de l'IA CNN...")
            try:
                from dl_ai_player import get_dl_move
                self.active_ai_function = get_dl_move
            except ImportError: print("[ERREUR] Impossible de charger IA CNN")
            
        elif mode == 'pve_resnet':
            print("[UI] Chargement de l'IA ResNet (Standard)...")
            try:
                # Celui-ci utilise USE_SE=False
                from dl_ai_player_resnet import get_resnet_move
                self.active_ai_function = get_resnet_move
            except ImportError: print("[ERREUR] Impossible de charger IA ResNet")

        elif mode == 'pve_seresnet':
            print("[UI] Chargement de l'IA SE-ResNet (Avancée)...")
            try:
                # Celui-ci utilise USE_SE=True
                # J'utilise 'as' pour renommer la fonction importée et éviter les confusions
                from dl_ai_player_seresnet import get_resnet_move as get_seresnet_move
                self.active_ai_function = get_seresnet_move 
            except ImportError: print("[ERREUR] Impossible de charger IA SE-ResNet")
            
        elif mode == 'pve_alphazero':
            print("[UI] Chargement de l'IA AlphaZero...")
            try:
                from dl_ai_player_alphazero import get_alphazero_move
                self.active_ai_function = get_alphazero_move
            except ImportError: print("[ERREUR] Impossible de charger IA AlphaZero")

        elif mode == 'pve_genetic':
            print("[UI] Chargement de l'IA Génétique...")
            try:
                from dl_ai_player_genetic import get_genetic_move
                self.active_ai_function = get_genetic_move
            except ImportError as e: print(f"[ERREUR] Impossible de charger IA Génétique: {e}")

    def update_visual_board_from_string(self, board_str):
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
        title = self.font_big.render("Historique", True, COLOR_TEXT)
        self.screen.blit(title, (BOARD_WIDTH + 20, 20))
        y = 70
        start_index = max(0, len(self.move_log) - 18)
        for i, move in enumerate(self.move_log[start_index:]): 
            color = (255, 255, 255) if (start_index + i) % 2 == 0 else (170, 170, 170)
            txt = f"{start_index + i + 1}. {move}"
            self.screen.blit(self.font.render(txt, True, color), (BOARD_WIDTH + 20, y))
            y += 25
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
                if self.last_move:
                    if (r, c) in self.last_move:
                        s = pygame.Surface((SQ_SIZE, SQ_SIZE)); s.set_alpha(150); s.fill(COLOR_LAST_MOVE)
                        self.screen.blit(s, (draw_c*SQ_SIZE, draw_r*SQ_SIZE))
                if self.selected_sq == (r, c):
                    s = pygame.Surface((SQ_SIZE, SQ_SIZE)); s.set_alpha(100); s.fill(COLOR_HIGHLIGHT)
                    self.screen.blit(s, (draw_c*SQ_SIZE, draw_r*SQ_SIZE))

    def draw_pieces(self, orientation='w'):
        for r in range(DIMENSION):
            for c in range(DIMENSION):
                piece = self.visual_board[r][c]
                if piece != "--" and piece in self.images:
                    draw_r, draw_c = (r, c) if orientation == 'w' else (7-r, 7-c)
                    self.screen.blit(self.images[piece], pygame.Rect(draw_c*SQ_SIZE, draw_r*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def sync_board_with_engine(self, engine_board_str):
        """
        Déduit le coup joué par le moteur en comparant la string reçue 
        avec tous les coups légaux possibles sur le plateau interne.
        """
        # 1. On nettoie la string reçue
        engine_board_str = engine_board_str.strip()
        
        # 2. On cherche quel coup légal transforme notre plateau actuel en ce plateau cible
        found_move = None
        
        for move in self.py_board.legal_moves:
            self.py_board.push(move) # On simule le coup
            
            # On génère la string correspondante à ce coup simulé
            # (On doit reproduire la logique de formatage de ton moteur C: ligne par ligne, haut vers bas)
            generated_str = []
            for row in range(8): # 0 (A8..H8) à 7 (A1..H1)
                for col in range(8):
                    square = chess.square(col, 7 - row) # Conversion coordonnées visuelles -> python-chess
                    piece = self.py_board.piece_at(square)
                    generated_str.append(piece.symbol() if piece else "-")
            
            generated_check = "".join(generated_str)
            
            # Si ça matche, c'est que c'est le bon coup !
            if generated_check == engine_board_str:
                found_move = move
                break # On garde le coup poussé (on ne pop pas)
            
            self.py_board.pop() # Ce n'était pas le bon coup, on annule
            
        return found_move

    def run(self):
        game_mode, player_color = self.show_start_screen()
        ai_color = 'b' if player_color == 'w' else 'w'
        running = True
        turn = 'w' 
        ai_thinking = False
        
        while running:
            # 1. Gestion Réseau
            messages = self.client.get_messages()
            for msg in messages:
                print(f"[API] {msg}")
                
                parts = msg.split(' ')
                for part in parts:
                    if part.startswith("board:"):
                        board_data = part.split(":")[1]
                        
                        # --- CORRECTION ICI ---
                        # 1. Mettre à jour le visuel (comme avant)
                        self.update_visual_board_from_string(board_data)
                        
                        # 2. Mettre à jour la logique interne (Sync)
                        # On ne le fait que si c'est au tour de l'IA (pour éviter les conflits si update humain)
                        if turn == ai_color:
                            played_move = self.sync_board_with_engine(board_data)
                            
                            if played_move:
                                uci = played_move.uci()
                                print(f"[SYNC] Moteur C a joué : {uci}")
                                self.move_log.append(uci)
                                self.last_move = self.parse_move_to_indices(uci)
                                # self.py_board est déjà mis à jour par sync_board_with_engine (il ne pop pas si trouvé)
                            else:
                                print("[ERREUR SYNC] Impossible de déduire le coup du moteur !")
                        
                        # 3. Changement de tour
                        turn = 'b' if turn == 'w' else 'w'
                        if turn == player_color: ai_thinking = False
                        
                    elif "game_over" in part: ai_thinking = True 
                    elif "illegal" in part:
                        # Gestion des coups illégaux (rare avec l'IA, possible pour l'humain)
                        if self.move_log: 
                            self.move_log.pop(); 
                            try: self.py_board.pop() 
                            except: pass
                        self.last_move = self.parse_move_to_indices(self.move_log[-1]) if self.move_log else []
                        ai_thinking = False

            # 2. IA LOGIQUE
            if game_mode != 'pvp' and turn == ai_color and not ai_thinking:
                ai_thinking = True
                
                # --- CAS A : IA Deep Learning ---
                if game_mode in ['pve_cnn', 'pve_resnet', 'pve_seresnet', 'pve_alphazero', 'pve_genetic']:
                    fen = self.py_board.fen()
                    if self.active_ai_function:
                        def ai_thread_func():
                            try:
                                time.sleep(0.5)
                                # Hybridation (Minimax en finale < 12 pièces)
                                if len(self.py_board.piece_map()) <= 20:
                                    print("[IA] Passage relais au Moteur C (Finale)")
                                    self.client.send_command("go")
                                    return 

                                ai_move_san = self.active_ai_function(fen)
                                if not ai_move_san: return
                                
                                move_obj = self.py_board.parse_san(ai_move_san)
                                uci_move = move_obj.uci()
                                print(f"[IA DL] Joue : {uci_move}")
                                
                                self.client.send_command(uci_move) # Envoi pour update C
                                self.move_log.append(uci_move)
                                self.last_move = self.parse_move_to_indices(uci_move)
                                self.py_board.push(move_obj) # Update Python
                            except Exception as e: print(f"[ERREUR IA] {e}")
                        threading.Thread(target=ai_thread_func).start()
                    else: print("[ERR] Fonction IA manquante")

                # --- CAS B : Moteur C (Minimax Pur) ---
                elif game_mode == 'pve_c':
                    # On envoie juste "go", on attend le retour "board:" pour la synchro
                    print("[IA C] Commande envoyée: go")
                    self.client.send_command("go")

            # 3. Dessin & Events (Inchangé)
            self.screen.fill((0,0,0)) 
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
                            undo_cnt = 2 if game_mode != 'pvp' else 1
                            if len(self.move_log) >= undo_cnt:
                                for _ in range(undo_cnt):
                                    self.client.send_command("undo")
                                    if self.move_log: self.move_log.pop()
                                    try: self.py_board.pop() 
                                    except: pass
                                    turn = 'b' if turn == 'w' else 'w'
                                self.last_move = self.parse_move_to_indices(self.move_log[-1]) if self.move_log else []
                        continue

                    if event.pos[0] > BOARD_WIDTH: continue
                    if (game_mode != 'pvp' and turn != player_color) or ai_thinking: continue

                    col, row = event.pos[0] // SQ_SIZE, event.pos[1] // SQ_SIZE
                    if player_color == 'b': col, row = 7 - col, 7 - row
                    
                    if self.selected_sq == (row, col): self.selected_sq = (); self.player_clicks = []
                    else: self.selected_sq = (row, col); self.player_clicks.append(self.selected_sq)
                    
                    if len(self.player_clicks) == 2:
                        start, end = self.player_clicks[0], self.player_clicks[1]
                        move_str = self.col_row_to_algebraic(start[1], start[0]) + self.col_row_to_algebraic(end[1], end[0])
                        piece_moving = self.visual_board[start[0]][start[1]]
                        if piece_moving[1] == 'P':
                            if (piece_moving[0] == 'w' and end[0] == 0) or (piece_moving[0] == 'b' and end[0] == 7): move_str += 'q'
                        try:
                            move_obj = chess.Move.from_uci(move_str)
                            if move_obj in self.py_board.legal_moves:
                                self.py_board.push(move_obj) # Update Python
                                self.client.send_command(move_str) # Update C
                                self.move_log.append(move_str)
                                self.last_move = [start, end]
                        except: self.client.send_command(move_str)
                        self.selected_sq = (); self.player_clicks = []
            self.clock.tick(FPS)
        self.client.close(); pygame.quit()

if __name__ == "__main__":
    g = Game()
    g.run()