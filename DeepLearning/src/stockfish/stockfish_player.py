import chess
import chess.engine
import os

# --- CONFIGURATION (ADAPTEZ CE CHEMIN) ---
# CHEMIN VERS VOTRE EXECUTABLE STOCKFISH
# Exemples :
# - Windows : 'C:/stockfish/stockfish-windows-x86-64-avx2.exe'
# - WSL/Linux : '/usr/games/stockfish' (ou le chemin exact du binaire)
STOCKFISH_PATH_LINUX = "stockfish-ubuntu" 
STOCKFISH_PATH_WINDOWS = "stockfish-windows.exe"

# --- PARAMÈTRES POUR DÉFINIR L'ELO DE L'ANCRE ---
# Pour définir un ELO précis, vous pouvez utiliser les paramètres UCI de Stockfish.
# ELO_RATING : Simule un joueur à un niveau donné (ex: 1800 ELO).
# En l'absence de ce paramètre, vous pouvez utiliser 'depth' (profondeur de recherche).
ANCHOR_ELO = 1800
ENGINE_TIME_LIMIT = 0.5  # Temps de réflexion max par coup (en secondes)
ENGINE_DEPTH_LIMIT = 5   # Profondeur de recherche alternative

class StockfishPlayer:
    def __init__(self,path = STOCKFISH_PATH_WINDOWS if os.name == 'nt' else STOCKFISH_PATH_LINUX, elo=ANCHOR_ELO, time_limit=ENGINE_TIME_LIMIT):
        try:
            self.engine = chess.engine.popen_uci([path])
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo}) # Limite l'ELO
            self.limit = chess.engine.Limit(time=time_limit) # Recherche par temps
            print(f"[UCI] Stockfish initialisé à {elo} ELO.")
        except FileNotFoundError:
            print(f"[ERREUR CRITIQUE] Stockfish introuvable à : {path}")
            self.engine = None
        except Exception as e:
            print(f"[ERREUR UCI] {e}")
            self.engine = None

    def get_move(self, fen_string: str) -> str:
        if self.engine is None: return None
        board = chess.Board(fen_string)
        
        try:
            # Calcule le meilleur coup avec les limites de temps/profondeur définies
            result = self.engine.play(board, self.limit)
            return result.move.uci()
        except Exception as e:
            print(f"[ERREUR Stockfish] {e}")
            return None

    def close(self):
        if self.engine:
            self.engine.quit()

# Initialisation de l'ancre pour les tests
STOCKFISH_ANCHOR = StockfishPlayer()