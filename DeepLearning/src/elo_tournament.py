import chess
import random
import time
import os

from stockfish.stockfish_player import StockfishPlayer

from dl_ai_player import get_dl_move
from dl_ai_player_resnet import get_resnet_move
from dl_ai_player_seresnet import get_resnet_move as get_seresnet_move
from dl_ai_player_alphazero import get_alphazero_move

# --- CONFIGURATION DU TOURNOI ---
NUM_RUNS = 5          # Nombre de répétitions complètes du tournoi
GAMES_PER_PAIR = 25   # Nombre de parties par paire par run
MAX_MOVES = 300 

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILENAME = os.path.join(CURRENT_DIR, "tournament_repeatability_results.txt")

STOCKFISH_ENGINES_CONFIG = [
    (1000, 0.1), (1400, 0.2), (1600, 0.3), (1800, 0.4), (2000, 0.5), (2200, 1.0)
]

# Initialisation unique des moteurs (on ne les ferme qu'à la toute fin)
print("Initialisation des moteurs Stockfish...")
stockfish_players = {}
for elo_val, time_limit in STOCKFISH_ENGINES_CONFIG:
    name = f"Stockfish {elo_val} ELO"
    stockfish_players[name] = StockfishPlayer(elo=elo_val, time_limit=time_limit)

PLAYERS = {
    "CNN": get_dl_move,
    "ResNet": get_resnet_move,
    "SE-ResNet": get_seresnet_move,
    "AlphaZero" : get_alphazero_move,
}
PLAYERS.update({name: engine.get_move for name, engine in stockfish_players.items()})

ELO_START_DL = 1500 

# =============================================================================
# FONCTIONS LOGIQUE & ELO (Inchangées)
# =============================================================================

def log_and_print(message, log_file):
    print(message)
    if log_file:
        log_file.write(message + '\n')
        log_file.flush()

def get_dynamic_k_factor(num_games):
    if num_games < 30: return 40
    elif num_games < 100: return 24
    else: return 10

def calculate_expected_score(elo_a, elo_b):
    return 1.0 / (1.0 + 10**((elo_b - elo_a) / 400))

def update_elo(elo_a, elo_b, score_a, games_played_a, is_fixed=False):
    if is_fixed: return elo_a
    k = get_dynamic_k_factor(games_played_a)
    expected = calculate_expected_score(elo_a, elo_b)
    return elo_a + k * (score_a - expected)

def play_game(white_name, black_name, log_file):
    board = chess.Board()
    player_white_func = PLAYERS[white_name]
    player_black_func = PLAYERS[black_name]

    for _ in range(MAX_MOVES):
        if board.is_game_over(): break
        current_fen = board.fen()
        player_func = player_white_func if board.turn == chess.WHITE else player_black_func
        
        try:
            uci_move = player_func(current_fen)
            if not uci_move: return 0.5 
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves: return 1.0 if board.turn == chess.BLACK else 0.0
            board.push(move)
        except: return 0.5

    result = board.result()
    if result == "1-0": return 1.0
    elif result == "0-1": return 0.0
    return 0.5 

# =============================================================================
# EXÉCUTION RÉPÉTÉE
# =============================================================================

def run_repeatability_test():
    print(f"Enregistrement des résultats dans : {LOG_FILENAME}")
    
    with open(LOG_FILENAME, "w", encoding='utf-8') as log_file:
        log_and_print(f"=== TEST DE RÉPÉTABILITÉ : {NUM_RUNS} RUNS ===", log_file)
        
        # --- BOUCLE PRINCIPALE DES RUNS ---
        for run_id in range(1, NUM_RUNS + 1):
            log_and_print(f"\n\n{'#'*40}", log_file)
            log_and_print(f"### DÉBUT DU RUN {run_id}/{NUM_RUNS} ###", log_file)
            log_and_print(f"{'#'*40}\n", log_file)
            
            # --- RESET COMPLET DES ELO ET COMPTEURS ---
            elos = {}
            games_played_count = {name: 0 for name in PLAYERS}
            for name in PLAYERS:
                elos[name] = float(name.split()[1]) if name.startswith("Stockfish") else float(ELO_START_DL)
            
            # --- GÉNÉRATION DES MATCHS ---
            all_games_list = []
            player_names = list(PLAYERS.keys())
            processed_pairs = set()

            for p1 in player_names:
                for p2 in player_names:
                    if p1 == p2: continue
                    pair_id = tuple(sorted((p1, p2)))
                    if pair_id in processed_pairs: continue
                    processed_pairs.add(pair_id)
                    games_per_side = max(1, GAMES_PER_PAIR // 2)
                    for _ in range(games_per_side):
                        all_games_list.append((p1, p2))
                        all_games_list.append((p2, p1))

            random.shuffle(all_games_list)
            total_games = len(all_games_list)
            
            # --- BOUCLE DE JEU ---
            for idx, (white_name, black_name) in enumerate(all_games_list):
                score_white = play_game(white_name, black_name, None) # On ne logue pas les erreurs individuelles ici
                
                games_played_count[white_name] += 1
                games_played_count[black_name] += 1
                
                is_w_fixed = white_name.startswith("Stockfish")
                is_b_fixed = black_name.startswith("Stockfish")
                
                elos[white_name] = update_elo(elos[white_name], elos[black_name], score_white, games_played_count[white_name], is_w_fixed)
                elos[black_name] = update_elo(elos[black_name], elos[white_name], 1.0 - score_white, games_played_count[black_name], is_b_fixed)

                # Progression console
                print(f"[Run {run_id}] Partie {idx+1}/{total_games} | {white_name} vs {black_name}", end='\r')

            # --- FIN DU RUN : AFFICHAGE CLASSEMENT ---
            print() # Saut de ligne
            log_and_print(f"--- RÉSULTATS FINAUX DU RUN {run_id} ---", log_file)
            
            sorted_elos = sorted(elos.items(), key=lambda x: x[1], reverse=True)
            for rank, (name, elo) in enumerate(sorted_elos):
                prefix = "• ANCRE •" if name.startswith("Stockfish") else f"{rank+1}."
                log_and_print(f"{prefix} {name}: {int(elo)} ELO", log_file)
            
            log_file.flush()

        # --- NETTOYAGE FINAL ---
        for engine in stockfish_players.values(): engine.close()
        log_and_print("\n=== TOUS LES TESTS TERMINÉS ===", log_file)

if __name__ == "__main__":
    run_repeatability_test()