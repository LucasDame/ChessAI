import torch
import copy
import random
import os
import time
import chess.engine
import multiprocessing
import platform
import chess
from concurrent.futures import ProcessPoolExecutor

# Import du coeur (assure-toi que genetic_core.py est dans le même dossier ou dans PYTHONPATH)
from genetic_core import TinyChessNet, get_best_move

# Se placer dans le dossier du script (pour que les chemins relatifs fonctionnent)
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)

# Chemins relatifs vers Stockfish (ajuste si besoin)
STOCKFISH_PATH_LINUX = os.path.join("..", "stockfish", "stockfish-ubuntu")
STOCKFISH_PATH_WINDOWS = os.path.join("..", "stockfish", "stockfish-windows.exe")

POPULATION_SIZE = 50
SURVIVORS = 10
MUTATION_RATE = 0.05
DEPTH = 3
STOCKFISH_TIME_LIMIT = 0.05
CURRICULUM_SCORE_THRESHOLD = 450
GAMES_PER_COLOR = 5
TOTAL_GAMES_PER_BOT = GAMES_PER_COLOR * 2

if platform.system() == "Windows":
    STOCKFISH_PATH = STOCKFISH_PATH_WINDOWS
elif platform.system() in ("Linux", "Darwin"):
    STOCKFISH_PATH = STOCKFISH_PATH_LINUX
else:
    STOCKFISH_PATH = STOCKFISH_PATH_LINUX

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}

# --- initialisation synchrone de stockfish (plus fiable dans des workers) ---
def init_stockfish_engine_sync(stockfish_path, elo):
    """Initialise Stockfish de façon synchrone. Retourne l'engine ou lève une exception."""
    if not os.path.isfile(stockfish_path):
        raise FileNotFoundError(f"Fichier introuvable: {stockfish_path}")
    if not os.access(stockfish_path, os.X_OK) and platform.system() != "Windows":
        # Sur Windows l'executable peut ne pas être marqué exécutable de la même façon
        raise PermissionError(f"Stockfish présent mais pas exécutable: {stockfish_path}")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
    except Exception:
        # Certaines builds ne supportent pas toutes les options -> on ignore l'exception
        pass
    return engine


# --- WORKER : Cette fonction tourne sur chaque CPU ---
def play_match_worker(args):
    """
    Joue une partie : IA (weights) vs Stockfish (elo)
    args = (weights_dict_cpu, stockfish_elo, stockfish_path, model_color)
    """
    weights, stockfish_elo, stockfish_path, model_color = args

    # 1. Recharger le modèle
    try:
        torch.set_num_threads(1)
        model = TinyChessNet()
        model.load_state_dict(weights)
        model.eval()
    except Exception as e:
        # problème de chargement du modèle -> score 0
        print(f"[worker] Erreur load model: {e}")
        return 0

    engine = None

    # 2. Configurer Stockfish (synchronement)
    try:
        engine = init_stockfish_engine_sync(stockfish_path, stockfish_elo)
    except FileNotFoundError as fnf:
        print(f"[worker] Stockfish non trouvé: {fnf}")
        return 0
    except PermissionError as perm:
        print(f"[worker] Permission error stockfish: {perm}")
        return 0
    except Exception as e:
        print(f"[worker] Erreur init stockfish: {e}")
        try:
            if engine:
                engine.quit()
        except Exception:
            pass
        return 0

    board = chess.Board()
    fitness = 0
    limit_moves = 100

    try:
        for _ in range(limit_moves):
            if board.is_game_over():
                break

            if board.turn == model_color:
                # --- Tour de notre IA ---
                move = None
                try:
                    move = get_best_move(model, board, depth=DEPTH)
                except Exception:
                    # l'IA a planté; on tombera sur un coup aléatoire
                    move = None

                if move is None:
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = random.choice(legal_moves)
                    else:
                        break

                try:
                    board.push(move)
                except Exception:
                    break

            else:
                # Tour de Stockfish
                try:
                    limit = chess.engine.Limit(time=STOCKFISH_TIME_LIMIT)
                    res = engine.play(board, limit=limit)
                    if res.move is None:
                        break
                    board.push(res.move)
                except Exception:
                    break

            fitness += 1  # Récompense pour la survie

    finally:
        # 3. Quitter l'engine proprement dans tous les cas
        try:
            if engine:
                engine.quit()
        except Exception:
            pass

    # --- CALCUL DU BONUS DE FITNESS ---
    result = board.result()
    ai_win_score = 500
    draw_score = 200

    if result == "1-0":
        if model_color == chess.WHITE:
            fitness += ai_win_score
    elif result == "0-1":
        if model_color == chess.BLACK:
            fitness += ai_win_score
    elif result == "1/2-1/2":
        fitness += draw_score

    # 2. Bonus Matériel
    ai_material = sum(len(board.pieces(pt, model_color)) * PIECE_VALUES[pt] for pt in chess.PIECE_TYPES)
    opp_material = sum(len(board.pieces(pt, not model_color)) * PIECE_VALUES[pt] for pt in chess.PIECE_TYPES)

    material_bonus = (ai_material - opp_material) * 5

    return fitness + material_bonus


def mutate(model):
    """ Crée une copie mutée du modèle parent. """
    child = copy.deepcopy(model)
    with torch.no_grad():
        for param in child.parameters():
            if random.random() < 0.2:
                noise = torch.randn_like(param) * MUTATION_RATE
                param.add_(noise)
    return child


def run_curriculum_evolution():
    # pour éviter d'utiliser tous les coeurs, on en laisse 2 libres
    num_workers = max(1, (os.cpu_count() or 2) - 2)
    print(f"--- DÉMARRAGE DE L'ÉVOLUTION PAR CURRICULUM ---")
    print(f"Utilisation de {num_workers} cœurs. {TOTAL_GAMES_PER_BOT} parties par bot.")

    # Vérification basique du binaire Stockfish (test synchrone)
    try:
        if not os.path.isfile(STOCKFISH_PATH):
            raise FileNotFoundError(f"{STOCKFISH_PATH} introuvable")
        # On tente d'ouvrir et de fermer un engine pour vérifier que tout est OK
        e = init_stockfish_engine_sync(STOCKFISH_PATH, 500)
        e.quit()
        print("Stockfish initialisé avec succès pour le test.")
    except Exception as exc:
        print("[ERREUR CRITIQUE] Impossible d'initialiser Stockfish. Vérifiez le chemin et les permissions.")
        print(f" Détail: {exc}")
        return 0

    population = [TinyChessNet() for _ in range(POPULATION_SIZE)]
    current_elo = 500
    gen = 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    save_dir = os.path.join(project_root, "models", "genetic_curriculum")
    os.makedirs(save_dir, exist_ok=True)

    total_start_time = time.time()

    while True:
        gen += 1
        start_time = time.time()

        print(f"\n=== GÉNÉRATION {gen} | Objectif : Stockfish {current_elo} ELO ({POPULATION_SIZE} BOTS) ===")

        tasks = []
        bot_indices = []

        # --- CRÉATION DES TÂCHES (GAMES_PER_COLOR pour chaque couleur) ---
        for i, bot in enumerate(population):
            # for safety ensure weights are on CPU (picklable)
            weights = {k: v.cpu() for k, v in bot.state_dict().items()}

            for _ in range(GAMES_PER_COLOR):
                tasks.append((weights, current_elo, STOCKFISH_PATH, chess.WHITE))
                bot_indices.append(i)

                tasks.append((weights, current_elo, STOCKFISH_PATH, chess.BLACK))
                bot_indices.append(i)

        # EXÉCUTION PARALLÈLE
        results = []
        try:
            print(f"Lancement de {len(tasks)} parties sur {num_workers} cœurs...")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(play_match_worker, tasks))
        except KeyboardInterrupt:
            print("\nArrêt manuel de l'exécution. Sauvegarde du champion actuel.")
            break
        except Exception as e:
            print(f"\nErreur critique lors du pool d'exécution : {e}")
            break

        # --- AGRÉGATION DES RÉSULTATS ---
        bot_scores_sum = {}
        for result_score, bot_idx in zip(results, bot_indices):
            if bot_idx not in bot_scores_sum:
                bot_scores_sum[bot_idx] = 0
            bot_scores_sum[bot_idx] += result_score

        final_scores_tuples = []
        for i in range(POPULATION_SIZE):
            total_score = bot_scores_sum.get(i, 0)
            final_scores_tuples.append((total_score, population[i]))

        final_scores_tuples.sort(key=lambda x: x[0], reverse=True)
        best_score = final_scores_tuples[0][0] if final_scores_tuples else 0
        avg_score_top10 = sum([s[0] for s in final_scores_tuples[:10]]) / (10 * TOTAL_GAMES_PER_BOT) if POPULATION_SIZE >= 10 else 0

        duration = time.time() - start_time
        print(f"  > Temps : {duration:.2f}s | Tps/Partie : {duration/max(1, len(tasks)):.3f}s")
        print(f"  > Meilleur Score Total : {best_score} pts | Moyenne Top 10/Partie : {avg_score_top10:.1f} pts")

        # Sauvegarde et Curriculum
        survivors = [s[1] for s in final_scores_tuples[:SURVIVORS]]

        if gen % 5 == 0 and survivors:
            torch.save(survivors[0].state_dict(), os.path.join(save_dir, f"gen_{gen}_elo_{current_elo}.pth"))
            print(f"  💾 Champion Génération {gen} sauvegardé.")

        if avg_score_top10 > CURRICULUM_SCORE_THRESHOLD and survivors:
            print(f"  >>> 🚀 NIVEAU RÉUSSI ! Passage à Stockfish {current_elo + 100} ELO <<<")
            current_elo += 100
            torch.save(survivors[0].state_dict(), os.path.join(save_dir, f"champion_elo_{current_elo}.pth"))

        # Reproduction
        if not survivors:
            print("[WARN] Aucun survivant — réinitialisation de la population.")
            population = [TinyChessNet() for _ in range(POPULATION_SIZE)]
        else:
            new_pop = survivors[:]
            while len(new_pop) < POPULATION_SIZE:
                parent = random.choice(survivors)
                child = mutate(parent)
                new_pop.append(child)
            population = new_pop

        if current_elo > 2000:
            print("\nObjectif ELO atteint. Fin de l'entraînement.")
            break

    total_duration = time.time() - total_start_time
    print(f"\n--- ÉVOLUTION TERMINÉE (durée {total_duration:.1f}s) ---")
    return 0


if __name__ == "__main__":
    # Nécessaire sous Windows pour multiprocessing
    multiprocessing.freeze_support()
    # Optionnel : forcer 'spawn' start method (surtout utile sous Linux/Mac si problème)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    run_curriculum_evolution()
