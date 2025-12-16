import torch
import torch.nn as nn
import copy
import random
import chess
import numpy as np
import sys
import os
import time

# --- GESTION DES IMPORTS (Dossier Parent) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dataset import board_to_tensor

# --- CONFIGURATION ---
POPULATION_SIZE = 50     
SURVIVORS = 10           
MUTATION_RATE = 0.05     
GENERATIONS = 50         
DEPTH_MINIMAX = 2        
GAMES_PER_BOT = 4        
SAVE_INTERVAL = 10       # <--- LA LIGNE MANQUANTE AJOUTÉE ICI

# --- 1. LE PETIT CERVEAU (TinyNet) ---
class TinyChessNet(nn.Module):
    def __init__(self):
        super(TinyChessNet, self).__init__()
        # Entrée: 17x8x8 (1088) -> Cachée: 64 -> Sortie: 1 (Score position)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() 
        )

    def forward(self, x):
        return self.net(x) * 10 

# --- 2. LE MOTEUR MINIMAX ---
def evaluate_position_with_net(model, board):
    with torch.no_grad():
        # On récupère les données du plateau
        data = board_to_tensor(board)
        
        # Si c'est un tableau NumPy (cas classique), on convertit
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float().unsqueeze(0)
        # Si c'est déjà un Tensor (votre cas actuel), on s'assure juste du type/format
        else:
            tensor = data.float().unsqueeze(0)
            
        return model(tensor).item()

def alpha_beta(board, depth, alpha, beta, maximizing_player, model):
    if depth == 0 or board.is_game_over():
        return evaluate_position_with_net(model, board)

    # OPTIMISATION : On trie les coups pour regarder les captures en premier
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    if maximizing_player:
        max_eval = -float('inf')
        for move in moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, False, model)
            board.pop()
            
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval) 
            
            if beta <= alpha:
                break # COUPURE BETA
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval = alpha_beta(board, depth - 1, alpha, beta, True, model)
            board.pop()
            
            min_eval = min(min_eval, eval)
            beta = min(beta, eval) 
            
            if beta <= alpha:
                break # COUPURE ALPHA
        return min_eval

def get_best_move_genetic(model, board, depth=DEPTH_MINIMAX):
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    
    is_white = (board.turn == chess.WHITE)
    best_val = -float('inf') if is_white else float('inf')
    
    moves = list(board.legal_moves)
    random.shuffle(moves) 
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    for move in moves:
        board.push(move)
        val = alpha_beta(board, depth - 1, alpha, beta, not is_white, model)
        board.pop()

        if is_white:
            if val > best_val:
                best_val = val
                best_move = move
            alpha = max(alpha, val)
        else:
            if val < best_val:
                best_val = val
                best_move = move
            beta = min(beta, val)
                
    return best_move if best_move else moves[0]

# --- 3. L'AFFRONTEMENT (1vs1) ---
def play_match(model_a, model_b):
    board = chess.Board()
    for _ in range(120): # Limite 120 coups
        if board.is_game_over(): break
        
        if board.turn == chess.WHITE:
            move = get_best_move_genetic(model_a, board)
        else:
            move = get_best_move_genetic(model_b, board)
            
        board.push(move)

    res = board.result()
    if res == "1-0": return 1
    elif res == "0-1": return -1
    return 0 

# --- 4. EVOLUTION ---
def mutate(model):
    child = copy.deepcopy(model)
    with torch.no_grad():
        for param in child.parameters():
            if random.random() < 0.3: 
                noise = torch.randn_like(param) * MUTATION_RATE
                param.add_(noise)
    return child

def run_evolution():
    print(f"--- BATTLE ROYALE : {POPULATION_SIZE} BOTS ---")
    population = [TinyChessNet() for _ in range(POPULATION_SIZE)]
    
    save_dir = os.path.join(parent_dir, "..", "..", "models", "evolution")
    os.makedirs(save_dir, exist_ok=True)

    total_start_time = time.time()

    for gen in range(GENERATIONS):
        gen_start_time = time.time()
        scores = [0] * POPULATION_SIZE
        
        print(f"\n--- GÉNÉRATION {gen+1}/{GENERATIONS} : LA BASTON ---")
        
        match_count = 0
        total_matches = POPULATION_SIZE * GAMES_PER_BOT // 2
        indices = list(range(POPULATION_SIZE))
        
        for _ in range(GAMES_PER_BOT):
            random.shuffle(indices)
            for i in range(0, POPULATION_SIZE, 2):
                idx_a = indices[i]
                idx_b = indices[i+1]
                
                res = play_match(population[idx_a], population[idx_b])
                
                if res == 1:
                    scores[idx_a] += 3; scores[idx_b] += 0
                elif res == -1:
                    scores[idx_a] += 0; scores[idx_b] += 3 
                else:
                    scores[idx_a] += 1; scores[idx_b] += 1
                
                match_count += 1
                print(f"Matchs joués: {match_count}/{total_matches}", end='\r')

        gen_end_time = time.time()
        duration = gen_end_time - gen_start_time
        print(f"\n  >>> Durée Gen {gen+1}: {(duration//60):.2f}min {(duration%60):.2f}s")

        ranked_population = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
        best_score = ranked_population[0][0]
        print(f"  >>> Champion Gen {gen+1} : {best_score} points")
        print("      Scores des 5 Meilleurs : ", end="")
        print([item[0] for item in ranked_population[:5]])
        print(f"      Scores des Survivants : {[item[0] for item in ranked_population[:SURVIVORS]]}")
        print("      Score médian : {ranked_population[POPULATION_SIZE//2][0]}")
        print("      Score le plus bas : {ranked_population[-1][0]}")
        
        survivors = [item[1] for item in ranked_population[:SURVIVORS]]
        
        if (gen + 1) % SAVE_INTERVAL == 0 or (gen + 1) == GENERATIONS:
            save_path = os.path.join(save_dir, f"genetic_best_gen_{gen+1}.pth")
            torch.save(survivors[0].state_dict(), save_path)
            print(f"  >>> 💾 Modèle sauvegardé : genetic_best_gen_{gen+1}.pth")
        
        new_pop = survivors[:] 
        while len(new_pop) < POPULATION_SIZE:
            parent = random.choice(survivors)
            child = mutate(parent)
            new_pop.append(child)
            
        population = new_pop

    total_duration = time.time() - total_start_time
    print(f"\n--- ÉVOLUTION TERMINÉE en {total_duration/60:.1f} minutes ---")

if __name__ == "__main__":
    run_evolution()