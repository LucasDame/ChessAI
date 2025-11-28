import torch
import chess
from model import ChessNet
from dataset import board_to_tensor, index_to_move

MODEL_PATH = "../models/chess_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialisation du Modèle (fait une seule fois pour la vitesse) ---
def load_model():
    # S'assurer que les poids correspondent à l'architecture
    model = ChessNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Très important pour désactiver Dropout/BatchNorm en inférence
        print(f"Modèle DL chargé sur {DEVICE}.")
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None

CHESS_MODEL = load_model()

def get_dl_move(fen_string: str) -> str:
    """Prédit le meilleur coup pour une position donnée."""
    if CHESS_MODEL is None:
        return "" # Échoue si le modèle n'est pas chargé

    board = chess.Board(fen_string)
    
    # 1. Préparation de l'entrée (Input Tensor)
    input_tensor = board_to_tensor(board).unsqueeze(0).to(DEVICE)
    # unsqueeze(0) ajoute la dimension "Batch Size" = 1

    # 2. Prédiction (Passe avant)
    with torch.no_grad(): # Désactive le calcul de gradient (gain de vitesse)
        output = CHESS_MODEL(input_tensor)
        
    # 3. Trouver le meilleur coup
    # torch.argmax retourne l'index (0-4095) avec le score le plus haut
    best_index = torch.argmax(output).item()
    
    # 4. Conversion Index -> Coup Algébrique
    best_move_obj = index_to_move(best_index)
    
    # 5. Vérification Légale (IMPORTANT)
    # Le NN peut suggérer un coup illégal (ex: le roi en échec). 
    # On doit le vérifier dans le moteur C ou ici. Faisons-le ici pour l'instant.
    
    # Si le coup suggéré est légal, on le retourne
    if best_move_obj in board.legal_moves:
        return board.san(best_move_obj) # Retourne au format standard (e4, Nf3)
    
    # Sinon, on cherche un coup légal
    # On itère sur les 10 meilleurs coups pour trouver le premier coup légal
    top_10_indices = torch.topk(output, 10).indices.squeeze().tolist()

    for idx in top_10_indices:
        candidate_move = index_to_move(idx)
        if candidate_move in board.legal_moves:
            print(f"WARN: Coup {best_move_obj} illégal. Choix du coup N°{idx}: {board.san(candidate_move)}")
            return board.san(candidate_move)
            
    # Si vraiment aucun des top 10 n'est légal, on prend le premier coup légal trouvé
    return board.san(list(board.legal_moves)[0]) 

# --- Exemple de Test ---
if __name__ == '__main__':
    # Position de départ
    move = get_dl_move(chess.STARTING_FEN)
    print(f"Le modèle prédit : {move}")