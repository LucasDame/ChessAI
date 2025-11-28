import torch
import chess
import os  # <--- INDISPENSABLE pour les chemins
from model import ChessNet
from dataset import board_to_tensor, index_to_move

# --- CORRECTION DU CHEMIN (Le point critique) ---
# On récupère le dossier où se trouve CE fichier (DeepLearning/src)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# On construit le chemin absolu vers le modèle (DeepLearning/models/chess_model.pth)
# On remonte d'un cran ("..") puis on va dans "models"
MODEL_PATH = os.path.join(CURRENT_DIR, "..", "models", "chess_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Initialisation du Modèle ---
def load_model():
    # Debug : Afficher où on cherche le fichier
    print(f"[DEBUG DL] Recherche du modèle ici : {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERREUR CRITIQUE] Le fichier modèle est introuvable !")
        print(f" -> Avez-vous lancé l'entraînement (train.py) ?")
        return None

    model = ChessNet().to(DEVICE)
    try:
        # map_location permet de charger sur CPU même si entraîné sur GPU
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Très important pour désactiver Dropout/BatchNorm en inférence
        print(f"[SUCCÈS] Modèle DL chargé sur {DEVICE}.")
        return model
    except Exception as e:
        print(f"[ERREUR] Échec lors du chargement des poids : {e}")
        return None

CHESS_MODEL = load_model()

def get_dl_move(fen_string: str) -> str:
    """Prédit le meilleur coup pour une position donnée."""
    if CHESS_MODEL is None:
        return "" # Retourne vide si pas de modèle, l'UI gérera l'erreur

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
    
    # Si le coup suggéré est légal, on le retourne (format SAN ex: "Nf3")
    if best_move_obj in board.legal_moves:
        return board.san(best_move_obj)
    
    # Sinon, on cherche un coup légal dans les 10 meilleurs choix
    top_10_indices = torch.topk(output, 10).indices.squeeze().tolist()

    for idx in top_10_indices:
        candidate_move = index_to_move(idx)
        if candidate_move in board.legal_moves:
            print(f"[IA DEBUG] Coup préféré {best_move_obj} illégal. Choix alternatif : {board.san(candidate_move)}")
            return board.san(candidate_move)
            
    # Si vraiment aucun des top 10 n'est légal (très rare), on prend le premier coup légal possible
    fallback_move = list(board.legal_moves)[0]
    print(f"[IA DEBUG] Aucun coup valide trouvé dans le Top 10. Fallback sur : {board.san(fallback_move)}")
    return board.san(fallback_move)

# --- Exemple de Test ---
if __name__ == '__main__':
    # Ce bloc ne s'exécute que si on lance ce fichier directement
    move = get_dl_move(chess.STARTING_FEN)
    print(f"Le modèle prédit : {move}")