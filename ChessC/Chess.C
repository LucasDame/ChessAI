#include <stdio.h>
#include <stdint.h>
#include <string.h>

// ============================================================================
//                               DÉFINITIONS ET TYPES
// ============================================================================

typedef uint64_t Bitboard;
typedef uint32_t MOVE; // Un coup encodé sur 32 bits

// Énumération des cases (0 à 63)
enum {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8
};

// Énumération des pièces
enum { wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK };

// Énumération des couleurs
enum { WHITE, BLACK, BOTH };

// Drapeaux de coups (Move Flags)
enum { 
    QUIET_MOVE = 0, 
    DOUBLE_PAWN_PUSH = 1, 
    KING_CASTLE = 2, 
    QUEEN_CASTLE = 3, 
    CAPTURE_FLAG = 4, 
    EP_CAPTURE = 5, 
    PROMOTION_KNIGHT = 8,
    PROMOTION_BISHOP = 9,
    PROMOTION_ROOK = 10,
    PROMOTION_QUEEN = 11,
    PROMOTION_CAPTURE = 12 
};

// Mode de la fonction make_move
enum { ALL_MOVES, ONLY_CAPTURES };

// ============================================================================
//                             MACROS ET CONSTANTES
// ============================================================================

// Masques de fichiers (pour éviter le débordement horizontal)
const Bitboard MASK_A_FILE = 0x0101010101010101ULL;
const Bitboard MASK_H_FILE = 0x8080808080808080ULL;
const Bitboard MASK_AB_FILE = 0x0303030303030303ULL;
const Bitboard MASK_GH_FILE = 0xC0C0C0C0C0C0C0C0ULL;

// Manipulation des bits
#define SET_BIT(bb, sq) ((bb) |= (1ULL << (sq)))
#define GET_BIT(bb, sq) ((bb) & (1ULL << (sq)))
#define POP_BIT(bb, sq) ((bb) &= ~(1ULL << (sq)))

// Encodage / Décodage des coups
// Bits: [From: 0-5] [To: 6-11] [Piece: 12-15] [Promoted: 16-19] [Flag: 20-23]
#define ENCODE_MOVE(from, to, piece, promoted, flag) \
    ( (from) | ((to) << 6) | ((piece) << 12) | ((promoted) << 16) | ((flag) << 20) )

#define GET_MOVE_FROM(move)      ((move) & 0x3f)
#define GET_MOVE_TO(move)        (((move) >> 6) & 0x3f)
#define GET_MOVE_PIECE(move)     (((move) >> 12) & 0xf)
#define GET_MOVE_PROMOTED(move)  (((move) >> 16) & 0xf)
#define GET_MOVE_FLAG(move)      (((move) >> 20) & 0xf)

// ============================================================================
//                            VARIABLES GLOBALES
// ============================================================================

Bitboard bitboards[12];          // Le plateau (12 bitboards)
Bitboard knight_attacks[64];     // Table d'attaques pré-calculée Cavaliers
Bitboard king_attacks[64];       // Table d'attaques pré-calculée Rois
int castling_rights_mask[64];    // Table pour mettre à jour les droits de roque

// État du jeu
int side = WHITE;                // Qui doit jouer ?
int en_passant_sq = 0;           // Case cible pour la prise en passant (0 = aucune)
int castle_rights = 15;          // 4 bits: WK WQ BK BQ (1111 = tout permis)

// Structure pour sauvegarder l'état (pour annuler un coup)
typedef struct {
    Bitboard bitboards[12];
    int side;
    int en_passant_sq;
    int castle_rights;
} GameState;

// ============================================================================
//                        INITIALISATION & PRÉ-CALCULS
// ============================================================================

// Initialise les masques pour la mise à jour des droits de roque
void init_castling_masks() {
    for (int i = 0; i < 64; i++) castling_rights_mask[i] = 15;
    
    // Si on touche ces cases, on perd les droits correspondants
    castling_rights_mask[A1] = 13; // Blanc perd Grand Roque (WQ)
    castling_rights_mask[H1] = 14; // Blanc perd Petit Roque (WK)
    castling_rights_mask[E1] = 12; // Blanc perd tout
    castling_rights_mask[A8] = 7;  // Noir perd Grand Roque (BQ)
    castling_rights_mask[H8] = 11; // Noir perd Petit Roque (BK)
    castling_rights_mask[E8] = 3;  // Noir perd tout
}

// Pré-calcul des attaques de sauteurs (Cavaliers et Rois)
void init_leapers_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        // --- CAVALIER ---
        Bitboard knight_bit = (1ULL << sq);
        Bitboard k_moves = 0;
        k_moves |= (knight_bit & ~MASK_GH_FILE) << 17;
        k_moves |= (knight_bit & ~MASK_AB_FILE) >> 17;
        k_moves |= (knight_bit & ~MASK_AB_FILE) << 15;
        k_moves |= (knight_bit & ~MASK_GH_FILE) >> 15;
        k_moves |= (knight_bit & ~MASK_H_FILE) << 10;
        k_moves |= (knight_bit & ~MASK_A_FILE) >> 10;
        k_moves |= (knight_bit & ~MASK_A_FILE) << 6;
        k_moves |= (knight_bit & ~MASK_H_FILE) >> 6;
        knight_attacks[sq] = k_moves;

        // --- ROI ---
        Bitboard king_bit = (1ULL << sq);
        Bitboard r_moves = 0;
        r_moves |= (king_bit & ~MASK_H_FILE) << 1;
        r_moves |= (king_bit & ~MASK_A_FILE) >> 1;
        r_moves |= (king_bit << 8);
        r_moves |= (king_bit >> 8);
        r_moves |= (king_bit & ~MASK_H_FILE) << 9;
        r_moves |= (king_bit & ~MASK_A_FILE) << 7;
        r_moves |= (king_bit & ~MASK_H_FILE) >> 7;
        r_moves |= (king_bit & ~MASK_A_FILE) >> 9;
        king_attacks[sq] = r_moves;
    }
}

// Initialise le plateau position de départ standard
void init_board() {
    memset(bitboards, 0, sizeof(bitboards));
    // Blancs
    SET_BIT(bitboards[wR], A1); SET_BIT(bitboards[wR], H1);
    SET_BIT(bitboards[wN], B1); SET_BIT(bitboards[wN], G1);
    SET_BIT(bitboards[wB], C1); SET_BIT(bitboards[wB], F1);
    SET_BIT(bitboards[wQ], D1); SET_BIT(bitboards[wK], E1);
    for (int i = A2; i <= H2; i++) SET_BIT(bitboards[wP], i);
    // Noirs
    SET_BIT(bitboards[bR], A8); SET_BIT(bitboards[bR], H8);
    SET_BIT(bitboards[bN], B8); SET_BIT(bitboards[bN], G8);
    SET_BIT(bitboards[bB], C8); SET_BIT(bitboards[bB], F8);
    SET_BIT(bitboards[bQ], D8); SET_BIT(bitboards[bK], E8);
    for (int i = A7; i <= H7; i++) SET_BIT(bitboards[bP], i);
    
    side = WHITE;
    castle_rights = 15;
    en_passant_sq = 0;
}

// ============================================================================
//                        LOGIQUE PIÈCES GLISSANTES
// ============================================================================

Bitboard get_bishop_attacks(int sq, Bitboard occupancy) {
    Bitboard attacks = 0;
    int r, f;
    int tr = sq / 8, tf = sq % 8;
    
    for (r=tr+1, f=tf+1; r<=7 && f<=7; r++, f++) { attacks |= (1ULL<<(r*8+f)); if((1ULL<<(r*8+f)) & occupancy) break; }
    for (r=tr+1, f=tf-1; r<=7 && f>=0; r++, f--) { attacks |= (1ULL<<(r*8+f)); if((1ULL<<(r*8+f)) & occupancy) break; }
    for (r=tr-1, f=tf+1; r>=0 && f<=7; r--, f++) { attacks |= (1ULL<<(r*8+f)); if((1ULL<<(r*8+f)) & occupancy) break; }
    for (r=tr-1, f=tf-1; r>=0 && f>=0; r--, f--) { attacks |= (1ULL<<(r*8+f)); if((1ULL<<(r*8+f)) & occupancy) break; }
    return attacks;
}

Bitboard get_rook_attacks(int sq, Bitboard occupancy) {
    Bitboard attacks = 0;
    int r, f;
    int tr = sq / 8, tf = sq % 8;
    
    for (r=tr+1; r<=7; r++) { attacks |= (1ULL<<(r*8+tf)); if((1ULL<<(r*8+tf)) & occupancy) break; }
    for (r=tr-1; r>=0; r--) { attacks |= (1ULL<<(r*8+tf)); if((1ULL<<(r*8+tf)) & occupancy) break; }
    for (f=tf+1; f<=7; f++) { attacks |= (1ULL<<(tr*8+f)); if((1ULL<<(tr*8+f)) & occupancy) break; }
    for (f=tf-1; f>=0; f--) { attacks |= (1ULL<<(tr*8+f)); if((1ULL<<(tr*8+f)) & occupancy) break; }
    return attacks;
}

// ============================================================================
//                        DÉTECTION D'ATTAQUE (ÉCHEC)
// ============================================================================

int is_square_attacked(int sq, int attacking_side) {
    Bitboard occupancy = 0;
    for (int i=0; i<12; i++) occupancy |= bitboards[i];

    // Indices des pièces attaquantes
    int pP = (attacking_side == WHITE) ? wP : bP;
    int pN = (attacking_side == WHITE) ? wN : bN;
    int pK = (attacking_side == WHITE) ? wK : bK;
    int pB = (attacking_side == WHITE) ? wB : bB;
    int pR = (attacking_side == WHITE) ? wR : bR;
    int pQ = (attacking_side == WHITE) ? wQ : bQ;

    // 1. Pions (Logique inverse : qui m'attaque ?)
    if (attacking_side == WHITE) {
        // Attaqué par pion blanc (venant du bas)
        if ( ((1ULL << sq) >> 9) & bitboards[pP] & ~MASK_H_FILE ) return 1;
        if ( ((1ULL << sq) >> 7) & bitboards[pP] & ~MASK_A_FILE ) return 1;
    } else {
        // Attaqué par pion noir (venant du haut)
        if ( ((1ULL << sq) << 9) & bitboards[pP] & ~MASK_A_FILE ) return 1;
        if ( ((1ULL << sq) << 7) & bitboards[pP] & ~MASK_H_FILE ) return 1;
    }

    // 2. Cavaliers & Roi
    if (knight_attacks[sq] & bitboards[pN]) return 1;
    if (king_attacks[sq] & bitboards[pK]) return 1;

    // 3. Glisseurs (Fou/Dame & Tour/Dame)
    if (get_bishop_attacks(sq, occupancy) & (bitboards[pB] | bitboards[pQ])) return 1;
    if (get_rook_attacks(sq, occupancy) & (bitboards[pR] | bitboards[pQ])) return 1;

    return 0;
}

// ============================================================================
//                        GESTION DES COUPS (MAKE MOVE)
// ============================================================================

// Sauvegarde de l'état
void save_board(GameState *state) {
    memcpy(state->bitboards, bitboards, 12 * sizeof(Bitboard));
    state->side = side;
    state->en_passant_sq = en_passant_sq;
    state->castle_rights = castle_rights;
}

// Restauration de l'état
void restore_board(GameState *state) {
    memcpy(bitboards, state->bitboards, 12 * sizeof(Bitboard));
    side = state->side;
    en_passant_sq = state->en_passant_sq;
    castle_rights = state->castle_rights;
}

// Fonction principale : Joue un coup et retourne 1 si légal, 0 sinon
int make_move(MOVE move, int capture_mode) {
    // 1. Sauvegarde
    GameState backup;
    save_board(&backup);

    // 2. Décodage
    int from = GET_MOVE_FROM(move);
    int to = GET_MOVE_TO(move);
    int piece = GET_MOVE_PIECE(move);
    int promoted = GET_MOVE_PROMOTED(move);
    int flag = GET_MOVE_FLAG(move);

    // Filtre pour "Captures Seulement"
    if (capture_mode == ONLY_CAPTURES) {
        if (flag != CAPTURE_FLAG && flag != EP_CAPTURE && flag != PROMOTION_CAPTURE) return 0;
    }

    // --- MISE À JOUR DU PLATEAU ---

    // Bouger la pièce
    POP_BIT(bitboards[piece], from);
    SET_BIT(bitboards[piece], to);

    // Gérer les captures (Retirer la pièce ennemie)
    if (flag == CAPTURE_FLAG || flag == PROMOTION_CAPTURE) {
        int start_p = (side == WHITE) ? bP : wP;
        int end_p   = (side == WHITE) ? bK : wK;
        for (int p = start_p; p <= end_p; p++) {
            if (GET_BIT(bitboards[p], to)) {
                POP_BIT(bitboards[p], to);
                break;
            }
        }
    }

    // Gérer la Promotion
    if (promoted) {
        POP_BIT(bitboards[piece], to);   // Enlever le pion
        SET_BIT(bitboards[promoted], to); // Mettre la dame/cavalier...
    }

    // Gérer la Prise en Passant
    if (flag == EP_CAPTURE) {
        int ep_pawn_sq = (side == WHITE) ? (to - 8) : (to + 8);
        int enemy_pawn = (side == WHITE) ? bP : wP;
        POP_BIT(bitboards[enemy_pawn], ep_pawn_sq);
    }

    // Gérer le Roque (Mouvement de la Tour)
    if (flag == KING_CASTLE) {
        if (side == WHITE) { POP_BIT(bitboards[wR], H1); SET_BIT(bitboards[wR], F1); }
        else               { POP_BIT(bitboards[bR], H8); SET_BIT(bitboards[bR], F8); }
    } else if (flag == QUEEN_CASTLE) {
        if (side == WHITE) { POP_BIT(bitboards[wR], A1); SET_BIT(bitboards[wR], D1); }
        else               { POP_BIT(bitboards[bR], A8); SET_BIT(bitboards[bR], D8); }
    }

    // Mise à jour de la case En Passant pour le prochain tour
    en_passant_sq = 0;
    if (flag == DOUBLE_PAWN_PUSH) {
        en_passant_sq = (from + to) / 2;
    }

    // Mise à jour des droits de Roque
    castle_rights &= castling_rights_mask[from];
    castle_rights &= castling_rights_mask[to];

    // Changer de côté
    side ^= 1;

    // --- VÉRIFICATION DE LA LÉGALITÉ ---
    
    // On regarde si le roi du joueur qui vient de jouer (maintenant side^1) est attaqué
    int king_side = side ^ 1;
    int king_bit = (king_side == WHITE) ? bitboards[wK] : bitboards[bK];
    int king_sq = __builtin_ctzll(king_bit);

    if (is_square_attacked(king_sq, side)) {
        // Illégal : Le roi est en échec
        restore_board(&backup);
        return 0;
    }

    return 1; // Légal
}

// ============================================================================
//                                    MAIN
// ============================================================================

// Convertit un index (0-63) en chaîne de caractères (ex: 8 -> "a2")
void print_square_coord(int sq) {
    if (sq < 0 || sq > 63) {
        printf(" - ");
        return;
    }
    printf("%c%d", 'a' + (sq % 8), 1 + (sq / 8));
}

// Affiche le plateau avec un look propre
void print_board() {
    printf("\n   +---+---+---+---+---+---+---+---+\n");

    for (int rank = 7; rank >= 0; rank--) {
        printf(" %d |", rank + 1); // Numéro du rang à gauche
        
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            char *piece_char = "   "; // Espace par défaut (case vide)

            // On cherche quelle pièce est sur la case
            // (Note: Sur Windows, si les symboles ne s'affichent pas, remplace par P, N, B...)
            if (GET_BIT(bitboards[wP], sq)) piece_char = " ♙ ";
            else if (GET_BIT(bitboards[wN], sq)) piece_char = " ♘ ";
            else if (GET_BIT(bitboards[wB], sq)) piece_char = " ♗ ";
            else if (GET_BIT(bitboards[wR], sq)) piece_char = " ♖ ";
            else if (GET_BIT(bitboards[wQ], sq)) piece_char = " ♕ ";
            else if (GET_BIT(bitboards[wK], sq)) piece_char = " ♔ ";
            else if (GET_BIT(bitboards[bP], sq)) piece_char = " ♟ ";
            else if (GET_BIT(bitboards[bN], sq)) piece_char = " ♞ ";
            else if (GET_BIT(bitboards[bB], sq)) piece_char = " ♝ ";
            else if (GET_BIT(bitboards[bR], sq)) piece_char = " ♜ ";
            else if (GET_BIT(bitboards[bQ], sq)) piece_char = " ♛ ";
            else if (GET_BIT(bitboards[bK], sq)) piece_char = " ♚ ";

            printf("%s|", piece_char);
        }
        printf("\n   +---+---+---+---+---+---+---+---+\n");
    }
    printf("     a   b   c   d   e   f   g   h\n\n");

    // --- AFFICHAGE DES STATISTIQUES ---
    printf("Trait au      : %s\n", (side == WHITE) ? "Blancs (White)" : "Noirs (Black)");


    printf("Droits Roque  : ");
    // Décodage des bits 1, 2, 4, 8
    if (castle_rights & 1) printf("K"); else printf("-"); // White King
    if (castle_rights & 2) printf("Q"); else printf("-"); // White Queen
    if (castle_rights & 4) printf("k"); else printf("-"); // Black King
    if (castle_rights & 8) printf("q"); else printf("-"); // Black Queen
    printf("\n");
}

// --- GÉNÉRATEUR DE COUPS (Nécessaire pour trouver le Mat et valider les entrées) ---

// Ajoute un coup à la liste
void add_move(MOVE *move_list, int *move_count, int move) {
    move_list[*move_count] = move;
    (*move_count)++;
}

// --- GÉNÉRATEUR DE COUPS COMPLET ---

void generate_moves(MOVE *move_list, int *move_count) {
    *move_count = 0;
    int src, tgt;
    Bitboard bitboard, attacks;
    
    // On définit qui sont les amis et les ennemis
    Bitboard occupancy_white = 0, occupancy_black = 0;
    for (int i = wP; i <= wK; i++) occupancy_white |= bitboards[i];
    for (int i = bP; i <= bK; i++) occupancy_black |= bitboards[i];
    Bitboard occupancy = occupancy_white | occupancy_black;
    
    Bitboard occupancy_friend = (side == WHITE) ? occupancy_white : occupancy_black;
    Bitboard occupancy_enemy  = (side == WHITE) ? occupancy_black : occupancy_white;

    // Boucle sur toutes les pièces du camp qui joue
    for (int piece = (side == WHITE ? wP : bP); piece <= (side == WHITE ? wK : bK); piece++) {
        bitboard = bitboards[piece];

        // ---------------- PIONS ----------------
        if (piece == wP || piece == bP) {
            while (bitboard) {
                src = __builtin_ctzll(bitboard);
                
                // Direction et Rangs
                int direction = (side == WHITE) ? 8 : -8;
                int start_rank_min = (side == WHITE) ? A2 : A7; // Pour double push
                int start_rank_max = (side == WHITE) ? H2 : H7;
                int prom_rank_min  = (side == WHITE) ? A7 : A2; // Pour promotion
                int prom_rank_max  = (side == WHITE) ? H7 : H2;

                tgt = src + direction;

                // 1. POUSSÉE SIMPLE (Si case vide)
                if (tgt >= 0 && tgt < 64 && !GET_BIT(occupancy, tgt)) {
                    // Promotion ?
                    if (src >= prom_rank_min && src <= prom_rank_max) {
                        int prom_flags[] = {PROMOTION_QUEEN, PROMOTION_ROOK, PROMOTION_BISHOP, PROMOTION_KNIGHT};
                        int prom_pieces[] = {(side==WHITE?wQ:bQ), (side==WHITE?wR:bR), (side==WHITE?wB:bB), (side==WHITE?wN:bN)};
                        for(int k=0; k<4; k++) add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, prom_pieces[k], prom_flags[k]));
                    } else {
                        add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, QUIET_MOVE));
                        
                        // 2. DOUBLE POUSSÉE (Si rang départ et chemin libre)
                        if ((src >= start_rank_min && src <= start_rank_max) && !GET_BIT(occupancy, tgt + direction)) {
                            add_move(move_list, move_count, ENCODE_MOVE(src, tgt + direction, piece, 0, DOUBLE_PAWN_PUSH));
                        }
                    }
                }

                // 3. CAPTURES (Diagonales)
                Bitboard attacks_bb = 0;
                if (side == WHITE) {
                    if (GET_BIT(bitboards[wP], src) && (src % 8) != 0) SET_BIT(attacks_bb, src + 7); // Gauche (exclut col A)
                    if (GET_BIT(bitboards[wP], src) && (src % 8) != 7) SET_BIT(attacks_bb, src + 9); // Droite (exclut col H)
                    // Note: Le if GET_BIT est redondant car on boucle sur le bitboard, mais clarifie l'idée.
                    // Mieux: utiliser tables pré-calculées, mais ici on fait calcul direct.
                    attacks_bb = 0; // Reset pour calcul propre
                    if ((src % 8) != 0) SET_BIT(attacks_bb, src + 7); // Capture Gauche
                    if ((src % 8) != 7) SET_BIT(attacks_bb, src + 9); // Capture Droite
                } else {
                    if ((src % 8) != 7) SET_BIT(attacks_bb, src - 7); // Capture Droite (pour noirs)
                    if ((src % 8) != 0) SET_BIT(attacks_bb, src - 9); // Capture Gauche (pour noirs)
                }

                while (attacks_bb) {
                    tgt = __builtin_ctzll(attacks_bb);
                    
                    // Capture normale
                    if (GET_BIT(occupancy_enemy, tgt)) {
                         if (src >= prom_rank_min && src <= prom_rank_max) {
                            int prom_flags[] = {PROMOTION_CAPTURE, PROMOTION_CAPTURE, PROMOTION_CAPTURE, PROMOTION_CAPTURE}; // Simplifié
                            // En vrai il faudrait distinguer Prom+Capture, ici on utilise un flag générique ou on combine
                            int prom_pieces[] = {(side==WHITE?wQ:bQ), (side==WHITE?wR:bR), (side==WHITE?wB:bB), (side==WHITE?wN:bN)};
                            for(int k=0; k<4; k++) add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, prom_pieces[k], PROMOTION_CAPTURE));
                        } else {
                            add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, CAPTURE_FLAG));
                        }
                    }
                    // Prise en passant
                    else if (en_passant_sq != 0 && tgt == en_passant_sq) {
                        add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, EP_CAPTURE));
                    }
                    
                    POP_BIT(attacks_bb, tgt);
                }

                POP_BIT(bitboard, src);
            }
        }

        // ---------------- CAVALIERS & ROIS (Leapers) ----------------
        else if (piece == wN || piece == bN || piece == wK || piece == bK) {
            while (bitboard) {
                src = __builtin_ctzll(bitboard);
                // On utilise nos tables pré-calculées
                attacks = (piece == wN || piece == bN) ? knight_attacks[src] : king_attacks[src];
                
                // On ne garde que les coups vers (vide OU ennemi) = Pas ami
                attacks &= ~occupancy_friend;

                while (attacks) {
                    tgt = __builtin_ctzll(attacks);
                    // Si ennemi = Capture, Sinon Quiet
                    int flag = GET_BIT(occupancy_enemy, tgt) ? CAPTURE_FLAG : QUIET_MOVE;
                    add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, flag));
                    POP_BIT(attacks, tgt);
                }

                // Cas Spécial : ROQUE (Castling) pour le Roi
                if (piece == wK || piece == bK) {
                    // Logique Roque (vérifier cases vides et droits)
                    // ... (Pour simplifier le code ici, je ne le remets pas en entier, 
                    // mais il faut vérifier que les cases F1/G1 ou B1/C1/D1 sont vides)
                    if (side == WHITE) {
                        if ((castle_rights & 1) && !GET_BIT(occupancy, F1) && !GET_BIT(occupancy, G1)) {
                            if (!is_square_attacked(E1, BLACK) && !is_square_attacked(F1, BLACK)) // G1 checké par make_move
                                add_move(move_list, move_count, ENCODE_MOVE(E1, G1, wK, 0, KING_CASTLE));
                        }
                        if ((castle_rights & 2) && !GET_BIT(occupancy, B1) && !GET_BIT(occupancy, C1) && !GET_BIT(occupancy, D1)) {
                            if (!is_square_attacked(E1, BLACK) && !is_square_attacked(D1, BLACK))
                                add_move(move_list, move_count, ENCODE_MOVE(E1, C1, wK, 0, QUEEN_CASTLE));
                        }
                    } else {
                        // Idem pour noirs (bits 4 et 8)
                        if ((castle_rights & 4) && !GET_BIT(occupancy, F8) && !GET_BIT(occupancy, G8)) {
                            if (!is_square_attacked(E8, WHITE) && !is_square_attacked(F8, WHITE))
                                add_move(move_list, move_count, ENCODE_MOVE(E8, G8, bK, 0, KING_CASTLE));
                        }
                        if ((castle_rights & 8) && !GET_BIT(occupancy, B8) && !GET_BIT(occupancy, C8) && !GET_BIT(occupancy, D8)) {
                            if (!is_square_attacked(E8, WHITE) && !is_square_attacked(D8, WHITE))
                                add_move(move_list, move_count, ENCODE_MOVE(E8, C8, bK, 0, QUEEN_CASTLE));
                        }
                    }
                }
                
                POP_BIT(bitboard, src);
            }
        }

        // ---------------- PIÈCES GLISSANTES (Sliders) ----------------
        else {
            while (bitboard) {
                src = __builtin_ctzll(bitboard);
                
                if (piece == wB || piece == bB) attacks = get_bishop_attacks(src, occupancy);
                else if (piece == wR || piece == bR) attacks = get_rook_attacks(src, occupancy);
                else attacks = get_bishop_attacks(src, occupancy) | get_rook_attacks(src, occupancy); // Dame

                attacks &= ~occupancy_friend; // Bloqué par amis

                while (attacks) {
                    tgt = __builtin_ctzll(attacks);
                    int flag = GET_BIT(occupancy_enemy, tgt) ? CAPTURE_FLAG : QUIET_MOVE;
                    add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, flag));
                    POP_BIT(attacks, tgt);
                }
                POP_BIT(bitboard, src);
            }
        }
    }
}

// --- INTERPRÉTEUR DE TEXTE ---

// Parse juste les coordonnées (ex: "e2e4" -> from=12, to=28)
// Ne vérifie PAS la légalité, ne devine PAS les flags.
int parse_input_squares(char *str, int *from, int *to) {
    if (str[0] < 'a' || str[0] > 'h') return 0;
    if (str[1] < '1' || str[1] > '8') return 0;
    if (str[2] < 'a' || str[2] > 'h') return 0;
    if (str[3] < '1' || str[3] > '8') return 0;

    *from = (str[0] - 'a') + ((str[1] - '1') * 8);
    *to   = (str[2] - 'a') + ((str[3] - '1') * 8);
    return 1;
}

int main() {
    init_leapers_attacks();
    init_castling_masks();
    init_board();
    
    char input[6];
    MOVE move_list[256]; // Une liste pour stocker les coups générés
    int move_count = 0;

    printf("\n   === MOTEUR ECHECS (Mode Strict) ===\n");

    while (1) {
        print_board();
        
        // 1. Générer TOUS les coups possibles maintenant
        generate_moves(move_list, &move_count);

        // 2. Vérifier le MAT (Si 0 coups légaux)
        // Note: generate_moves produit du pseudo-legal. Pour le vrai mat, 
        // il faudrait tester make_move sur chacun. Pour l'instant on garde check simple.
        int king_sq = __builtin_ctzll((side == WHITE) ? bitboards[wK] : bitboards[bK]);
        if (is_square_attacked(king_sq, side ^ 1)) printf("\n   !!! ECHEC !!!\n");

        printf("\nCoup > ");
        if (scanf("%s", input) == EOF || strcmp(input, "quit") == 0) break;

        // 3. Convertir texte -> cases départ/arrivée
        int from, to;
        if (!parse_input_squares(input, &from, &to)) {
            printf("Format invalide (ex: e2e4)\n");
            continue;
        }

        // 4. CHERCHER LE COUP DANS LA LISTE GÉNÉRÉE
        int move_found = 0;
        MOVE chosen_move = 0;

        for (int i = 0; i < move_count; i++) {
            MOVE m = move_list[i];
            
            // On vérifie si ce coup correspond à ce que le joueur a tapé
            if (GET_MOVE_FROM(m) == from && GET_MOVE_TO(m) == to) {
                // CAS SPÉCIAL PROMOTION : Si c'est une promotion, l'utilisateur tape souvent "a7a8q"
                // Pour simplifier ici, si on trouve une promotion, on prend la Dame par défaut
                // ou on vérifie si c'est bien une promotion.
                chosen_move = m;
                move_found = 1;
                break; // On a trouvé le coup valide !
            }
        }

        if (!move_found) {
            printf("   /!\\ Coup impossible (interdit par les règles ou pièce bloquée).\n");
            continue;
        }

        // 5. Jouer le coup trouvé (qui contient les bons flags générés par le moteur)
        if (make_move(chosen_move, ALL_MOVES) == 0) {
            printf("   /!\\ Coup ILLEGAL (Laisse le Roi en echec).\n");
        } else {
            printf("   Coup joue.\n");
        }
    }
    return 0;
}