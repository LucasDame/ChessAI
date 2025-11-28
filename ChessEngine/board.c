#include "defs.h"

// ============================================================================
//                       DÉFINITION DES VARIABLES GLOBALES
// ============================================================================
// (Elles sont déclarées 'extern' dans defs.h, ici on réserve la mémoire)

Bitboard bitboards[12];
Bitboard knight_attacks[64];
Bitboard king_attacks[64];
int castling_rights_mask[64];
int side = WHITE;
int en_passant_sq = 0;
int castle_rights = 15; // 1111 (KQkq)

// Historique de la partie
GameState history[MAX_GAME_MOVES];
int game_ply = 0;

// Masques de colonnes (utiles pour éviter les débordements de bitboards)
const Bitboard MASK_A_FILE = 0x0101010101010101ULL;
const Bitboard MASK_H_FILE = 0x8080808080808080ULL;
const Bitboard MASK_AB_FILE = 0x0303030303030303ULL;
const Bitboard MASK_GH_FILE = 0xC0C0C0C0C0C0C0C0ULL;

// ============================================================================
//                          GESTION DU PLATEAU
// ============================================================================

void init_board() {
    // 1. Nettoyage de la mémoire
    memset(bitboards, 0, sizeof(bitboards));
    memset(history, 0, sizeof(history));
    game_ply = 0;

    // 2. Placement des pièces Blanches
    SET_BIT(bitboards[wR], A1); SET_BIT(bitboards[wR], H1);
    SET_BIT(bitboards[wN], B1); SET_BIT(bitboards[wN], G1);
    SET_BIT(bitboards[wB], C1); SET_BIT(bitboards[wB], F1);
    SET_BIT(bitboards[wQ], D1); 
    SET_BIT(bitboards[wK], E1);
    
    // Pions blancs (Rangée 2 : A2 à H2)
    for (int i = A2; i <= H2; i++) {
        SET_BIT(bitboards[wP], i);
    }

    // 3. Placement des pièces Noires
    SET_BIT(bitboards[bR], A8); SET_BIT(bitboards[bR], H8);
    SET_BIT(bitboards[bN], B8); SET_BIT(bitboards[bN], G8);
    SET_BIT(bitboards[bB], C8); SET_BIT(bitboards[bB], F8);
    SET_BIT(bitboards[bQ], D8); 
    SET_BIT(bitboards[bK], E8);

    // Pions noirs (Rangée 7 : A7 à H7)
    for (int i = A7; i <= H7; i++) {
        SET_BIT(bitboards[bP], i);
    }

    // 4. Initialisation des états
    side = WHITE;
    castle_rights = 15; // Tout le monde peut roquer
    en_passant_sq = 0;  // Pas de prise en passant possible au début
}

// Sauvegarde l'état actuel dans une structure (pour Undo ou Recherche)
void save_board(GameState *state) {
    memcpy(state->bitboards, bitboards, 12 * sizeof(Bitboard));
    state->side = side;
    state->en_passant_sq = en_passant_sq;
    state->castle_rights = castle_rights;
}

// Restaure l'état depuis une structure
void restore_board(GameState *state) {
    memcpy(bitboards, state->bitboards, 12 * sizeof(Bitboard));
    side = state->side;
    en_passant_sq = state->en_passant_sq;
    castle_rights = state->castle_rights;
}

// ============================================================================
//                          COMMUNICATION / PARSING
// ============================================================================

// Convertit l'état du plateau en une chaîne de 64 caractères pour Python
// Ordre : A8..H8, A7..H7, ..., A1..H1 (Lecture visuelle)
void serialize_board(char *buffer) {
    char *piece_chars = "PNBRQKpnbrqk"; // Correspond aux indices 0-11
    int idx = 0;

    // Parcours de la Rangée 8 (haut) vers la Rangée 1 (bas)
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            int piece_found = -1;

            // Quel type de pièce est sur cette case ?
            for (int p = wP; p <= bK; p++) {
                if (GET_BIT(bitboards[p], sq)) {
                    piece_found = p;
                    break;
                }
            }

            if (piece_found != -1) {
                buffer[idx] = piece_chars[piece_found];
            } else {
                buffer[idx] = '-'; // Case vide
            }
            idx++;
        }
    }
    buffer[idx] = '\0'; // Fin de chaîne
}

// Convertit une commande "e2e4" ou "a7a8q" en indices et char de promotion
// Retourne 1 si succès, 0 si format invalide
int parse_input_squares(char *str, int *from, int *to, char *prom_char) {
    // Vérification basique de la longueur et des bornes
    if (strlen(str) < 4) return 0;
    if (str[0] < 'a' || str[0] > 'h') return 0;
    if (str[1] < '1' || str[1] > '8') return 0;
    if (str[2] < 'a' || str[2] > 'h') return 0;
    if (str[3] < '1' || str[3] > '8') return 0;

    // Conversion : 'a'=0, '1'=0 pour le moteur
    // Formule : index = col + (row * 8)
    int f_col = str[0] - 'a';
    int f_row = str[1] - '1';
    int t_col = str[2] - 'a';
    int t_row = str[3] - '1';

    *from = f_col + (f_row * 8);
    *to   = t_col + (t_row * 8);

    // Gestion Promotion (5ème caractère)
    if (strlen(str) > 4) {
        *prom_char = str[4]; // ex: 'q', 'r', 'b', 'n'
    } else {
        *prom_char = 0;
    }

    return 1;
}