#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>     // Pour close()
#include <sys/socket.h> // Pour socket(), bind(), listen(), accept()
#include <arpa/inet.h>  // Pour sockaddr_in, inet_addr

#define PORT 12345
#define BUFFER_SIZE 1024

// ============================================================================
//                              DEFINITIONS
// ============================================================================

typedef uint64_t Bitboard;
typedef uint32_t MOVE;

enum { A1, B1, C1, D1, E1, F1, G1, H1, A2, B2, C2, D2, E2, F2, G2, H2, A3, B3, C3, D3, E3, F3, G3, H3, A4, B4, C4, D4, E4, F4, G4, H4, A5, B5, C5, D5, E5, F5, G5, H5, A6, B6, C6, D6, E6, F6, G6, H6, A7, B7, C7, D7, E7, F7, G7, H7, A8, B8, C8, D8, E8, F8, G8, H8 };
enum { wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK };
enum { WHITE, BLACK, BOTH };
enum { QUIET_MOVE=0, DOUBLE_PAWN_PUSH=1, KING_CASTLE=2, QUEEN_CASTLE=3, CAPTURE_FLAG=4, EP_CAPTURE=5, PROMOTION_KNIGHT=8, PROMOTION_BISHOP=9, PROMOTION_ROOK=10, PROMOTION_QUEEN=11, PROMOTION_CAPTURE=12 };
enum { ALL_MOVES, ONLY_CAPTURES };

// Masques
const Bitboard MASK_A_FILE = 0x0101010101010101ULL;
const Bitboard MASK_H_FILE = 0x8080808080808080ULL;
const Bitboard MASK_AB_FILE = 0x0303030303030303ULL;
const Bitboard MASK_GH_FILE = 0xC0C0C0C0C0C0C0C0ULL;

// Macros (avec 1ULL pour forcer 64 bits)
#define SET_BIT(bb, sq) ((bb) |= (1ULL << (sq)))
#define GET_BIT(bb, sq) ((bb) & (1ULL << (sq)))
#define POP_BIT(bb, sq) ((bb) &= ~(1ULL << (sq)))

// Encodage
#define ENCODE_MOVE(from, to, piece, promoted, flag) ( (from) | ((to) << 6) | ((piece) << 12) | ((promoted) << 16) | ((flag) << 20) )
#define GET_MOVE_FROM(move)      ((move) & 0x3f)
#define GET_MOVE_TO(move)        (((move) >> 6) & 0x3f)
#define GET_MOVE_PIECE(move)     (((move) >> 12) & 0xf)
#define GET_MOVE_PROMOTED(move)  (((move) >> 16) & 0xf)
#define GET_MOVE_FLAG(move)      (((move) >> 20) & 0xf)

// Globales
Bitboard bitboards[12];
Bitboard knight_attacks[64];
Bitboard king_attacks[64];
int castling_rights_mask[64];
int side = WHITE;
int en_passant_sq = 0;
int castle_rights = 15;

typedef struct {
    Bitboard bitboards[12];
    int side;
    int en_passant_sq;
    int castle_rights;
} GameState;

// ============================================================================
//                              INITIALISATION
// ============================================================================

void init_castling_masks() {
    for (int i = 0; i < 64; i++) castling_rights_mask[i] = 15;
    castling_rights_mask[A1] = 13; castling_rights_mask[H1] = 14; castling_rights_mask[E1] = 12;
    castling_rights_mask[A8] = 7;  castling_rights_mask[H8] = 11; castling_rights_mask[E8] = 3;
}

void init_leapers_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        // --- CAVALIER ---
        Bitboard knight_bit = (1ULL << sq);
        Bitboard k_moves = 0;
        
        // On n'applique le décalage QUE si la destination théorique est sur le plateau
        // Et on applique les masques de colonnes pour éviter le wrap-around
        
        if ((sq + 17) < 64) k_moves |= (knight_bit & ~MASK_H_FILE) << 17;
        if ((sq + 15) < 64) k_moves |= (knight_bit & ~MASK_A_FILE) << 15;
        if ((sq + 10) < 64) k_moves |= (knight_bit & ~MASK_GH_FILE) << 10;
        if ((sq + 6)  < 64) k_moves |= (knight_bit & ~MASK_AB_FILE) << 6;
        
        if ((sq - 17) >= 0) k_moves |= (knight_bit & ~MASK_A_FILE) >> 17;
        if ((sq - 15) >= 0) k_moves |= (knight_bit & ~MASK_H_FILE) >> 15;
        if ((sq - 10) >= 0) k_moves |= (knight_bit & ~MASK_AB_FILE) >> 10;
        if ((sq - 6)  >= 0) k_moves |= (knight_bit & ~MASK_GH_FILE) >> 6;
        
        knight_attacks[sq] = k_moves;

        // --- ROI ---
        Bitboard king_bit = (1ULL << sq);
        Bitboard r_moves = 0;
        
        if ((sq + 8) < 64) r_moves |= (king_bit << 8);
        if ((sq - 8) >= 0) r_moves |= (king_bit >> 8);
        if ((sq + 1) < 64) r_moves |= (king_bit & ~MASK_H_FILE) << 1; // Droite
        if ((sq - 1) >= 0) r_moves |= (king_bit & ~MASK_A_FILE) >> 1; // Gauche
        
        if ((sq + 9) < 64) r_moves |= (king_bit & ~MASK_H_FILE) << 9;
        if ((sq + 7) < 64) r_moves |= (king_bit & ~MASK_A_FILE) << 7;
        if ((sq - 7) >= 0) r_moves |= (king_bit & ~MASK_H_FILE) >> 7;
        if ((sq - 9) >= 0) r_moves |= (king_bit & ~MASK_A_FILE) >> 9;
        
        king_attacks[sq] = r_moves;
    }
}

void init_board() {
    memset(bitboards, 0, sizeof(bitboards));
    SET_BIT(bitboards[wR], A1); SET_BIT(bitboards[wR], H1);
    SET_BIT(bitboards[wN], B1); SET_BIT(bitboards[wN], G1);
    SET_BIT(bitboards[wB], C1); SET_BIT(bitboards[wB], F1);
    SET_BIT(bitboards[wQ], D1); SET_BIT(bitboards[wK], E1);
    for (int i = A2; i <= H2; i++) SET_BIT(bitboards[wP], i);
    SET_BIT(bitboards[bR], A8); SET_BIT(bitboards[bR], H8);
    SET_BIT(bitboards[bN], B8); SET_BIT(bitboards[bN], G8);
    SET_BIT(bitboards[bB], C8); SET_BIT(bitboards[bB], F8);
    SET_BIT(bitboards[bQ], D8); SET_BIT(bitboards[bK], E8);
    for (int i = A7; i <= H7; i++) SET_BIT(bitboards[bP], i);
    side = WHITE; castle_rights = 15; en_passant_sq = 0;
}

// ============================================================================
//                              LOGIQUE
// ============================================================================

Bitboard get_bishop_attacks(int sq, Bitboard occupancy) {
    Bitboard attacks = 0;
    int r, f, tr = sq / 8, tf = sq % 8;
    for (r=tr+1, f=tf+1; r<=7 && f<=7; r++, f++) { attacks |= (1ULL<<(r*8+f)); if((1ULL<<(r*8+f)) & occupancy) break; }
    for (r=tr+1, f=tf-1; r<=7 && f>=0; r++, f--) { attacks |= (1ULL<<(r*8+f)); if((1ULL<<(r*8+f)) & occupancy) break; }
    for (r=tr-1, f=tf+1; r>=0 && f<=7; r--, f++) { attacks |= (1ULL<<(r*8+f)); if((1ULL<<(r*8+f)) & occupancy) break; }
    for (r=tr-1, f=tf-1; r>=0 && f>=0; r--, f--) { attacks |= (1ULL<<(r*8+f)); if((1ULL<<(r*8+f)) & occupancy) break; }
    return attacks;
}

Bitboard get_rook_attacks(int sq, Bitboard occupancy) {
    Bitboard attacks = 0;
    int r, f, tr = sq / 8, tf = sq % 8;
    for (r=tr+1; r<=7; r++) { attacks |= (1ULL<<(r*8+tf)); if((1ULL<<(r*8+tf)) & occupancy) break; }
    for (r=tr-1; r>=0; r--) { attacks |= (1ULL<<(r*8+tf)); if((1ULL<<(r*8+tf)) & occupancy) break; }
    for (f=tf+1; f<=7; f++) { attacks |= (1ULL<<(tr*8+f)); if((1ULL<<(tr*8+f)) & occupancy) break; }
    for (f=tf-1; f>=0; f--) { attacks |= (1ULL<<(tr*8+f)); if((1ULL<<(tr*8+f)) & occupancy) break; }
    return attacks;
}

// --- VERSION DE BOGAGE INCLUSE ---
int is_square_attacked(int sq, int attacking_side) {
    Bitboard occupancy = 0;
    for (int i=0; i<12; i++) occupancy |= bitboards[i];

    int pP = (attacking_side == WHITE) ? wP : bP;
    int pN = (attacking_side == WHITE) ? wN : bN;
    int pK = (attacking_side == WHITE) ? wK : bK;
    int pB = (attacking_side == WHITE) ? wB : bB;
    int pR = (attacking_side == WHITE) ? wR : bR;
    int pQ = (attacking_side == WHITE) ? wQ : bQ;

    if (attacking_side == WHITE) {
        if ( ((1ULL << sq) >> 9) & bitboards[pP] & ~MASK_H_FILE ) return 1;
        if ( ((1ULL << sq) >> 7) & bitboards[pP] & ~MASK_A_FILE ) return 1;
    } else {
        if ( ((1ULL << sq) << 9) & bitboards[pP] & ~MASK_A_FILE ) return 1;
        if ( ((1ULL << sq) << 7) & bitboards[pP] & ~MASK_H_FILE ) return 1;
    }

    if (knight_attacks[sq] & bitboards[pN]) return 1;

    if (king_attacks[sq] & bitboards[pK]) return 1;

    Bitboard bishop_attacks = get_bishop_attacks(sq, occupancy);
    if (bishop_attacks & (bitboards[pB] | bitboards[pQ])) return 1;

    Bitboard rook_attacks = get_rook_attacks(sq, occupancy);
    if (rook_attacks & (bitboards[pR] | bitboards[pQ])) return 1;

    return 0;
}

void save_board(GameState *state) {
    memcpy(state->bitboards, bitboards, 12 * sizeof(Bitboard));
    state->side = side; state->en_passant_sq = en_passant_sq; state->castle_rights = castle_rights;
}
void restore_board(GameState *state) {
    memcpy(bitboards, state->bitboards, 12 * sizeof(Bitboard));
    side = state->side; en_passant_sq = state->en_passant_sq; castle_rights = state->castle_rights;
}

int make_move(MOVE move, int capture_mode) {
    GameState backup;
    save_board(&backup);

    int from = GET_MOVE_FROM(move);
    int to = GET_MOVE_TO(move);
    int piece = GET_MOVE_PIECE(move);
    int promoted = GET_MOVE_PROMOTED(move);
    int flag = GET_MOVE_FLAG(move);

    if (capture_mode == ONLY_CAPTURES && !(flag == CAPTURE_FLAG || flag == EP_CAPTURE || flag == PROMOTION_CAPTURE)) return 0;

    POP_BIT(bitboards[piece], from);
    SET_BIT(bitboards[piece], to);

    if (flag == CAPTURE_FLAG || flag == PROMOTION_CAPTURE) {
        int start_p = (side == WHITE) ? bP : wP;
        int end_p   = (side == WHITE) ? bK : wK;
        for (int p = start_p; p <= end_p; p++) if (GET_BIT(bitboards[p], to)) { POP_BIT(bitboards[p], to); break; }
    }
    if (promoted) { POP_BIT(bitboards[piece], to); SET_BIT(bitboards[promoted], to); }
    if (flag == EP_CAPTURE) {
        int ep_pawn_sq = (side == WHITE) ? (to - 8) : (to + 8);
        int enemy_pawn = (side == WHITE) ? bP : wP;
        POP_BIT(bitboards[enemy_pawn], ep_pawn_sq);
    }
    if (flag == KING_CASTLE) {
        if (side == WHITE) { POP_BIT(bitboards[wR], H1); SET_BIT(bitboards[wR], F1); } else { POP_BIT(bitboards[bR], H8); SET_BIT(bitboards[bR], F8); }
    } else if (flag == QUEEN_CASTLE) {
        if (side == WHITE) { POP_BIT(bitboards[wR], A1); SET_BIT(bitboards[wR], D1); } else { POP_BIT(bitboards[bR], A8); SET_BIT(bitboards[bR], D8); }
    }

    en_passant_sq = 0;
    if (flag == DOUBLE_PAWN_PUSH) en_passant_sq = (from + to) / 2;

    castle_rights &= castling_rights_mask[from];
    castle_rights &= castling_rights_mask[to];
    side ^= 1;

    int king_sq = __builtin_ctzll((side == WHITE) ? bitboards[wK] : bitboards[bK]); // Roi du camp qui a joué (maintenant passif) ? Non, side a changé
    // Attends, ici on check si le roi du coté 'side' (qui vient de changer, donc le joueur adverse) est en echec ? 
    // NON. make_move doit vérifier si le joueur qui A JOUÉ (side^1) s'est mis en échec.
    if (is_square_attacked(__builtin_ctzll((side==WHITE)?bitboards[bK]:bitboards[wK]), side)) { // On vérifie si le roi du joueur qui a joué est attaqué par 'side' (le nouveau joueur)
        restore_board(&backup);
        return 0;
    }
    return 1;
}

void print_board() {
    printf("\n   +---+---+---+---+---+---+---+---+\n");
    for (int rank = 7; rank >= 0; rank--) {
        printf(" %d |", rank + 1);
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            char *piece_char = "   ";
            if (GET_BIT(bitboards[wP], sq)) piece_char = " P "; else if (GET_BIT(bitboards[wN], sq)) piece_char = " N ";
            else if (GET_BIT(bitboards[wB], sq)) piece_char = " B "; else if (GET_BIT(bitboards[wR], sq)) piece_char = " R ";
            else if (GET_BIT(bitboards[wQ], sq)) piece_char = " Q "; else if (GET_BIT(bitboards[wK], sq)) piece_char = " K ";
            else if (GET_BIT(bitboards[bP], sq)) piece_char = " p "; else if (GET_BIT(bitboards[bN], sq)) piece_char = " n ";
            else if (GET_BIT(bitboards[bB], sq)) piece_char = " b "; else if (GET_BIT(bitboards[bR], sq)) piece_char = " r ";
            else if (GET_BIT(bitboards[bQ], sq)) piece_char = " q "; else if (GET_BIT(bitboards[bK], sq)) piece_char = " k ";
            printf("%s|", piece_char);
        }
        printf("\n   +---+---+---+---+---+---+---+---+\n");
    }
    printf("     a   b   c   d   e   f   g   h\n");
}

void add_move(MOVE *move_list, int *move_count, int move) {
    move_list[*move_count] = move;
    (*move_count)++;
}

void generate_moves(MOVE *move_list, int *move_count) {
    *move_count = 0;
    int src, tgt;
    Bitboard bitboard, attacks;
    
    Bitboard occupancy_white = 0, occupancy_black = 0;
    for (int i = wP; i <= wK; i++) occupancy_white |= bitboards[i];
    for (int i = bP; i <= bK; i++) occupancy_black |= bitboards[i];
    Bitboard occupancy = occupancy_white | occupancy_black;
    Bitboard occupancy_friend = (side == WHITE) ? occupancy_white : occupancy_black;
    Bitboard occupancy_enemy  = (side == WHITE) ? occupancy_black : occupancy_white;

    for (int piece = (side == WHITE ? wP : bP); piece <= (side == WHITE ? wK : bK); piece++) {
        bitboard = bitboards[piece];

        if (piece == wP || piece == bP) {
            while (bitboard) {
                src = __builtin_ctzll(bitboard);
                int direction = (side == WHITE) ? 8 : -8;
                int start_rank_min = (side == WHITE) ? A2 : A7;
                int start_rank_max = (side == WHITE) ? H2 : H7;
                int prom_rank_min  = (side == WHITE) ? A7 : A2;
                int prom_rank_max  = (side == WHITE) ? H7 : H2;
                tgt = src + direction;

                if (tgt >= 0 && tgt < 64 && !GET_BIT(occupancy, tgt)) {
                    if (src >= prom_rank_min && src <= prom_rank_max) {
                        int prom_flags[] = {PROMOTION_QUEEN, PROMOTION_ROOK, PROMOTION_BISHOP, PROMOTION_KNIGHT};
                        int prom_pieces[] = {(side==WHITE?wQ:bQ), (side==WHITE?wR:bR), (side==WHITE?wB:bB), (side==WHITE?wN:bN)};
                        for(int k=0; k<4; k++) add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, prom_pieces[k], prom_flags[k]));
                    } else {
                        add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, QUIET_MOVE));
                        if ((src >= start_rank_min && src <= start_rank_max) && !GET_BIT(occupancy, tgt + direction)) {
                            add_move(move_list, move_count, ENCODE_MOVE(src, tgt + direction, piece, 0, DOUBLE_PAWN_PUSH));
                        }
                    }
                }
                Bitboard attacks_bb = 0;
                if (side == WHITE) {
                    if ((src % 8) != 0) SET_BIT(attacks_bb, src + 7);
                    if ((src % 8) != 7) SET_BIT(attacks_bb, src + 9);
                } else {
                    if ((src % 8) != 7) SET_BIT(attacks_bb, src - 7);
                    if ((src % 8) != 0) SET_BIT(attacks_bb, src - 9);
                }
                while (attacks_bb) {
                    tgt = __builtin_ctzll(attacks_bb);
                    if (GET_BIT(occupancy_enemy, tgt)) {
                         if (src >= prom_rank_min && src <= prom_rank_max) {
                            int prom_pieces[] = {(side==WHITE?wQ:bQ), (side==WHITE?wR:bR), (side==WHITE?wB:bB), (side==WHITE?wN:bN)};
                            for(int k=0; k<4; k++) add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, prom_pieces[k], PROMOTION_CAPTURE));
                        } else add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, CAPTURE_FLAG));
                    } else if (en_passant_sq != 0 && tgt == en_passant_sq) add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, EP_CAPTURE));
                    POP_BIT(attacks_bb, tgt);
                }
                POP_BIT(bitboard, src);
            }
        } else if (piece == wN || piece == bN || piece == wK || piece == bK) {
            while (bitboard) {
                src = __builtin_ctzll(bitboard);
                attacks = (piece == wN || piece == bN) ? knight_attacks[src] : king_attacks[src];
                attacks &= ~occupancy_friend;
                while (attacks) {
                    tgt = __builtin_ctzll(attacks);
                    int flag = GET_BIT(occupancy_enemy, tgt) ? CAPTURE_FLAG : QUIET_MOVE;
                    add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, flag));
                    POP_BIT(attacks, tgt);
                }
                if (piece == wK || piece == bK) {
                    if (side == WHITE) {
                        if ((castle_rights & 1) && !GET_BIT(occupancy, F1) && !GET_BIT(occupancy, G1) && !is_square_attacked(E1, BLACK) && !is_square_attacked(F1, BLACK)) add_move(move_list, move_count, ENCODE_MOVE(E1, G1, wK, 0, KING_CASTLE));
                        if ((castle_rights & 2) && !GET_BIT(occupancy, B1) && !GET_BIT(occupancy, C1) && !GET_BIT(occupancy, D1) && !is_square_attacked(E1, BLACK) && !is_square_attacked(D1, BLACK)) add_move(move_list, move_count, ENCODE_MOVE(E1, C1, wK, 0, QUEEN_CASTLE));
                    } else {
                        if ((castle_rights & 4) && !GET_BIT(occupancy, F8) && !GET_BIT(occupancy, G8) && !is_square_attacked(E8, WHITE) && !is_square_attacked(F8, WHITE)) add_move(move_list, move_count, ENCODE_MOVE(E8, G8, bK, 0, KING_CASTLE));
                        if ((castle_rights & 8) && !GET_BIT(occupancy, B8) && !GET_BIT(occupancy, C8) && !GET_BIT(occupancy, D8) && !is_square_attacked(E8, WHITE) && !is_square_attacked(D8, WHITE)) add_move(move_list, move_count, ENCODE_MOVE(E8, C8, bK, 0, QUEEN_CASTLE));
                    }
                }
                POP_BIT(bitboard, src);
            }
        } else {
            while (bitboard) {
                src = __builtin_ctzll(bitboard);
                if (piece == wB || piece == bB) attacks = get_bishop_attacks(src, occupancy);
                else if (piece == wR || piece == bR) attacks = get_rook_attacks(src, occupancy);
                else attacks = get_bishop_attacks(src, occupancy) | get_rook_attacks(src, occupancy);
                attacks &= ~occupancy_friend;
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

// Retourne : 0 (Jeu en cours), 1 (Mat), 2 (Pat/Nul)
int check_game_over() {
    MOVE move_list[256];
    int move_count = 0;
    generate_moves(move_list, &move_count);

    int legal_moves = 0;
    
    // On sauvegarde l'état actuel car on va simuler des coups
    GameState current_state;
    save_board(&current_state);

    for (int i = 0; i < move_count; i++) {
        // On tente de jouer le coup
        if (make_move(move_list[i], ALL_MOVES)) {
            // Si make_move renvoie 1, c'est que le coup est légal (pas d'échec au roi)
            legal_moves++;
            
            // IMPORTANT : On annule immédiatement le coup pour tester le suivant
            // make_move a modifié le plateau, il faut revenir en arrière
            restore_board(&current_state);
            
            // Dès qu'on trouve UN coup légal, le jeu n'est pas fini. Pas besoin de tester le reste.
            return 0; 
        }
    }

    // Si on arrive ici, c'est qu'aucun coup n'est légal.
    // Est-ce un Mat ou un Pat ?
    
    // On vérifie si le Roi du camp qui doit jouer (side) est actuellement attaqué
    int king_sq = __builtin_ctzll((side == WHITE) ? bitboards[wK] : bitboards[bK]);
    
    // Note: is_square_attacked prend le camp de l'ATTAQUANT en argument.
    // Si c'est aux Blancs de jouer (side == WHITE), on regarde si les Noirs (side ^ 1) attaquent.
    if (is_square_attacked(king_sq, side ^ 1)) {
        return 1; // ECHEC ET MAT
    } else {
        return 2; // PAT (Stalemate)
    }
}

// Modifié pour extraire la promotion (q, r, b, n) ou 0 si absent
int parse_input_squares(char *str, int *from, int *to, char *prom_char) {
    if (str[0] < 'a' || str[0] > 'h') return 0;
    if (str[1] < '1' || str[1] > '8') return 0;
    if (str[2] < 'a' || str[2] > 'h') return 0;
    if (str[3] < '1' || str[3] > '8') return 0;

    *from = (str[0] - 'a') + ((str[1] - '1') * 8);
    *to   = (str[2] - 'a') + ((str[3] - '1') * 8);
    
    // Vérifie s'il y a un 5ème caractère pour la promotion
    if (strlen(str) > 4) {
        *prom_char = str[4]; // ex: 'q', 'n'...
    } else {
        *prom_char = 0; // Pas de promotion spécifiée
    }
    return 1;
}

// ============================================================================
//                              INTELLIGENCE ARTIFICIELLE
// ============================================================================

#define INFINITY 50000
#define MATE_SCORE 49000
#define MATE_VALUE (MATE_SCORE - 100) // Seuil pour dire "c'est un mat"

// Scores matériels (P, N, B, R, Q, K)
const int material_score[12] = {
    100, 300, 300, 500, 1000, 20000, // Blancs
    -100, -300, -300, -500, -1000, -20000 // Noirs
};

// 1. Évaluation Statique (L'intuition)
int evaluate() {
    int score = 0;
    Bitboard bitboard;
    
    for (int piece = wP; piece <= bK; piece++) {
        bitboard = bitboards[piece];
        while (bitboard) {
            score += material_score[piece];
            POP_BIT(bitboard, __builtin_ctzll(bitboard));
        }
    }
    
    // Negamax : on retourne le score du point de vue du joueur actif
    return (side == WHITE) ? score : -score;
}

// 2. Recherche Récursive (Le Cerveau)
int negamax(int depth, int alpha, int beta) {
    if (depth == 0) {
        return evaluate();
    }

    MOVE move_list[256];
    int move_count = 0;
    generate_moves(move_list, &move_count);

    if (move_count == 0) {
        int king_sq = __builtin_ctzll((side == WHITE) ? bitboards[wK] : bitboards[bK]);
        if (is_square_attacked(king_sq, side ^ 1)) {
            return -MATE_SCORE + depth; // Mat favorisé si proche (depth grand)
        }
        return 0; // Pat
    }

    int max_eval = -INFINITY;

    for (int i = 0; i < move_count; i++) {
        GameState copy;
        save_board(&copy);
        
        if (make_move(move_list[i], ALL_MOVES) == 0) {
            continue; 
        }

        int eval = -negamax(depth - 1, -beta, -alpha);
        
        restore_board(&copy);

        if (eval > max_eval) max_eval = eval;
        if (eval > alpha) alpha = eval;
        if (alpha >= beta) break; // Coupure Beta
    }
    return max_eval;
}

// 3. Fonction Racine (Le Pilote)
// Cherche le meilleur coup à une profondeur donnée
MOVE search_best_move(int depth) {
    MOVE best_move = 0;
    int max_eval = -INFINITY;
    int alpha = -INFINITY;
    int beta = INFINITY;

    MOVE move_list[256];
    int move_count = 0;
    generate_moves(move_list, &move_count);
    
    printf("[MOTEUR] Recherche a profondeur %d sur %d coups...\n", depth, move_count);

    for (int i = 0; i < move_count; i++) {
        GameState copy;
        save_board(&copy);
        
        // Si le coup est illégal, on passe
        if (make_move(move_list[i], ALL_MOVES) == 0) {
            continue;
        }

        // Appel récursif
        int eval = -negamax(depth - 1, -beta, -alpha);
        
        restore_board(&copy);

        // Si on trouve mieux
        if (eval > max_eval) {
            max_eval = eval;
            best_move = move_list[i];
            
            // Optionnel : Afficher ce qu'il pense en temps réel
            // printf("info score cp %d currmove index %d\n", max_eval, i);
        }
        
        // Mise à jour de la borne basse (Alpha)
        if (eval > alpha) {
            alpha = eval;
        }
    }
    
    printf("[MOTEUR] Meilleur coup trouve : Score %d\n", max_eval);
    return best_move;
}

// --- SÉRIALISATION DU PLATEAU ---
// Remplit un buffer avec 64 caractères représentant le plateau (de A8 à H1)
void serialize_board(char *buffer) {
    char *piece_chars = "PNBRQKpnbrqk"; // Ordre: wP, wN, wB... bK
    int idx = 0;

    // On parcourt de la rangée 8 (index 7) à 1 (index 0) pour l'ordre visuel
    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int sq = rank * 8 + file;
            int piece_found = -1;

            // On cherche quelle pièce est sur cette case
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

int main() {
    // 1. Initialisations du Moteur
    init_leapers_attacks();
    init_castling_masks();
    init_board();

    // 2. Configuration du Serveur TCP
    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    socklen_t addrlen = sizeof(address);
    char buffer[BUFFER_SIZE] = {0};

    // Création du socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Echec creation socket");
        exit(EXIT_FAILURE);
    }

    // Options pour relancer le script rapidement sans erreur "Address already in use"
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY; // Ecoute sur localhost
    address.sin_port = htons(PORT);       // Port 12345

    // Bind
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Echec du bind");
        exit(EXIT_FAILURE);
    }

    // Listen
    if (listen(server_fd, 3) < 0) {
        perror("Echec du listen");
        exit(EXIT_FAILURE);
    }

    printf("[MOTEUR] Serveur en attente sur le port %d...\n", PORT);

    // Accept (Bloquant jusqu'à connexion de Python)
    if ((client_fd = accept(server_fd, (struct sockaddr *)&address, &addrlen)) < 0) {
        perror("Echec du accept");
        exit(EXIT_FAILURE);
    }
    printf("[MOTEUR] Client Python connecte !\n");

    // 3. Boucle Principale
    while (1) {
        memset(buffer, 0, BUFFER_SIZE);
        int valread = recv(client_fd, buffer, BUFFER_SIZE - 1, 0); // -1 pour garder une place pour \0
        
        if (valread <= 0) {
            printf("[MOTEUR] Client deconnecte.\n");
            break;
        }
        
        // 1. Forcer la fin de chaine (Sécurité)
        buffer[valread] = '\0';

        // 2. NETTOYAGE AGRESSIF (Trim right)
        // On enlève les \r, \n et les espaces à la fin
        while (valread > 0 && (buffer[valread-1] == '\n' || buffer[valread-1] == '\r' || buffer[valread-1] == ' ')) {
            buffer[valread-1] = '\0';
            valread--;
        }

        // 3. LOG DE DEBUG COMPLET (Pour voir les fantômes)
        // On affiche le mot ET ses codes ASCII
        printf("[CMD RECUE] : '%s' (Longueur: %d)\n", buffer, valread);
        printf("[DEBUG HEX] : ");
        for(int i=0; i<valread; i++) {
            printf("%d ", (int)buffer[i]); 
        }
        printf("\n");

        // 4. Traitement
        if (strcmp(buffer, "quit") == 0) {
            break;
        }
        
        // Ici on utilise strncmp pour être plus permissif (les 2 premiers chars sont 'g' et 'o')
        else if (strncmp(buffer, "go", 2) == 0) {
            
            // Lancer la recherche (Profondeur 3 ou 4)
            MOVE best = search_best_move(6);
            
            if (best != 0) {
                // Conversion du coup en texte (ex: "e2e4")
                int f = GET_MOVE_FROM(best);
                int t = GET_MOVE_TO(best);
                int p = GET_MOVE_PROMOTED(best);
                char move_str[6];
                sprintf(move_str, "%c%d%c%d", 'a'+(f%8), 1+(f/8), 'a'+(t%8), 1+(t/8));
                
                // Gestion de la promotion (ajouter la lettre)
                if (p) {
                    char promo_char = 'q';
                    if (p == wR || p == bR) promo_char = 'r';
                    else if (p == wB || p == bB) promo_char = 'b';
                    else if (p == wN || p == bN) promo_char = 'n';
                    
                    size_t len = strlen(move_str);
                    move_str[len] = promo_char;
                    move_str[len+1] = '\0';
                }
                
                // Jouer le coup sur le plateau interne du C
                make_move(best, ALL_MOVES);
                
                // Préparer la réponse : "bestmove:e2e4 board:rnbq..."
                char board_str[65];
                serialize_board(board_str);
                
                // Vérifier Mat/Pat
                int status = check_game_over();
                char response[256];
                
                if (status == 0) {
                    sprintf(response, "bestmove:%s board:%s", move_str, board_str);
                } else if (status == 1) {
                    sprintf(response, "bestmove:%s board:%s game_over:checkmate", move_str, board_str);
                } else {
                    sprintf(response, "bestmove:%s board:%s game_over:stalemate", move_str, board_str);
                }
                
                send(client_fd, response, strlen(response), 0);
            } else {
                // Pas de coup trouvé (Mat ou Pat avant même de jouer ?)
                send(client_fd, "bestmove:none", 13, 0);
            }
            continue; // On passe au tour suivant
        }

        // --- MODE HUMAIN : Coordonnées (ex: "e2e4") ---
        int from, to;
        char prom_char;
        
        // On essaie de parser comme un coup normal
        if (parse_input_squares(buffer, &from, &to, &prom_char)) {
            // Générer les coups légaux pour vérifier si c'est valide
            MOVE move_list[256];
            int move_count = 0;
            generate_moves(move_list, &move_count);
            
            int move_found = 0; 
            MOVE chosen_move = 0;
            
            // On cherche le coup correspondant dans la liste
            for (int i = 0; i < move_count; i++) {
                MOVE m = move_list[i];
                if (GET_MOVE_FROM(m) == from && GET_MOVE_TO(m) == to) {
                    // Vérification de la promotion
                    int promoted = GET_MOVE_PROMOTED(m);
                    if (promoted) {
                        if (prom_char != 0) {
                            // Le joueur a spécifié une promotion (ex: "a7a8q")
                            int is_match = 0;
                            if ((promoted == wQ || promoted == bQ) && prom_char == 'q') is_match = 1;
                            else if ((promoted == wR || promoted == bR) && prom_char == 'r') is_match = 1;
                            else if ((promoted == wB || promoted == bB) && prom_char == 'b') is_match = 1;
                            else if ((promoted == wN || promoted == bN) && prom_char == 'n') is_match = 1;
                            
                            if (is_match) { chosen_move = m; move_found = 1; break; }
                        } else {
                            // Par défaut Dame si non spécifié (cas rare via API)
                            if (promoted == wQ || promoted == bQ) { chosen_move = m; move_found = 1; break; }
                        }
                    } else {
                        chosen_move = m; move_found = 1; break;
                    }
                }
            }
            
            char response[256];
            if (move_found) {
                // Tenter de jouer le coup
                if (make_move(chosen_move, ALL_MOVES)) {
                    // Coup Valide : On renvoie le plateau
                    char board_str[65];
                    serialize_board(board_str);
                    
                    int status = check_game_over();
                    if (status == 0) sprintf(response, "board:%s", board_str);
                    else if (status == 1) sprintf(response, "board:%s game_over:checkmate", board_str);
                    else sprintf(response, "board:%s game_over:stalemate", board_str);
                    
                } else {
                    sprintf(response, "illegal_move_king_check");
                }
            } else {
                sprintf(response, "illegal_move_rules");
            }
            
            send(client_fd, response, strlen(response), 0);
        } else {
            // Commande non reconnue
            char *msg = "unknown_command";
            send(client_fd, msg, strlen(msg), 0);
        }
    }

    // 4. Fermeture
    close(client_fd);
    close(server_fd);
    return 0;
}