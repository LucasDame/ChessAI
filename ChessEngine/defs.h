#ifndef DEFS_H
#define DEFS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <inttypes.h>

// --- TYPES ---
typedef uint64_t Bitboard;
typedef uint32_t MOVE;

// --- ENUMS ---
enum { A1, B1, C1, D1, E1, F1, G1, H1, A2, B2, C2, D2, E2, F2, G2, H2, A3, B3, C3, D3, E3, F3, G3, H3, A4, B4, C4, D4, E4, F4, G4, H4, A5, B5, C5, D5, E5, F5, G5, H5, A6, B6, C6, D6, E6, F6, G6, H6, A7, B7, C7, D7, E7, F7, G7, H7, A8, B8, C8, D8, E8, F8, G8, H8 };
enum { wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK };
enum { WHITE, BLACK, BOTH };
enum { QUIET_MOVE=0, DOUBLE_PAWN_PUSH=1, KING_CASTLE=2, QUEEN_CASTLE=3, CAPTURE_FLAG=4, EP_CAPTURE=5, PROMOTION_KNIGHT=8, PROMOTION_BISHOP=9, PROMOTION_ROOK=10, PROMOTION_QUEEN=11, PROMOTION_CAPTURE=12 };
enum { ALL_MOVES, ONLY_CAPTURES };

// --- MACROS ---
#define SET_BIT(bb, sq) ((bb) |= (1ULL << (sq)))
#define GET_BIT(bb, sq) ((bb) & (1ULL << (sq)))
#define POP_BIT(bb, sq) ((bb) &= ~(1ULL << (sq)))
#define COUNT_BITS(bb) __builtin_popcountll(bb)

#define ENCODE_MOVE(from, to, piece, promoted, flag) ( (from) | ((to) << 6) | ((piece) << 12) | ((promoted) << 16) | ((flag) << 20) )
#define GET_MOVE_FROM(move)      ((move) & 0x3f)
#define GET_MOVE_TO(move)        (((move) >> 6) & 0x3f)
#define GET_MOVE_PIECE(move)     (((move) >> 12) & 0xf)
#define GET_MOVE_PROMOTED(move)  (((move) >> 16) & 0xf)
#define GET_MOVE_FLAG(move)      (((move) >> 20) & 0xf)

// --- CONSTANTES ---
#define MAX_GAME_MOVES 2048
#define INFINITY 50000
#define MATE_SCORE 49000

extern const Bitboard MASK_A_FILE;
extern const Bitboard MASK_H_FILE;
extern const Bitboard MASK_AB_FILE;
extern const Bitboard MASK_GH_FILE;

// --- GLOBALES (extern = définies ailleurs) ---
typedef struct {
    Bitboard bitboards[12];
    int side;
    int en_passant_sq;
    int castle_rights;
} GameState;

extern Bitboard bitboards[12];
extern Bitboard knight_attacks[64];
extern Bitboard king_attacks[64];
extern int castling_rights_mask[64];
extern int side;
extern int en_passant_sq;
extern int castle_rights;
extern GameState history[MAX_GAME_MOVES];
extern int game_ply;

// --- PROTOTYPES DE FONCTIONS ---

// board.c
void init_board();
void save_board(GameState *state);
void restore_board(GameState *state);
void serialize_board(char *buffer);
int parse_input_squares(char *str, int *from, int *to, char *prom_char);

// move.c
void init_leapers_attacks();
void init_castling_masks();
Bitboard get_bishop_attacks(int sq, Bitboard occupancy);
Bitboard get_rook_attacks(int sq, Bitboard occupancy);
int is_square_attacked(int sq, int attacking_side);
void generate_moves(MOVE *move_list, int *move_count);
int make_move(MOVE move, int capture_mode);
int check_game_over();
int is_repetition();

// search.c
int evaluate();
MOVE search_best_move(int depth);

#endif