#include "defs.h"

// ============================================================================
//                              TABLES D'EVALUATION (PST)
// ============================================================================

// Pions : Encourager l'avancée vers la promotion
const int pawn_table[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
   50, 50, 50, 50, 50, 50, 50, 50,
   10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
};

// Cavaliers : Centre fort, bords faibles
const int knight_table[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};

// Fous : Grandes diagonales
const int bishop_table[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};

// Tours : 7ème rangée et colonnes centrales
const int rook_table[64] = {
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
};

// Roi : Sécurité avant tout
const int king_table[64] = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
};

// Scores Matériels
const int material_score[12] = {
    100, 320, 330, 500, 900, 20000, 
    -100, -320, -330, -500, -900, -20000 
};

// ============================================================================
//                              EVALUATION AGRESSIVE
// ============================================================================

int evaluate() {
    int score = 0;
    Bitboard bitboard, attacks;
    int sq;
    
    // Occupation globale pour calcul des sliders
    Bitboard occupancy = 0;
    for (int i=0; i<12; i++) occupancy |= bitboards[i];

    // --- BLANCS ---
    bitboard = bitboards[wP];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score += material_score[wP] + pawn_table[sq];
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[wN];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score += material_score[wN] + knight_table[sq];
        score += COUNT_BITS(knight_attacks[sq]) * 5; // Mobilité
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[wB];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score += material_score[wB] + bishop_table[sq];
        attacks = get_bishop_attacks(sq, occupancy);
        score += COUNT_BITS(attacks) * 5;
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[wR];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score += material_score[wR] + rook_table[sq];
        attacks = get_rook_attacks(sq, occupancy);
        score += COUNT_BITS(attacks) * 2;
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[wQ];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score += material_score[wQ] + bishop_table[sq]; // Dame utilise table fou
        attacks = get_bishop_attacks(sq, occupancy) | get_rook_attacks(sq, occupancy);
        score += COUNT_BITS(attacks) * 1;
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[wK];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score += material_score[wK] + king_table[sq];
        POP_BIT(bitboard, sq);
    }

    // --- NOIRS (Miroir) ---
    bitboard = bitboards[bP];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score -= (100 + pawn_table[sq ^ 56]); // Flip vertical
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[bN];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score -= (320 + knight_table[sq ^ 56]);
        score -= COUNT_BITS(knight_attacks[sq]) * 5;
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[bB];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score -= (330 + bishop_table[sq ^ 56]);
        attacks = get_bishop_attacks(sq, occupancy);
        score -= COUNT_BITS(attacks) * 5;
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[bR];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score -= (500 + rook_table[sq ^ 56]);
        attacks = get_rook_attacks(sq, occupancy);
        score -= COUNT_BITS(attacks) * 2;
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[bQ];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score -= (900 + bishop_table[sq ^ 56]);
        attacks = get_bishop_attacks(sq, occupancy) | get_rook_attacks(sq, occupancy);
        score -= COUNT_BITS(attacks) * 1;
        POP_BIT(bitboard, sq);
    }
    bitboard = bitboards[bK];
    while (bitboard) {
        sq = __builtin_ctzll(bitboard);
        score -= (20000 + king_table[sq ^ 56]);
        POP_BIT(bitboard, sq);
    }

    return (side == WHITE) ? score : -score;
}

// ============================================================================
//                              RECHERCHE (IA)
// ============================================================================

// Déclaration de la fonction externe (définie dans move.c) pour vérifier la répétition
int is_repetition(); 

int negamax(int depth, int alpha, int beta) {
    // 1. Détection Répétition dans la recherche
    // Si la position actuelle est une répétition, c'est nul (score 0)
    if (is_repetition()) {
        return 0;
    }

    if (depth == 0) {
        return evaluate();
    }

    MOVE move_list[256];
    int move_count = 0;
    generate_moves(move_list, &move_count);

    if (move_count == 0) {
        int king_sq = __builtin_ctzll((side == WHITE) ? bitboards[wK] : bitboards[bK]);
        if (is_square_attacked(king_sq, side ^ 1)) {
            return -MATE_SCORE + depth; // Mat favorisé au plus tôt
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

        // --- SIMULATION HISTORIQUE ---
        // On sauvegarde l'état qu'on quitte dans l'historique
        history[game_ply] = copy; 
        game_ply++; 
        // -----------------------------

        int eval = -negamax(depth - 1, -beta, -alpha);
        
        // --- ANNULATION HISTORIQUE ---
        game_ply--;
        // -----------------------------

        restore_board(&copy);

        if (eval > max_eval) max_eval = eval;
        if (eval > alpha) alpha = eval;
        if (alpha >= beta) break; // Coupure Alpha-Beta
    }
    return max_eval;
}

MOVE search_best_move(int depth) {
    MOVE best_move = 0;
    int max_eval = -INFINITY;
    int alpha = -INFINITY;
    int beta = INFINITY;

    MOVE move_list[256];
    int move_count = 0;
    generate_moves(move_list, &move_count);
    
    printf("[MOTEUR] Recherche profondeur %d (%d coups)...\n", depth, move_count);

    for (int i = 0; i < move_count; i++) {
        GameState copy;
        save_board(&copy);
        
        if (make_move(move_list[i], ALL_MOVES) == 0) continue;

        // --- SIMULATION HISTORIQUE ---
        history[game_ply] = copy;
        game_ply++;
        // -----------------------------

        int eval = -negamax(depth - 1, -beta, -alpha);
        
        // --- ANNULATION HISTORIQUE ---
        game_ply--;
        // -----------------------------

        restore_board(&copy);

        if (eval > max_eval) {
            max_eval = eval;
            best_move = move_list[i];
        }
        if (eval > alpha) alpha = eval;
    }
    
    printf("[MOTEUR] Meilleur coup trouve. Eval: %d\n", max_eval);
    return best_move;
}