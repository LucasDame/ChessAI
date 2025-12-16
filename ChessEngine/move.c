#include "defs.h"

// Fonction helper pour ajouter un coup à la liste
void add_move(MOVE *move_list, int *move_count, int move) {
    move_list[*move_count] = move;
    (*move_count)++;
}

// ============================================================================
//                              INITIALISATIONS
// ============================================================================

void init_castling_masks() {
    for (int i = 0; i < 64; i++) castling_rights_mask[i] = 15;
    // Si on touche ces cases (départ ou arrivée), on perd les droits
    castling_rights_mask[A1] = 13; // Perd WQ (1101)
    castling_rights_mask[H1] = 14; // Perd WK (1110)
    castling_rights_mask[E1] = 12; // Perd Tout Blanc (1100)
    castling_rights_mask[A8] = 7;  // Perd BQ (0111)
    castling_rights_mask[H8] = 11; // Perd BK (1011)
    castling_rights_mask[E8] = 3;  // Perd Tout Noir (0011)
}

void init_leapers_attacks() {
    for (int sq = 0; sq < 64; sq++) {
        // --- CAVALIER (Avec correction B8) ---
        Bitboard knight_bit = (1ULL << sq);
        Bitboard k_moves = 0;
        
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
        if ((sq + 1) < 64) r_moves |= (king_bit & ~MASK_H_FILE) << 1;
        if ((sq - 1) >= 0) r_moves |= (king_bit & ~MASK_A_FILE) >> 1;
        
        if ((sq + 9) < 64) r_moves |= (king_bit & ~MASK_H_FILE) << 9;
        if ((sq + 7) < 64) r_moves |= (king_bit & ~MASK_A_FILE) << 7;
        if ((sq - 7) >= 0) r_moves |= (king_bit & ~MASK_H_FILE) >> 7;
        if ((sq - 9) >= 0) r_moves |= (king_bit & ~MASK_A_FILE) >> 9;
        
        king_attacks[sq] = r_moves;
    }
}

// ============================================================================
//                              ATTAQUES
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

int is_square_attacked(int sq, int attacking_side) {
    Bitboard occupancy = 0;
    for (int i=0; i<12; i++) occupancy |= bitboards[i];

    int pP = (attacking_side == WHITE) ? wP : bP;
    int pN = (attacking_side == WHITE) ? wN : bN;
    int pK = (attacking_side == WHITE) ? wK : bK;
    int pB = (attacking_side == WHITE) ? wB : bB;
    int pR = (attacking_side == WHITE) ? wR : bR;
    int pQ = (attacking_side == WHITE) ? wQ : bQ;

    // 1. Pions
    if (attacking_side == WHITE) {
        if ( ((1ULL << sq) >> 9) & bitboards[pP] & ~MASK_H_FILE ) return 1;
        if ( ((1ULL << sq) >> 7) & bitboards[pP] & ~MASK_A_FILE ) return 1;
    } else {
        if ( ((1ULL << sq) << 9) & bitboards[pP] & ~MASK_A_FILE ) return 1;
        if ( ((1ULL << sq) << 7) & bitboards[pP] & ~MASK_H_FILE ) return 1;
    }
    
    // 2. Cavaliers & Rois
    if (knight_attacks[sq] & bitboards[pN]) return 1;
    if (king_attacks[sq] & bitboards[pK]) return 1;
    
    // 3. Sliders (Fous, Tours, Dames)
    if (get_bishop_attacks(sq, occupancy) & (bitboards[pB] | bitboards[pQ])) return 1;
    if (get_rook_attacks(sq, occupancy) & (bitboards[pR] | bitboards[pQ])) return 1;

    return 0;
}

// ============================================================================
//                              EXECUTION DE COUP
// ============================================================================

int make_move(MOVE move, int capture_mode) {
    // 1. Sauvegarde pour restauration si illégal
    GameState backup;
    save_board(&backup);

    int from = GET_MOVE_FROM(move);
    int to = GET_MOVE_TO(move);
    int piece = GET_MOVE_PIECE(move);
    int promoted = GET_MOVE_PROMOTED(move);
    int flag = GET_MOVE_FLAG(move);

    // Filtre Captures
    if (capture_mode == ONLY_CAPTURES && !(flag == CAPTURE_FLAG || flag == EP_CAPTURE || flag == PROMOTION_CAPTURE)) return 0;

    // Déplacement de la pièce
    POP_BIT(bitboards[piece], from);
    SET_BIT(bitboards[piece], to);

    // Gérer les captures
    if (flag == CAPTURE_FLAG || flag == PROMOTION_CAPTURE) {
        int start_p = (side == WHITE) ? bP : wP;
        int end_p   = (side == WHITE) ? bK : wK;
        for (int p = start_p; p <= end_p; p++) if (GET_BIT(bitboards[p], to)) { POP_BIT(bitboards[p], to); break; }
    }
    
    // Gérer la Promotion
    if (promoted) {
        POP_BIT(bitboards[piece], to);
        SET_BIT(bitboards[promoted], to);
    }
    
    // Gérer En Passant
    if (flag == EP_CAPTURE) {
        int ep_pawn_sq = (side == WHITE) ? (to - 8) : (to + 8);
        int enemy_pawn = (side == WHITE) ? bP : wP;
        POP_BIT(bitboards[enemy_pawn], ep_pawn_sq);
    }
    
    // Gérer Roque
    if (flag == KING_CASTLE) {
        if (side == WHITE) { POP_BIT(bitboards[wR], H1); SET_BIT(bitboards[wR], F1); } else { POP_BIT(bitboards[bR], H8); SET_BIT(bitboards[bR], F8); }
    } else if (flag == QUEEN_CASTLE) {
        if (side == WHITE) { POP_BIT(bitboards[wR], A1); SET_BIT(bitboards[wR], D1); } else { POP_BIT(bitboards[bR], A8); SET_BIT(bitboards[bR], D8); }
    }

    // Mise à jour En Passant
    en_passant_sq = 0;
    if (flag == DOUBLE_PAWN_PUSH) en_passant_sq = (from + to) / 2;

    // Mise à jour Roque et Coté
    castle_rights &= castling_rights_mask[from];
    castle_rights &= castling_rights_mask[to];
    side ^= 1;

    // Vérification de la légalité : Le roi de celui qui a joué (side^1) est-il en échec ?
    // Note : 'side' vient de changer, donc 'side' est maintenant l'adversaire.
    // On vérifie si l'adversaire attaque le roi du joueur courant.
    // (J'ai supprimé l'ancienne variable 'king_sq' inutile ici)
    
    int my_king_sq = __builtin_ctzll((side == WHITE) ? bitboards[bK] : bitboards[wK]);
    
    if (is_square_attacked(my_king_sq, side)) {
        restore_board(&backup);
        return 0; // Illégal
    }

    return 1; // Légal
}

// ============================================================================
//                              GENERATION DE COUPS
// ============================================================================

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

        // --- PIONS ---
        if (piece == wP || piece == bP) {
            while (bitboard) {
                src = __builtin_ctzll(bitboard);
                int direction = (side == WHITE) ? 8 : -8;
                int start_rank_min = (side == WHITE) ? A2 : A7;
                int start_rank_max = (side == WHITE) ? H2 : H7;
                int prom_rank_min  = (side == WHITE) ? A7 : A2;
                int prom_rank_max  = (side == WHITE) ? H7 : H2;
                tgt = src + direction;

                // Avance simple
                if (tgt >= 0 && tgt < 64 && !GET_BIT(occupancy, tgt)) {
                    if (src >= prom_rank_min && src <= prom_rank_max) {
                        int prom_flags[] = {PROMOTION_QUEEN, PROMOTION_ROOK, PROMOTION_BISHOP, PROMOTION_KNIGHT};
                        int prom_pieces[] = {(side==WHITE?wQ:bQ), (side==WHITE?wR:bR), (side==WHITE?wB:bB), (side==WHITE?wN:bN)};
                        for(int k=0; k<4; k++) add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, prom_pieces[k], prom_flags[k]));
                    } else {
                        add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, QUIET_MOVE));
                        // Double avance
                        if ((src >= start_rank_min && src <= start_rank_max) && !GET_BIT(occupancy, tgt + direction)) {
                            add_move(move_list, move_count, ENCODE_MOVE(src, tgt + direction, piece, 0, DOUBLE_PAWN_PUSH));
                        }
                    }
                }
                
                // Captures
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
                    } else if (en_passant_sq != 0 && tgt == en_passant_sq) {
                        add_move(move_list, move_count, ENCODE_MOVE(src, tgt, piece, 0, EP_CAPTURE));
                    }
                    POP_BIT(attacks_bb, tgt);
                }
                POP_BIT(bitboard, src);
            }
        } 
        
        // --- CAVALIERS & ROIS ---
        else if (piece == wN || piece == bN || piece == wK || piece == bK) {
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
                
                // ROQUE
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
        } 
        
        // --- GLISSEURS (Fous, Tours, Dames) ---
        else {
            while (bitboard) {
                src = __builtin_ctzll(bitboard);
                if (piece == wB || piece == bB) attacks = get_bishop_attacks(src, occupancy);
                else if (piece == wR || piece == bR) attacks = get_rook_attacks(src, occupancy);
                else attacks = get_bishop_attacks(src, occupancy) | get_rook_attacks(src, occupancy); // Dame
                
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

// ============================================================================
//                              GAME OVER DETECTION
// ============================================================================

int is_repetition() {
    int count = 1;
    // On remonte l'historique de 2 en 2 (mêmes joueurs)
    for (int i = game_ply - 2; i >= 0; i -= 2) {
        if (history[i].side == side && history[i].castle_rights == castle_rights && history[i].en_passant_sq == en_passant_sq) {
            if (memcmp(history[i].bitboards, bitboards, 12 * sizeof(Bitboard)) == 0) {
                count++;
            }
        }
        if (count >= 3) return 1; // 3 fois la même position
    }
    return 0;
}

int check_game_over() {
    // 1. Répétition
    if (is_repetition()) return 3;

    // 2. Plus de coups ?
    MOVE move_list[256];
    int move_count = 0;
    generate_moves(move_list, &move_count);

    GameState current_state;
    save_board(&current_state);

    int has_legal_move = 0;
    for (int i = 0; i < move_count; i++) {
        if (make_move(move_list[i], ALL_MOVES)) {
            has_legal_move = 1;
            restore_board(&current_state); // Important de restaurer !
            break; 
        }
    }

    if (!has_legal_move) {
        int king_sq = __builtin_ctzll((side == WHITE) ? bitboards[wK] : bitboards[bK]);
        // Si le roi est attaqué -> MAT, sinon -> PAT
        if (is_square_attacked(king_sq, side ^ 1)) {
            return 1; // MAT
        } else {
            return 2; // PAT
        }
    }
    
    return 0; // Jeu continue
}