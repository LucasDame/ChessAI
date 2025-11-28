#include "defs.h"

#define PORT 12345
#define BUFFER_SIZE 1024

int main() {
    // 1. Initialisations (Fonctions définies dans move.c et board.c)
    init_leapers_attacks();
    init_castling_masks();
    init_board();

    // 2. Configuration du Serveur
    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    socklen_t addrlen = sizeof(address);
    char buffer[BUFFER_SIZE] = {0};

    // Création du socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    // Force l'attachement au port 12345 même si utilisé récemment
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    if (listen(server_fd, 3) < 0) {
        perror("Listen failed");
        exit(EXIT_FAILURE);
    }

    printf("[MOTEUR] En attente sur le port %d...\n", PORT);

    if ((client_fd = accept(server_fd, (struct sockaddr *)&address, &addrlen)) < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }
    printf("[MOTEUR] Client connecte !\n");

    // 3. Boucle Principale
    while (1) {
        memset(buffer, 0, BUFFER_SIZE);
        int valread = recv(client_fd, buffer, BUFFER_SIZE - 1, 0);
        
        if (valread <= 0) {
            printf("[MOTEUR] Client deconnecte.\n");
            break;
        }
        
        // Nettoyage agressif du buffer (retrait \r, \n, espaces)
        buffer[valread] = '\0';
        while (valread > 0 && (buffer[valread-1] == '\n' || buffer[valread-1] == '\r' || buffer[valread-1] == ' ')) {
            buffer[valread-1] = '\0';
            valread--;
        }
        printf("[CMD RECUE] : '%s'\n", buffer);

        // --- QUIT ---
        if (strcmp(buffer, "quit") == 0) {
            break;
        }
        
        // --- UNDO ---
        else if (strcmp(buffer, "undo") == 0) {
            if (game_ply > 0) {
                game_ply--;
                restore_board(&history[game_ply]);
                printf("[MOTEUR] Undo -> Retour Ply %d\n", game_ply);
            }
            // Toujours renvoyer le plateau pour mettre à jour l'UI
            char board_str[65];
            serialize_board(board_str);
            char response[100];
            sprintf(response, "board:%s", board_str);
            send(client_fd, response, strlen(response), 0);
            continue;
        }

        // --- MODE IA (Commande "go") ---
        else if (strncmp(buffer, "go", 2) == 0) {
            printf("[MOTEUR] L'IA reflechit...\n");
            
            // 1. Sauvegarde Historique AVANT de jouer
            save_board(&history[game_ply]);

            // 2. Recherche du meilleur coup (depth 4)
            MOVE best = search_best_move(4);
            
            if (best != 0) {
                // Conversion du coup en texte
                int f = GET_MOVE_FROM(best);
                int t = GET_MOVE_TO(best);
                int p = GET_MOVE_PROMOTED(best);
                char move_str[6];
                sprintf(move_str, "%c%d%c%d", 'a'+(f%8), 1+(f/8), 'a'+(t%8), 1+(t/8));
                
                // Promotion
                if (p) {
                    char promo_char = 'q';
                    if (p==wR||p==bR) promo_char='r'; 
                    else if (p==wB||p==bB) promo_char='b'; 
                    else if (p==wN||p==bN) promo_char='n';
                    size_t len = strlen(move_str);
                    move_str[len] = promo_char;
                    move_str[len+1] = '\0';
                }
                
                // 3. Jouer le coup
                make_move(best, ALL_MOVES);
                game_ply++; // Avancer l'historique

                // 4. Préparer la réponse
                char board_str[65];
                serialize_board(board_str);
                
                int status = check_game_over();
                char response[256];
                
                if (status == 0) sprintf(response, "bestmove:%s board:%s", move_str, board_str);
                else if (status == 1) sprintf(response, "bestmove:%s board:%s game_over:checkmate", move_str, board_str);
                else if (status == 2) sprintf(response, "bestmove:%s board:%s game_over:stalemate", move_str, board_str);
                else if (status == 3) sprintf(response, "bestmove:%s board:%s game_over:draw_repetition", move_str, board_str);
                
                send(client_fd, response, strlen(response), 0);
            } else {
                // Aucun coup trouvé (Mat ou Pat immédiat)
                send(client_fd, "bestmove:none", 13, 0);
            }
            continue;
        }

        // --- MODE HUMAIN (Coordonnées) ---
        int from, to;
        char prom_char;
        
        // Parsing des coordonnées (ex: "e2e4")
        if (parse_input_squares(buffer, &from, &to, &prom_char)) {
            
            // Génération des coups pour vérifier la légalité
            MOVE move_list[256];
            int move_count = 0;
            generate_moves(move_list, &move_count);
            
            int move_found = 0; 
            MOVE chosen_move = 0;
            
            // Recherche du coup dans la liste
            for (int i = 0; i < move_count; i++) {
                MOVE m = move_list[i];
                if (GET_MOVE_FROM(m) == from && GET_MOVE_TO(m) == to) {
                    int promoted = GET_MOVE_PROMOTED(m);
                    if (promoted) {
                        if (prom_char != 0) {
                            int is_match = 0;
                            if ((promoted == wQ || promoted == bQ) && prom_char == 'q') is_match = 1;
                            else if ((promoted == wR || promoted == bR) && prom_char == 'r') is_match = 1;
                            else if ((promoted == wB || promoted == bB) && prom_char == 'b') is_match = 1;
                            else if ((promoted == wN || promoted == bN) && prom_char == 'n') is_match = 1;
                            if (is_match) { chosen_move = m; move_found = 1; break; }
                        } else {
                            if (promoted == wQ || promoted == bQ) { chosen_move = m; move_found = 1; break; }
                        }
                    } else {
                        chosen_move = m; move_found = 1; break;
                    }
                }
            }
            
            char response[256];
            if (move_found) {
                // 1. Sauvegarde Historique
                save_board(&history[game_ply]);

                // 2. Jouer le coup
                if (make_move(chosen_move, ALL_MOVES)) {
                    game_ply++; // Avancer historique

                    char board_str[65];
                    serialize_board(board_str);
                    
                    int status = check_game_over();
                    if (status == 0) sprintf(response, "board:%s", board_str);
                    else if (status == 1) sprintf(response, "board:%s game_over:checkmate", board_str);
                    else if (status == 2) sprintf(response, "board:%s game_over:stalemate", board_str);
                    else if (status == 3) sprintf(response, "board:%s game_over:draw_repetition", board_str);
                    
                } else {
                    sprintf(response, "illegal_move_king_check");
                }
            } else {
                sprintf(response, "illegal_move_rules");
            }
            
            send(client_fd, response, strlen(response), 0);
        } else {
            char *msg = "unknown_command";
            send(client_fd, msg, strlen(msg), 0);
        }
    }

    close(client_fd);
    close(server_fd);
    return 0;
}