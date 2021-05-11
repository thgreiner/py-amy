#include "bits.h"
#include "board.h"
#include "edgetpu.h"
#include "heap.h"
#include "magic.h"
#include "mcts.h"
#include "movegen.h"
#include "perft.h"
#include "position.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>
#include <fstream>

void san_test(void) {
    heap_t heap = allocate_heap();

    for (int i = 0; i < sizeof(PERFT_POSITIONS) / sizeof(char *); i++) {
        const char *epd = PERFT_POSITIONS[i];
        struct position pos;

        bool success = parse_epd(&pos, epd);

        if (!success) {
            printf("Illegal EPD!\n");
            continue;
        }

        print_position(&pos);
        push_section(heap);
        generate_pseudolegal_moves(heap, &pos);

        int cnt = 0;

        for (int i = heap->current_section->start;
             i < heap->current_section->end; i++) {
            uint32_t move = heap->data[i];
            struct position t;

            if (move & (CASTLE_KINGSIDE | CASTLE_QUEENSIDE)) {
                if (is_king_in_check(&pos, pos.turn)) {
                    continue;
                }
                if (!is_castle_legal(&pos, move)) {
                    continue;
                }
            }
            do_move(&pos, &t, move);
            if (is_king_in_check(&t, !t.turn))
                continue;

            char buffer[16];
            san(buffer, heap, &pos, move);
            cnt += strlen(buffer) + 1;

            if (cnt > 70) {
                printf("\n");
                cnt = 0;
            }
            uint32_t move_from_san;
            bool result = parse_san(buffer, heap, &pos, &move_from_san);

            printf("%s%s ", buffer, result ? "" : "?");
        }
        printf("\n\n");
    }
    free_heap(heap);
}

int random_uniform(const int bound) {
    int divider = RAND_MAX / bound;
    int high = bound * divider;

    int r;
    while ((r = rand()) >= high)
        ;

    return r / divider;
}

void play_random_move(heap_t heap, position_t pos) {
    push_section(heap);
    generate_pseudolegal_moves(heap, pos);

    int cnt = heap->current_section->end - heap->current_section->start;

    while (cnt) {
        uint32_t index = heap->current_section->start + random_uniform(cnt);
        uint32_t move = heap->data[index];
        struct position t;

        if (move & (CASTLE_KINGSIDE | CASTLE_QUEENSIDE)) {
            if (is_king_in_check(pos, pos->turn)) {
                cnt--;
                heap->data[index] =
                    heap->data[heap->current_section->start + cnt];
                continue;
            }
            if (!is_castle_legal(pos, move)) {
                cnt--;
                heap->data[index] =
                    heap->data[heap->current_section->start + cnt];
                continue;
            }
        }
        do_move(pos, &t, move);
        if (is_king_in_check(&t, !t.turn)) {
            cnt--;
            heap->data[index] = heap->data[heap->current_section->start + cnt];
            continue;
        }

        static char buffer[16];
        san(buffer, heap, pos, move);

        if (pos->ply % 2 == 0) {
            printf("%d. ", (pos->ply / 2) + 1);
        }
        printf("%s ", buffer);

        uint32_t move_from_san;
        bool result = parse_san(buffer, heap, pos, &move_from_san);

        assert(result);
        assert(move == move_from_san);

        if (t.irrev_count >= 100) {
            printf("{50 moves rule} 1/2-1/2");
        } else if (is_insufficient_material(&t)) {
            printf("{Insufficient material} 1/2-1/2");
        } else if (is_repeated(&t, 3)) {
            printf("{Draw by repetition} 1/2-1/2");
        } else {
            play_random_move(heap, &t);
        }
        break;
    }

    if (cnt == 0) {
        if (is_king_in_check(pos, pos->turn)) {
            if (is_checkmate(heap, pos)) {
                if (pos->turn) {
                    printf("{Black mates} 0-1");
                } else {
                    printf("{White mates} 1-0");
                }
            }
        } else {
            printf("{Stalemate} 1/2-1/2");
        }
    }
    pop_section(heap);
}

void play_random_game() {
    heap_t heap = allocate_heap();

    struct position pos;

    bool success = parse_epd(&pos, INITIAL_POSITION_EPD);

    if (!success) {
        printf("Illegal EPD!\n");
        return;
    }

    play_random_move(heap, &pos);

    free_heap(heap);

    printf("\n\n");
}

int main(int argc, char *argv[]) {

    srand(time(NULL));

#ifdef USE_MAGIC
    init_rook_table();
    init_bishop_table();
#endif

    for (int i = 1; i < argc; i++) {
        if (0 == strcmp("--san", argv[i])) {
            san_test();
        } else if (0 == strcmp("--perft", argv[i])) {
            int depth = 6;
            if (++i < argc)
                depth = atoi(argv[i]);

            perft(depth);
        } else if (0 == strcmp("--game", argv[i])) {
            int count = 1000;
            if (++i < argc)
                count = atoi(argv[i]);

            for (int j = count; j > 0; j--)
                play_random_game();
        } else if (0 == strcmp("--edgetpu", argv[i])) {
            if (++i >= argc) {
                return 1;
            }

            std::shared_ptr<EdgeTpuModel> model =
                std::make_shared<EdgeTpuModel>(argv[i]);
            MCTS mcts(model);

            i++;

            if (i == argc) {
                Board board;
                mcts.mcts(board);
            } else {
                std::ifstream infile(argv[i]);
                std::string line;

                while (std::getline(infile, line)) {
                    Board board(line);

                    std::cout << line << std::endl;
                    board.print();
                    mcts.mcts(board);

                    std::cout << std::endl;
                }
            }
        } else if (0 == strcmp("--board", argv[i])) {
            Board b;

            std::vector<uint32_t> moves;
            b.generate_legal_moves(moves);

            for (auto move : moves) {
                std::cout << b.san(move) << std::endl;

                b.do_move(move);

                std::vector<uint32_t> moves2;
                b.generate_legal_moves(moves2);

                for (auto move2 : moves2) {
                    std::cout << "  " << b.san(move2) << std::endl;
                }
                b.undo_move();
            }
        }
    }

    return 0;
}
