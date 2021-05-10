#include "board.h"

#include "movegen.h"

Board::Board() {
    std::shared_ptr<struct position> p = std::make_shared<struct position>();
    parse_epd(p.get(), INITIAL_POSITION_EPD);
    positions.push_back(p);

    heap = allocate_heap();
}

Board::Board(std::string &epd) {
    std::shared_ptr<struct position> p = std::make_shared<struct position>();
    parse_epd(p.get(), epd.c_str());
    positions.push_back(p);

    heap = allocate_heap();
}

Board::~Board() { free_heap(heap); }

void Board::generate_legal_moves(std::vector<uint32_t> &moves) {
    position_t pos = positions.back().get();

    push_section(heap);
    generate_pseudolegal_moves(heap, pos);

    for (int i = heap->current_section->start; i < heap->current_section->end;
         i++) {

        struct position t;
        uint32_t move = heap->data[i];

        if (move & (CASTLE_KINGSIDE | CASTLE_QUEENSIDE)) {
            if (is_king_in_check(pos, pos->turn)) {
                continue;
            }
            if (!is_castle_legal(pos, move)) {
                continue;
            }
        }
        ::do_move(pos, &t, move);
        if (is_king_in_check(&t, !t.turn))
            continue;

        moves.push_back(move);
    }

    pop_section(heap);
}

void Board::do_move(uint32_t move) {
    position_t pos = positions.back().get();
    std::shared_ptr<struct position> next = std::make_shared<struct position>();

    ::do_move(pos, next.get(), move);

    positions.push_back(next);
}

void Board::undo_move() { positions.pop_back(); }

std::string Board::san(uint32_t move) {
    position_t pos = positions.back().get();

    static char buffer[16];
    ::san(buffer, heap, pos, move);

    return std::string(buffer);
}

position_t Board::current_position() { return positions.back().get(); }

bool Board::turn() { return current_position()->turn; }
