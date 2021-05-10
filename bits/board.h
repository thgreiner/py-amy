#ifndef BOARD_H
#define BOARD_H

#include <memory>
#include <string>
#include <vector>

#include "heap.h"
#include "position.h"

class Board {
  public:
    Board();
    Board(std::string &);
    void generate_legal_moves(std::vector<uint32_t> &moves);
    void do_move(uint32_t move);
    void undo_move();
    std::string san(uint32_t);
    ~Board();
    position_t current_position();
    bool turn();

  private:
    std::vector<std::shared_ptr<struct position>> positions;
    heap_t heap;
};

#endif
