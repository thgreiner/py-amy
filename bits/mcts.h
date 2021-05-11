#ifndef MCTS_H
#define MCTS_H

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "board.h"
#include "edgetpu.h"
#include "heap.h"
#include "position.h"

class Node {
  public:
    Node(float p) : prior(p){};

    float value() const {
        return (visit_count == 0) ? 0.0f : (value_sum / visit_count);
    }

    int visit_count = 0;
    bool turn;
    float prior;
    std::map<uint32_t, std::shared_ptr<Node>> children;

    bool is_expanded() const { return children.size() != 0; }

    float value_sum = 0.0;
    bool is_root = false;
    int forced_playouts = 0;
};

class MCTS {
  public:
    MCTS(std::shared_ptr<EdgeTpuModel> m) : model(m) {
        heap = allocate_heap();
    };
    std::shared_ptr<Node> mcts(Board &);
    void correct_forced_playouts(std::shared_ptr<Node>);

    static constexpr float FORCED_PLAYOUT = 1e5;

  private:
    std::shared_ptr<EdgeTpuModel> model;
    float evaluate(std::shared_ptr<Node> node, Board &board);
    std::pair<uint32_t, float> select_child(std::shared_ptr<Node>);
    void backpropagate(std::vector<std::shared_ptr<Node>>, float, bool);
    void add_exploration_noise(std::shared_ptr<Node>);
    void print_pv(std::shared_ptr<Node>, Board &board);
    heap_t heap;
};

uint32_t select_most_visited_move(std::shared_ptr<Node>);
uint32_t select_randomized_move(std::shared_ptr<Node>);

#endif