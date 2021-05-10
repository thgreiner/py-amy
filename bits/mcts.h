#ifndef MCTS_H
#define MCTS_H

#include <map>
#include <memory>
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

  private:
    bool is_root;
    int forced_playouts;
};

class MCTS {
  public:
    MCTS(std::shared_ptr<EdgeTpuModel> m) : model(m) {
        heap = allocate_heap();
    };
    void mcts(Board &);

  private:
    std::shared_ptr<EdgeTpuModel> model;
    float evaluate(std::shared_ptr<Node> node, Board &board);
    uint32_t select_child(std::shared_ptr<Node>);
    void backpropagate(std::vector<std::shared_ptr<Node>>, float, bool);
    void add_exploration_noise(std::shared_ptr<Node>);
    heap_t heap;
};

#endif
