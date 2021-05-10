#include <iostream>
#include <random>
#include <sys/time.h>
#include <algorithm>

#include "mcts.h"
#include "movegen.h"

void MCTS::mcts(Board &board) {

    auto n = 800;

    struct timeval begin, end;
    gettimeofday(&begin, 0);

    std::shared_ptr<Node> root = std::make_shared<Node>(0);

    float value = evaluate(root, board);
    std::cout << "Value: " << 100 * value << "%." << std::endl;

    add_exploration_noise(root);

    std::vector<std::shared_ptr<Node>> search_path;

    for (int simulation = 0; simulation < n; simulation++) {

        // std::cout << simulation << ": ";

        std::shared_ptr<Node> node = root;

        search_path.clear();
        search_path.push_back(node);

        int depth = 0;

        while (node->is_expanded()) {
            uint32_t move = select_child(node);

            board.do_move(move);
            depth++;

            node = node->children[move];
            search_path.push_back(node);
        }

        float value = evaluate(node, board);
        backpropagate(search_path, value, board.turn());

        for (auto i = 0; i < depth; i++)
            board.undo_move();
    }

    gettimeofday(&end, 0);

    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    float elapsed = seconds + microseconds * 1e-6;

    std::cout << "Inference took " << elapsed << "secs." << std::endl;

    std::cout << " = " << (n / elapsed) << " 1/s." << std::endl;

    std::vector<uint32_t> moves;
    board.generate_legal_moves(moves);

    std::sort(moves.begin(), moves.end(), [root](uint32_t a, uint32_t b) { return root->children[a]->visit_count > root->children[b]->visit_count; });
    for (auto move : moves) {
        auto child = root->children[move];
        if (child->visit_count > 0) {
            std::cout << board.san(move) << ":\t" << child->visit_count << ",\t"
                      << 100.0 * child->value() << "%" << std::endl;
        }
    }

    print_pv(root, board);
    std::cout << std::endl;
}

float MCTS::evaluate(std::shared_ptr<Node> node, Board &board) {
    std::vector<uint32_t> moves;

    node->turn = board.turn();

    board.generate_legal_moves(moves);

    if (moves.size() == 0) {
        if (board.is_in_check()) {
            return 0.0f;
        } else {
            return 0.5f;
        }
    }

    // std::cout << "Generated " << moves.size() << " legal moves." <<
    // std::endl;
    // TODO

    model->predict(board.current_position());

    int eor = board.turn() ? 0 : 0x38;

    std::map<uint32_t, float> move_probs;
    float prob_sum = 0.0f;

    for (auto move : moves) {
        float value = model->get_logit(move, eor);
        float probability = exp(value);
        move_probs[move] = probability;
        prob_sum += probability;
    }

    for (const auto &[move, probability] : move_probs) {

        float prior = probability / prob_sum;

        node->children[move] = std::make_shared<Node>(prior);

        // std::cout << san(pos, move) << ": " << (100 * prior) << "%"
        //           << std::endl;
    }

    return model->get_value();
}

float ucb_score(std::shared_ptr<Node> parent, std::shared_ptr<Node> child) {
    static float pb_c_init = 1.25f;
    static float pb_c_base = 19652.0f;

    float pb_c =
        log((parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
    pb_c *= sqrt(parent->visit_count) / (child->visit_count + 1);

    return child->value() + child->prior * pb_c;
}

uint32_t MCTS::select_child(std::shared_ptr<Node> node) {
    uint32_t best_action = 0;
    float best_value = 0.0;

    for (const auto &[action, child] : node->children) {
        auto value = ucb_score(node, child);
        if (best_action == 0 || value > best_value) {
            best_action = action;
            best_value = value;
        }
    }
    // std::cout << best_value << " ";

    return best_action;
}

void MCTS::backpropagate(std::vector<std::shared_ptr<Node>> search_path,
                         float value, bool turn) {
    for (auto node : search_path) {
        node->value_sum += (node->turn != turn) ? value : 1.0 - value;
        node->visit_count += 1;
    }
}

void MCTS::add_exploration_noise(std::shared_ptr<Node> node) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::gamma_distribution<> d(0.3, 1);
    static float fraction = 0.25;

    std::vector<float> noise;

    float noise_sum = 0.0;

    for (auto i = 0; i < node->children.size(); i++) {
        auto g = d(gen);
        noise.push_back(g);
        noise_sum += g;
    }

    float policy_sum = 0.0;

    auto n = noise.begin();

    for (const auto &[move, child] : node->children) {
        child->prior =
            (1 - fraction) * child->prior + fraction * *(n++) / noise_sum;
        policy_sum += child->prior;
    }

    for (const auto &[move, child] : node->children) {
        child->prior /= policy_sum;
    }
}

void MCTS::print_pv(std::shared_ptr<Node> node, Board &board) {
    if (node->visit_count < 2) return;
    if (node->children.size() == 0) return;

    using pair = decltype(node->children)::value_type;

    auto result = std::max_element(node->children.begin(), node->children.end(),
        [](const pair &a, const pair &b) { return a.second->visit_count < b.second->visit_count; });

    uint32_t move = result->first;

    std::cout << board.move_number_if_white() << board.san(move) << " ";
    board.do_move(move);

    print_pv(result->second, board);

    board.undo_move();
}
