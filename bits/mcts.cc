#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/time.h>

#include "mcts.h"
#include "movegen.h"

std::shared_ptr<Node> MCTS::mcts(Board &board, const int n) {

    struct timeval begin, end;
    gettimeofday(&begin, 0);

    std::shared_ptr<Node> root = std::make_shared<Node>(0);
    root->is_root = true;

    float value = evaluate(root, board);
    // std::cout << "Value: " << 100 * value << "%." << std::endl;

    add_exploration_noise(root);

    std::vector<std::shared_ptr<Node>> search_path;

    std::cout << std::fixed << std::setprecision(1);

    for (int simulation = 0; simulation < n; simulation++) {

        // std::cout << simulation << ": ";

        std::shared_ptr<Node> node = root;

        search_path.clear();
        search_path.push_back(node);

        int depth = 0;

        while (node->is_expanded()) {
            std::pair<uint32_t, float> selection = select_child(node);
            uint32_t move = selection.first;

            board.do_move(move);
            depth++;

            node = node->children[move];
            search_path.push_back(node);

            if (selection.second == FORCED_PLAYOUT) {
                node->forced_playouts++;
            }
        }

        float value = evaluate(node, board);
        backpropagate(search_path, value, board.turn());

        for (auto i = 0; i < depth; i++)
            board.undo_move();

        if (simulation > 0 && simulation % 800 == 0)
            print_search_status(root, board, simulation);
    }

    gettimeofday(&end, 0);

    print_search_status(root, board, n);

    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    float elapsed = seconds + microseconds * 1e-6;

    std::cout << "Inference took " << elapsed << "s, " << (n / elapsed)
              << " 1/s." << std::endl;

    /*
    std::vector<uint32_t> moves;
    board.generate_legal_moves(moves);

    std::sort(moves.begin(), moves.end(), [root](uint32_t a, uint32_t b) {
        return root->children[a]->visit_count > root->children[b]->visit_count;
    });

    int cnt = 0;
    for (auto move : moves) {
        auto child = root->children[move];
        if (child->visit_count > 0) {
            std::cout << board.san(move) << ":\t" << std::setw(5)
                      << child->visit_count << "\t" << 100.0 * child->value()
                      << "%" << std::endl;
        }
        if (++cnt >= 5)
            break;
    }
    */

    return root;
}

void MCTS::print_search_status(std::shared_ptr<Node> root, Board &board,
                               int simulation) {
    auto best_move = select_most_visited_move(root);
    std::cout << std::setw(5) << simulation << ":  ";
    std::cout << std::setw(5) << (root->children[best_move]->value() * 100)
              << "%  ";
    print_pv(root, board);
    std::cout << std::endl;
}

float MCTS::evaluate(std::shared_ptr<Node> node, Board &board) {
    std::vector<uint32_t> moves;

    node->turn = board.turn();

    if (board.is_repeated(3) || board.is_insufficient_material() ||
        board.is_fifty_move_rule()) {
        return 0.5f;
    }

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

float ucb_score(std::shared_ptr<Node> parent, std::shared_ptr<Node> child,
                bool force_playouts = false) {
    static float pb_c_init = 1.25f;
    static float pb_c_base = 19652.0f;

    if (force_playouts) {
        float n_forced_playouts =
            sqrtf(child->prior * parent->visit_count * 2.0);
        if (child->visit_count < n_forced_playouts) {
            return MCTS::FORCED_PLAYOUT;
        }
    }

    float pb_c =
        logf((parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
    pb_c *= sqrtf(parent->visit_count) / (child->visit_count + 1);

    return child->value() + child->prior * pb_c;
}

std::pair<uint32_t, float> MCTS::select_child(std::shared_ptr<Node> node) {
    uint32_t best_action = 0;
    float best_value = 0.0;

    for (const auto &[action, child] : node->children) {
        auto value = ucb_score(node, child, node->is_root);
        if (best_action == 0 || value > best_value) {
            best_action = action;
            best_value = value;
        }
    }

    return std::pair<uint32_t, float>(best_action, best_value);
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

void recurse_pv(std::shared_ptr<Node> node, Board &board,
                std::vector<uint32_t> &line) {
    if (node->visit_count < 2)
        return;
    if (node->children.size() == 0)
        return;

    uint32_t move = select_most_visited_move(node);
    line.push_back(move);

    board.do_move(move);
    recurse_pv(node->children[move], board, line);
    board.undo_move();
}

void MCTS::print_pv(std::shared_ptr<Node> node, Board &board) {
    std::vector<uint32_t> line;
    recurse_pv(node, board, line);

    std::cout << board.variation_san(line);
}

uint32_t select_most_visited_move(std::shared_ptr<Node> node) {
    using pair = decltype(node->children)::value_type;

    auto result = std::max_element(node->children.begin(), node->children.end(),
                                   [](const pair &a, const pair &b) {
                                       return a.second->visit_count <
                                              b.second->visit_count;
                                   });

    return result->first;
}

uint32_t select_randomized_move(std::shared_ptr<Node> node) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> d(0, 1);

    std::map<uint32_t, float> move_to_prob;

    for (auto n : node->children) {
        move_to_prob[n.first] =
            logf(powf(n.second->visit_count, 2.0)) - logf(-logf(d(gen)));
    }

    using pair = decltype(move_to_prob)::value_type;
    auto result = std::max_element(
        move_to_prob.begin(), move_to_prob.end(),
        [](const pair &a, const pair &b) { return a.second < b.second; });

    return result->first;
}

void MCTS::correct_forced_playouts(std::shared_ptr<Node> node) {
    uint32_t best_move = select_most_visited_move(node);
    auto best_child = node->children[best_move];

    float best_ucb_score = ucb_score(node, best_child);

    for (auto n : node->children) {
        if (n.first == best_move)
            continue;
        auto child = n.second;

        int playouts = child->visit_count;
        for (int i = 1; i <= child->forced_playouts; i++) {
            child->visit_count = playouts - i;
            if (ucb_score(node, child) > best_ucb_score) {
                child->visit_count = playouts - i + 1;
                break;
            }
        }
    }
}
