#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/registry.h>

#include "edgetpu.h"
#include "mcts.h"

static char file_name_buffer[128];
static char game_date_buffer[128];

void format_root_node(std::ostream &game_text, std::shared_ptr<Node> root,
                      Board &b) {
    game_text << "{ q=" << (1.0 - root->value()) << "; p=[";

    bool emit_comma = false;
    for (auto n : root->children) {
        if (emit_comma)
            game_text << ", ";
        game_text << b.san(n.first) << ":" << n.second->visit_count;
        emit_comma = true;
    }

    game_text << "] } ";
}

void header(std::ostream &pgn_file, int round, std::string &outcome) {

    std::time_t t = std::time(nullptr);
    std::strftime(game_date_buffer, sizeof(game_date_buffer), "%Y.%m.%d",
                  std::localtime(&t));

    pgn_file << "[Event \"Test Game\"]" << std::endl;
    pgn_file << "[Site \"?\"]" << std::endl;
    pgn_file << "[Date \"" << game_date_buffer << "\"]" << std::endl;
    pgn_file << "[Round \"" << round << "\"]" << std::endl;
    pgn_file << "[White \"Amy Zero\"]" << std::endl;
    pgn_file << "[Black \"Amy Zero\"]" << std::endl;
    pgn_file << "[Result \"" << outcome << "\"]" << std::endl;
}

bool fully_playout_game() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> d(0, 12);

    return d(gen) == 0;
}

bool fully_playout_move() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> d(0, 4);

    return d(gen) == 0;
}

void selfplay(std::string model_name) {

    std::shared_ptr<EdgeTpuModel> model =
        std::make_shared<EdgeTpuModel>(model_name);

    MCTS mcts(model);
    mcts.use_exploration_noise(true);

    std::time_t t = std::time(nullptr);
    std::strftime(file_name_buffer, sizeof(file_name_buffer),
                  "LearnGames-%Y-%m-%d-%H-%M-%S.pgn", std::localtime(&t));

    std::cout << file_name_buffer << std::endl;
    std::ofstream pgn_file;

    pgn_file.open(file_name_buffer);

    // Monitoring
    prometheus::Exposer exposer{"0.0.0.0:9100"};
    auto registry = std::make_shared<prometheus::Registry>();

    auto &nodes_counter = prometheus::BuildCounter()
                              .Name("nodes_total")
                              .Help("Nodes visited")
                              .Register(*registry)
                              .Add({{"instance", "selfplay "}});

    auto &games_counter = prometheus::BuildCounter()
                              .Name("games_total")
                              .Help("Games played by selfplay")
                              .Register(*registry)
                              .Add({});

    auto &positions_counter = prometheus::BuildCounter()
                                  .Name("positions_total")
                                  .Help("Positions generated by selfplay")
                                  .Register(*registry)
                                  .Add({});

    auto &evaluation_gauge = prometheus::BuildGauge()
                                 .Name("white_prob")
                                 .Help("Win probability white")
                                 .Register(*registry)
                                 .Add({});

    // ask the exposer to scrape the registry on incoming HTTP requests
    exposer.RegisterCollectable(registry);

    for (int round = 1;; round++) {
        Board b;

        std::stringstream game_text;
        game_text << std::fixed << std::setprecision(3);

        bool is_full_playout = fully_playout_game();

        while (!b.game_ended()) {
            b.print();

            bool is_move_fully_playedout =
                is_full_playout || fully_playout_move();

            int visits = is_move_fully_playedout ? 800 : 100;

            std::shared_ptr<Node> root = mcts.mcts(b, visits);

            nodes_counter.Increment(root->visit_count);

            mcts.correct_forced_playouts(root);

            uint32_t move;
            if (b.move_number() <= 30) {
                move = select_randomized_move(root);
            } else {
                move = select_most_visited_move(root);
            }

            game_text << b.move_number_if_white() << b.san(move) << " ";

            if (is_move_fully_playedout) {
                format_root_node(game_text, root, b);
                positions_counter.Increment();

                evaluation_gauge.Set(1.0 - root->value());
            }

            b.do_move(move);
        }
        auto outcome = b.outcome();

        game_text << outcome;

        header(pgn_file, round, outcome);
        pgn_file << std::endl;
        pgn_file << game_text.str() << std::endl;
        pgn_file << std::endl;
        pgn_file.flush();

        games_counter.Increment();
    }
}
