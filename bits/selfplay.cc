#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include <served/served.hpp>

#include "edgetpu.h"
#include "mcts.h"
#include "monitoring.h"

void setup_server(void);

static char file_name_buffer[128];
static char game_date_buffer[128];

void format_root_node(std::ostream &game_text, std::shared_ptr<Node> root,
                      Board &b) {
    game_text << "{ q=" << (1.0 - root->value()) << "; p=[";

    bool emit_comma = false;
    for (auto n : root->children) {
        if (n.second->visit_count > 0) {
            if (emit_comma)
                game_text << ", ";
            game_text << b.san(n.first) << ":" << n.second->visit_count;
            emit_comma = true;
        }
    }

    game_text << "] } ";
}

void header(std::ostream &pgn_file, int round, const std::string &outcome) {

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
    static std::uniform_int_distribution<int> d(0, 99);

    return d(gen) < 9;
}

bool fully_playout_move() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<int> d(0, 99);

    return d(gen) < 25;
}

static std::string current_epd;
static std::mutex epd_mutex;

void update_epd(Board &board) {
    const std::lock_guard<std::mutex> lock(epd_mutex);
    current_epd = board.epd();
}

void selfplay(std::string model_name, const int sims) {

    std::thread server_thread(setup_server);

    std::shared_ptr<EdgeTpuModel> model =
        std::make_shared<EdgeTpuModel>(model_name);

    MCTS mcts(model);
    mcts.set_kldgain_stop(1e-5);

    std::time_t t = std::time(nullptr);
    std::strftime(file_name_buffer, sizeof(file_name_buffer),
                  "LearnGames-%Y-%m-%d-%H-%M-%S.pgn", std::localtime(&t));

    std::cout << file_name_buffer << std::endl;
    std::ofstream pgn_file;

    pgn_file.open(file_name_buffer);

    for (int round = 1;; round++) {
        Board b;

        std::stringstream game_text;
        game_text << std::fixed << std::setprecision(4);

        const bool is_full_playout = fully_playout_game();

        while (!b.game_ended()) {

            update_epd(b);

            const bool is_move_fully_playedout =
                is_full_playout || fully_playout_move();

            if (is_move_fully_playedout) {
                b.print();
            }

            mcts.use_exploration_noise(is_move_fully_playedout);
            mcts.use_forced_playouts(is_move_fully_playedout);

            const std::shared_ptr<Node> root =
                mcts.mcts(b, is_move_fully_playedout ? sims : 100);

            if (is_move_fully_playedout) {
                mcts.correct_forced_playouts(root);
            }

            const uint32_t move =
                (is_move_fully_playedout && (b.move_number() <= 15))
                    ? select_randomized_move(root)
                    : select_most_visited_move(root);

            game_text << b.move_number_if_white() << b.san(move) << " ";

            if (is_move_fully_playedout) {
                format_root_node(game_text, root, b);
                monitoring::monitoring::instance()->observe_position();
                monitoring::monitoring::instance()->observe_evaluation(
                    1.0 - root->value());
            }

            b.do_move(move);
        }

        const auto outcome = b.outcome();
        game_text << outcome;

        header(pgn_file, round, outcome);
        pgn_file << std::endl;
        pgn_file << game_text.str() << std::endl;
        pgn_file << std::endl;
        pgn_file.flush();

        monitoring::monitoring::instance()->observe_game();
    }
}

void setup_server(void) {
    // Create a multiplexer for handling requests
    served::multiplexer mux;

    // GET /hello
    mux.handle("/epd").get(
        [](served::response &res, const served::request &req) {
            const std::lock_guard<std::mutex> lock(epd_mutex);
            res << current_epd;
        });

    // Create the server and run with 10 handler threads.
    served::net::server server("127.0.0.1", "8080", mux);
    server.run(1);
}
