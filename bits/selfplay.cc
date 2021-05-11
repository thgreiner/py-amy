#include "edgetpu.h"
#include "mcts.h"

void selfplay(char *model_name) {
    std::shared_ptr<EdgeTpuModel> model =
        std::make_shared<EdgeTpuModel>(model_name);
    MCTS mcts(model);

    for (;;) {
        Board b;

        while (!b.game_ended()) {
            b.print();
            std::shared_ptr<Node> root = mcts.mcts(b);
            mcts.correct_forced_playouts(root);

            uint32_t move;
            if (b.move_number() <= 30) {
                move = select_randomized_move(root);
            } else {
                move = select_most_visited_move(root);
            }
            b.do_move(move);
        }
    }
}
