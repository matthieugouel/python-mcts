"""Main execution."""

import random

from players import MCTS, Random
from tictactoe import TicTacToe


EPISODES = 100


if __name__ == "__main__":
    game = TicTacToe()
    players = [MCTS(game, "X"), Random(game, "O")]

    for _ in range(EPISODES):

        observation = game.reset()
        random.shuffle(players)

        while True:
            for current_player in players:
                action = current_player.act(observation)
                observation, _, done, player = game.step(current_player.token, action)

                if done:
                    for p in players:
                        if player == p.token:
                            p.score["wins"] += 1
                        elif player is None:
                            p.score["ties"] += 1
                        else:
                            p.score["losses"] += 1
                    break
            else:
                continue
            break

    # Print MCTS player score
    print(players)
