import argparse
import time

import numpy as np

from AzulMCHeuristic import MCTSPlayer
from AzulMCScorePolicies import HeuristicScore
from AzulMCTestPlayers import play_game, mcts_vs_random, mcts_vs_mcts, interactive_game, RandomPlayer
from AzulSimluatedMoveActors import heuristic_mover, heuristic_mover_fast

parser = argparse.ArgumentParser(description="Azul MCTS")
args = parser.parse_args()
players = [
    MCTSPlayer(
        verbose=False,
        iterations=5000,
        terminus_evaluator=HeuristicScore.heuristic_score,
        opponent_sim_strategy=heuristic_mover_fast,
        filter_floor=True
        ),
    RandomPlayer(),
]

"""
    50 iterations, mcts(heuristic opp, heuristic terminus) v random, units seconds
    Original : 38.8
    Fast env copy: 31.5
"""

class Timer:
    def __init__(self):
        self.start_times = {}

    def start_timer(self, name="default"):
        self.start_times[name] = time.time()

    def current(self, name="default"):
        return time.time() - self.start_times[name]

def test_speed():
    timer = Timer()
    timer.start_timer()
    timer.start_timer("game1")
    print("testing games")
    res = play_game(players, seed=24, verbose=False)
    print("game 1", timer.current("game1"))
    print(res)
    timer.start_timer("game2")
    res = play_game(players, seed=25, verbose=False)
    print("game 2", timer.current("game2"))
    print(res)
    timer.start_timer("game3")
    res = play_game(players, seed=26, verbose=False)
    print("game 3", timer.current("game3"))
    print(res)
    print("Games took", timer.current(), "seconds!")

if __name__ == "__main__":
    interactive_game(mcts_iterations=1000, human_player=0)





    # parser.add_argument(
    #     "--mode",
    #     type=str,
    #     default="mcts_vs_random",
    #     choices=["mcts_vs_random", "mcts_vs_mcts", "interactive", "single"],
    #     help="Game mode",
    # )
    # parser.add_argument(
    #     "--games", type=int, default=20, help="Number of games for tournament modes"
    # )
    # parser.add_argument(
    #     "--iterations", type=int, default=500, help="MCTS iterations per move"
    # )
    # parser.add_argument(
    #     "--iterations2",
    #     type=int,
    #     default=1000,
    #     help="MCTS iterations for second player (mcts_vs_mcts mode)",
    # )
    # parser.add_argument(
    #     "--exploration", type=float, default=1.41, help="UCB1 exploration weight"
    # )
    # parser.add_argument("--seed", type=int, default=None, help="Random seed")
    # parser.add_argument("--verbose", action="store_true", help="Verbose output")
    #
    # if args.seed is not None:
    #     np.random.seed(args.seed)
    #
    # if args.mode == "mcts_vs_random":
    #     mcts_vs_random(num_games=args.games, mcts_iterations=args.iterations)
    #
    # elif args.mode == "mcts_vs_mcts":
    #     mcts_vs_mcts(
    #         num_games=args.games,
    #         iterations1=args.iterations,
    #         iterations2=args.iterations2,
    #     )
    #
    # elif args.mode == "interactive":
    #     interactive_game(mcts_iterations=args.iterations)
    #
    # elif args.mode == "single":
    #     # Single game with verbose output
    #     play_game(players, verbose=True, seed=args.seed)


