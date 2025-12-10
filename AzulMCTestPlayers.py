import argparse
from typing import Optional

import numpy as np

from AzulEnvHeuristic import Azul, Phase
from AzulMCHeuristic import MCTSPlayer
from AzulMCScorePolicies import HeuristicScore
from AzulSimluatedMoveActors import heuristic_mover, heuristic_mover_fast

ANSI_GREEN = '\033[92m'   # Bright green for input
ANSI_CYAN = '\033[96m'    # Cyan for prompts
ANSI_WHITE = '\033[35m'
ANSI_RESET = '\033[0m'
class RandomPlayer:
    """Simple random player for comparison."""

    def __init__(self):
        pass

    @staticmethod
    def select_action(game: Azul) -> tuple:
        """Select a random legal action."""
        moves = game.get_legal_moves()
        if not moves:
            return ()
        return moves[np.random.randint(len(moves))]


class HeuristicPlayer:
    """Heuristic player for a smarter model."""

    def __init__(self):
        pass

    @staticmethod
    def select_action(game: Azul) -> tuple:
        moves = game.get_legal_moves()
        if not moves:
            return ()
        return heuristic_mover(game)


def play_game(
        game_players: list,
        num_players: int = 2,
        verbose: bool = True,
        seed: Optional[int] = None,
) -> dict:
    """
    Play a game between the given players.

    Args:
            game_players: List of player objects (MCTSPlayer, RandomPlayer, etc.)
            num_players: Number of players
            verbose: Print game progress
            seed: Random seed for reproducibility

    Returns:
            Dictionary with game results
    """
    if seed is not None:
        np.random.seed(seed)

    if len(game_players) != num_players:
        raise ValueError(f"Expected {num_players} players, got {len(game_players)}")

    game = Azul(num_players=num_players)
    game.reset()

    if verbose:
        print(f"Starting game with {num_players} players")
        print(f"Players: {[type(p).__name__ for p in game_players]}")
        game.print_game()
        print("=" * 50)
    turn = 0
    round_num = 1

    while not game.game_over:
        current = game.current_player
        player = game_players[current]

        # Handle wall-tiling phase
        if game.phase == Phase.WALL_TILING:
            if verbose:
                print(f"\n--- End of Round {round_num} ---")
                print(f"Scores: {game.scores}")
                game.print_game()
            game.step(())
            round_num += 1
            continue

        # Check for legal moves
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            # No moves available - this shouldn't happen in a well-formed game
            if verbose:
                print("Warning: No legal moves available, skipping turn")
            continue

        # Get action from current player
        action = player.select_action(game)

        if verbose and action != ():
            source = (
                f"Factory {action[0]}" if action[0] < game.num_factories else "Center"
            )
            color = ["Blue", "Yellow", "Red", "Black", "White"][action[1]]
            dest = f"Line {action[2] + 1}" if action[2] < 5 else "Floor"
            print(
                f"Turn {turn + 1}: Player {current} ({type(player).__name__}) "
                f"takes {color} from {source} -> {dest}"
            )

        game.step(action)
        if verbose:
            game.print_game(None, current)
        turn += 1

    if verbose:
        print("\n" + "=" * 50)
        print("GAME OVER!")
        print(f"Final scores: {game.scores}")
        if game.winner is not None:
            print(
                f"Winner: Player {game.winner} ({type(game_players[game.winner]).__name__})"
            )
        else:
            print("Result: Tie!")
        print("=" * 50)

    return {
        "winner": game.winner,
        "scores": game.scores.copy(),
        "turns": turn,
        "rounds": round_num,
    }


def run_tournament(
        player_configs: list[dict],
        num_games: int = 100,
        num_players: int = 2,
        verbose: bool = True,
) -> dict:
    """
    Run a tournament between different player configurations.

    Args:
            player_configs: List of dicts with 'type' and optional params
                                       e.g., [{'type': 'mcts', 'iterations': 1000},
                                                      {'type': 'random'}]
            num_games: Number of games to play
            num_players: Number of players per game
            verbose: Print progress

    Returns:
            Dictionary with tournament statistics
    """

    def create_player(config: dict):
        if config["type"] == "random":
            return RandomPlayer()
        elif config["type"] == "mcts":
            return MCTSPlayer(
                iterations=config.get("iterations", 1000),
                exploration_weight=config.get("exploration_weight", 1.41),
                time_limit=config.get("time_limit"),
                verbose=config.get("verbose", False),
            )
        else:
            raise ValueError(f"Unknown player type: {config['type']}")

    wins = np.zeros(num_players)
    ties = 0
    total_scores = np.zeros(num_players)

    for game_num in range(num_games):
        # Create fresh players for each game
        game_players = [
            create_player(player_configs[i % len(player_configs)])
            for i in range(num_players)
        ]

        result = play_game(game_players, num_players=num_players, verbose=False)

        if result["winner"] is not None:
            wins[result["winner"]] += 1
        else:
            ties += 1

        total_scores += result["scores"]

        if verbose and (game_num + 1) % 10 == 0:
            print(
                f"Completed {game_num + 1}/{num_games} games... "
                f"Wins: {wins.astype(int)}, Ties: {ties}"
            )

    stats = {
        "num_games": num_games,
        "wins": wins.astype(int),
        "ties": ties,
        "win_rates": wins / num_games,
        "avg_scores": total_scores / num_games,
        "player_configs": player_configs,
    }

    return stats


def mcts_vs_random(
        num_games: int = 20, mcts_iterations: int = 500, verbose: bool = True
) -> dict:
    """
    Run MCTS vs Random player comparison.

    Args:
            num_games: Number of games to play
            mcts_iterations: MCTS iterations per move
            verbose: Print progress

    Returns:
            Tournament statistics
    """
    configs = [{"type": "mcts", "iterations": mcts_iterations}, {"type": "random"}]

    if verbose:
        print(f"MCTS ({mcts_iterations} iterations) vs Random")
        print("=" * 50)

    stats = run_tournament(configs, num_games=num_games, verbose=verbose)

    if verbose:
        print("\n" + "=" * 50)
        print("RESULTS:")
        print(f"  MCTS wins: {stats['wins'][0]} ({stats['win_rates'][0] * 100:.1f}%)")
        print(f"  Random wins: {stats['wins'][1]} ({stats['win_rates'][1] * 100:.1f}%)")
        print(f"  Ties: {stats['ties']} ({stats['ties'] / num_games * 100:.1f}%)")
        print(
            f"  Avg scores: MCTS={stats['avg_scores'][0]:.1f}, Random={stats['avg_scores'][1]:.1f}"
        )

    return stats


def mcts_vs_mcts(
        num_games: int = 20,
        iterations1: int = 500,
        iterations2: int = 1000,
        exploration1: float = 1.41,
        exploration2: float = 1.41,
        verbose: bool = True,
) -> dict:
    """
    Run MCTS vs MCTS with different parameters.

    Args:
            num_games: Number of games to play
            iterations1: MCTS iterations for player 0
            iterations2: MCTS iterations for player 1
            exploration1: Exploration weight for player 0
            exploration2: Exploration weight for player 1
            verbose: Print progress

    Returns:
            Tournament statistics
    """
    configs = [
        {"type": "mcts", "iterations": iterations1, "exploration_weight": exploration1},
        {"type": "mcts", "iterations": iterations2, "exploration_weight": exploration2},
    ]

    if verbose:
        print(
            f"MCTS({iterations1}, c={exploration1}) vs MCTS({iterations2}, c={exploration2})"
        )
        print("=" * 50)

    stats = run_tournament(configs, num_games=num_games, verbose=verbose)

    if verbose:
        print("\n" + "=" * 50)
        print("RESULTS:")
        print(f"  MCTS1 wins: {stats['wins'][0]} ({stats['win_rates'][0] * 100:.1f}%)")
        print(f"  MCTS2 wins: {stats['wins'][1]} ({stats['win_rates'][1] * 100:.1f}%)")
        print(f"  Ties: {stats['ties']} ({stats['ties'] / num_games * 100:.1f}%)")
        print(
            f"  Avg scores: MCTS1={stats['avg_scores'][0]:.1f}, MCTS2={stats['avg_scores'][1]:.1f}"
        )

    return stats


def interactive_game(mcts_iterations: int = 1000, human_player: int = 0):
    """
    Play an interactive game against MCTS.

    Args:
            mcts_iterations: MCTS iterations per move
            human_player: Which player is human (0 or 1)
    """
    game = Azul(num_players=2)
    game.reset()

    mcts = MCTSPlayer(
        iterations=mcts_iterations,
        terminus_evaluator=HeuristicScore.heuristic_score,
        opponent_sim_strategy=heuristic_mover_fast,
        verbose=False,
        filter_floor=True,
    )

    # Color name to index mapping
    color_map = {
        'blue': 0, 'b': 0,
        'yellow': 1, 'y': 1,
        'red': 2, 'r': 2,
        'black': 3, 'k': 3,
        'white': 4, 'w': 4,
    }
    color_names = ["Blue", "Yellow", "Red", "Black", "White"]

    print("=" * 50)
    print("AZUL - Human vs MCTS")
    print(f"You are Player {human_player}")
    print("=" * 50)
    print("\nInput format: factory color line")
    print("  factory: 0-4 for factories, 'c' for center")
    print("  color: blue/b, yellow/y, red/r, black/k, white/w")
    print("  line: 1-5 for pattern lines, 'f' for floor")
    print("Example: '0 blue 2' or '0 b 2' or 'c r f'")
    print("=" * 50)

    round_num = 1

    while not game.game_over:
        # Handle wall-tiling
        if game.phase == Phase.WALL_TILING:
            print(f"\n--- End of Round {round_num} ---")
            game.print_game()
            input("Press Enter to continue to wall-tiling...")
            game.step(())
            round_num += 1
            continue

        current = game.current_player
        game.print_game()

        if current == human_player:
            # Human turn
            moves = game.get_legal_moves()
            print("\nYour turn!")

            while True:
                try:
                    user_input = input(f"{ANSI_CYAN}Enter move (factory color line): {ANSI_WHITE}").strip().lower()
                    parts = user_input.split()

                    if len(parts) != 3:
                        print("Invalid format. Use: factory color line (e.g., '0 blue 2')")
                        continue

                    # Parse factory/center
                    if parts[0] == 'c' or parts[0] == 'center':
                        source = game.num_factories
                    else:
                        source = int(parts[0])
                        if not (0 <= source < game.num_factories):
                            print(f"Factory must be 0-{game.num_factories - 1} or 'c' for center")
                            continue

                    # Parse color
                    if parts[1] not in color_map:
                        print("Color must be: blue/b, yellow/y, red/r, black/k, white/w")
                        continue
                    color = color_map[parts[1]]

                    # Parse line/floor
                    if parts[2] == 'f' or parts[2] == 'floor':
                        dest = 5
                    else:
                        dest = int(parts[2]) - 1  # Convert 1-5 to 0-4
                        if not (0 <= dest < 5):
                            print("Line must be 1-5 or 'f' for floor")
                            continue

                    action = (source, color, dest)

                    if action in moves:
                        break
                    else:
                        print(f"Illegal move: {action}. Check the legal moves list above.")

                except ValueError:
                    print("Invalid input. Use: factory color line (e.g., '0 blue 2')")
        else:
            # MCTS turn
            print(f"\nMCTS is thinking...")
            action = mcts.select_action(game)
            source = (
                f"Factory {action[0]}" if action[0] < game.num_factories else "Center"
            )
            color = color_names[action[1]]
            dest = f"Line {action[2] + 1}" if action[2] < 5 else "Floor"
            print(f"MCTS plays: {source} -> {color} -> {dest}")

        game.step(action)

    print("\n" + "=" * 50)
    print("GAME OVER!")
    game.print_game()
    print(f"Final scores: {game.scores}")
    if game.winner is not None:
        if game.winner == human_player:
            print("Congratulations, you win!")
        else:
            print("MCTS wins!")
    else:
        print("It's a tie!")

def x_interactive_game(mcts_iterations: int = 1000, human_player: int = 0):
    """
    Play an interactive game against MCTS.

    Args:
            mcts_iterations: MCTS iterations per move
            human_player: Which player is human (0 or 1)
    """
    game = Azul(num_players=2)
    game.reset()

    mcts = MCTSPlayer(
        iterations=mcts_iterations,
        terminus_evaluator=HeuristicScore.heuristic_score,
        opponent_sim_strategy=heuristic_mover_fast,
        verbose=True,
        filter_floor=True,
    )

    print("=" * 50)
    print("AZUL - Human vs MCTS")
    print(f"You are Player {human_player}")
    print("=" * 50)

    round_num = 1

    while not game.game_over:
        # Handle wall-tiling
        if game.phase == Phase.WALL_TILING:
            print(f"\n--- End of Round {round_num} ---")
            game.print_game()
            input("Press Enter to continue to wall-tiling...")
            game.step(())
            round_num += 1
            continue

        current = game.current_player
        game.print_game()

        if current == human_player:
            # Human turn
            moves = game.get_legal_moves()
            print("\nYour turn! Legal moves:")
            for i, move in enumerate(moves):
                source = (
                    f"Factory {move[0]}" if move[0] < game.num_factories else "Center"
                )
                color = ["Blue", "Yellow", "Red", "Black", "White"][move[1]]
                dest = f"Line {move[2] + 1}" if move[2] < 5 else "Floor"
                print(f"  {i}: {source} -> {color} -> {dest}")

            while True:
                try:
                    choice = int(input("Enter move number: "))
                    if 0 <= choice < len(moves):
                        action = moves[choice]
                        break
                    else:
                        print("Invalid choice, try again.")
                except ValueError:
                    print("Please enter a number.")
        else:
            # MCTS turn
            print(f"\nMCTS is thinking...")
            action = mcts.select_action(game)
            source = (
                f"Factory {action[0]}" if action[0] < game.num_factories else "Center"
            )
            color = ["Blue", "Yellow", "Red", "Black", "White"][action[1]]
            dest = f"Line {action[2] + 1}" if action[2] < 5 else "Floor"
            print(f"MCTS plays: {source} -> {color} -> {dest}")

        game.step(action)

    print("\n" + "=" * 50)
    print("GAME OVER!")
    game.print_game()
    print(f"Final scores: {game.scores}")
    if game.winner is not None:
        if game.winner == human_player:
            print("Congratulations, you win!")
        else:
            print("MCTS wins!")
    else:
        print("It's a tie!")

