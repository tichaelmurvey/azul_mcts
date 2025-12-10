"""
Monte Carlo Tree Search implementation for Azul.

Provides MCTS players that can play against random opponents or other MCTS players
with different parameters.
"""
import numpy as np
import math
import time
from typing import Optional
from AzulEnvHeuristic import (
    Azul,
)
from AzulMCScorePolicies import HeuristicScore
from AzulSimluatedMoveActors import random_mover, filter_bad_moves

type Wall = np.ndarray[tuple[int, int], np.dtype[np.bool_]]


class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(
            self,
            game_state: Azul,
            parent: Optional["MCTSNode"] = None,
            action: Optional[tuple] = None,
            filter_floor: bool = False,
            player: int = 0,
    ):
        self.game = game_state  # The game state at this node
        self.parent = parent
        self.action = action  # Action taken to reach this node
        self.player = player  # Player who will act at this node
        self.children: dict[tuple, "MCTSNode"] = {}
        self.untried_actions: list[tuple] = []
        self.filter_floor = filter_floor
        self.visits = 0
        self.wins = np.zeros(game_state.num_players)  # Track wins for each player
        # Get legal actions if game isn't over
        if not self.game.game_over:
            self.untried_actions = self.game.get_legal_moves()
            #print(f"new node with actions ", self.untried_actions[:5])
            if self.filter_floor:
                self.untried_actions = filter_bad_moves(self.untried_actions)

    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over or no legal moves)."""
        return self.game.game_over or (
                len(self.untried_actions) == 0 and len(self.children) == 0
        )

    def ucb1(self, child: "MCTSNode", exploration_weight: float = 1.41) -> float:
        """Calculate UCB1 value for a child node."""
        if child.visits == 0:
            return float("inf")

        # Win rate from the perspective of the player making the decision
        win_rate = child.wins[self.player] / child.visits

        exploration = exploration_weight * math.sqrt(
            math.log(self.visits) / child.visits
        )

        return win_rate + exploration

    def best_child(self, exploration_weight: float = 1.41) -> "MCTSNode":
        """Select the best child using UCB1."""
        return max(
            self.children.values(), key=lambda c: self.ucb1(c, exploration_weight)
        )

    def expand(self) -> "MCTSNode":
        """Expand by adding a new child node."""
        action = self.untried_actions.pop(np.random.randint(len(self.untried_actions)))

        # Create a copy of the game and apply the action
        new_game = self.game.copy()
        new_game.legal_moves = new_game.update_legal_moves()
        #print(f"attempting action {action} at step {self.game.steps}")
        new_game.step(action)

        # Determine next player
        next_player = new_game.current_player

        child = MCTSNode(new_game, parent=self, action=action, player=next_player)
        self.children[action] = child

        return child

    def backpropagate(self, result: np.ndarray):
        """Backpropagate the result up the tree."""
        self.visits += 1
        self.wins += result

        if self.parent:
            self.parent.backpropagate(result)

class MCTSPlayer:
    """Monte Carlo Tree Search player for Azul."""

    def __init__(
            self,
            iterations: int = 1000,
            exploration_weight: float = 1.41,
            time_limit: Optional[float] = None,
            verbose: bool = False,
            terminus_evaluator=HeuristicScore.heuristic_score,
            opponent_sim_strategy=random_mover,
            filter_floor: bool = False,
    ):
        """
        Initialize MCTS player.

        Args:
                iterations: Number of MCTS iterations per move (if no time_limit)
                exploration_weight: UCB1 exploration constant (sqrt(2) ≈ 1.41 is standard)
                time_limit: Optional time limit in seconds per move (overrides iterations)
                verbose: Print search statistics
        """
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.time_limit = time_limit
        self.verbose = verbose
        self.terminus_evaluator = terminus_evaluator
        self.simulate_playout_strategy = opponent_sim_strategy
        self.depth_max = 0
        self.filter_floor = filter_floor

    def select_action(self, game: Azul) -> tuple:
        """Select the best action using MCTS."""
        if game.game_over:
            return ()

        legal_moves = game.get_legal_moves()

        # Handle no legal moves (shouldn't happen, but safety)
        if not legal_moves:
            return ()

        # Handle wall-tiling phase (only one option)
        if legal_moves == [()]:
            return ()

        # If only one legal move, return it immediately
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Create root node
        root = MCTSNode(game.copy(), player=game.current_player, filter_floor=self.filter_floor)

        # Run MCTS iterations
        if self.time_limit:
            start_time = time.time()
            iterations_done = 0
            while time.time() - start_time < self.time_limit:
                self._mcts_iteration(root)
                iterations_done += 1
            if self.verbose:
                print(f"MCTS: {iterations_done} iterations in {self.time_limit:.2f}s")
        else:
            for _ in range(self.iterations):
                self._mcts_iteration(root)
            if self.verbose:
                print(f"MCTS: {self.iterations} iterations")

        # Select best action (most visited child)
        if not root.children:
            # Fallback: if no children were expanded, return first legal move
            # This shouldn't happen normally but provides safety
            if self.verbose:
                print(f"Warning: No children expanded, returning first legal move")
            return legal_moves[0]

        #select
        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)

        if self.verbose:
            self._print_search_stats(root, best_action, self.depth_max)

        return best_action

    def _mcts_iteration(self, root: MCTSNode):
        """Run one iteration of MCTS: select, expand, simulate, backpropagate."""
        node = root
        iteration_depth = 0
        # Selection: traverse tree using UCB1
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
            iteration_depth += 1

        # Expansion: add a new child if not terminal
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
            iteration_depth += 1

        # Simulation: play game to completion using simulation policy (random, heuristic, etc)
        result = self._simulate(node.game)

        # Backpropagation
        node.backpropagate(result)

        if iteration_depth > self.depth_max:
            self.depth_max = iteration_depth

    def _simulate(self, game: Azul) -> np.ndarray:
        """
        Simulate a random playout until hitting a random event (factory refill)
        or game end, then evaluate with heuristic.
        """
        sim_game = game.copy()

        # Track if we're about to hit wall-tiling (which leads to factory refill)
        while not sim_game.game_over:
            moves = sim_game.get_legal_moves()
            if not moves:
                break

            # Check if this is wall-tiling phase - after this, factories get refilled (random event)
            if moves == [()]:
                # Execute wall-tiling but stop before next round's random factory fill
                sim_game.step(())
                # Now evaluate - either game ended or we're at a random event boundary
                break

            # Continue with random action
            action = self.simulate_playout_strategy(sim_game)
            sim_game.step(action)

        # Evaluate position using heuristic
        return self._evaluate_position(sim_game)

    def _evaluate_position(self, game: Azul) -> np.ndarray:
        """
        Evaluate the game position for all players using heuristic.

        Returns array of scores normalized to sum to 1 (like a probability distribution).
        """
        if game.game_over:
            # Terminal state - use actual winner
            result = np.zeros(game.num_players)
            if game.winner is not None:
                result[game.winner] = 1.0
            else:
                # Tie based on scores
                max_score = game.scores.max()
                winners = game.scores == max_score
                result[winners] = 1.0 / winners.sum()
            return result

        # Non-terminal: use heuristic evaluation
        evaluations = np.zeros(game.num_players)

        for player in range(game.num_players):
            evaluations[player] = self.terminus_evaluator(game, player)

        # Convert to probability distribution using softmax
        # Temperature controls how decisive the evaluation is:
        # - Lower temp = more winner-take-all
        # - Higher temp = more uniform
        temperature = 10.0

        # Softmax with numerical stability
        eval_shifted = evaluations - evaluations.max()
        exp_evals = np.exp(eval_shifted / temperature)
        result = exp_evals / exp_evals.sum()

        return result

    @staticmethod
    def _print_search_stats(root: MCTSNode, best_action: tuple, max_depth: int = -1):
        """Print statistics about the search."""
        print(f"  Root visits: {root.visits}")
        print(f"  Children: {len(root.children)}")
        print(f" Max tree depth: {max_depth}")

        # Sort children by visits
        sorted_children = sorted(
            root.children.items(), key=lambda x: x[1].visits, reverse=True
        )

        print("  Top actions:")
        for action, child in sorted_children[:10]:
            win_rate = child.wins[root.player] / child.visits if child.visits > 0 else 0
            source = f"F{action[0]}" if action[0] < root.game.num_factories else "C"
            color = ["B", "Y", "R", "K", "W"][action[1]]
            dest = f"L{action[2] + 1}" if action[2] < 5 else "Floor"
            marker = " <--" if action == best_action else ""
            source_obj = root.game.factories[action[0]] if action[0] < root.game.num_factories else root.game.center
            print(
                f"    {source}->{color}{source_obj[action[1]]}->{dest}: visits={child.visits}, "
                f"win_rate={win_rate:.3f}{marker}"
            )
