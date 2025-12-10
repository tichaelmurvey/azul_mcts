from copy import deepcopy, copy

import numpy as np
import numpy.typing as npt
from typing import Optional
from enum import IntEnum

# Constants
NUM_COLORS = 5
NUM_PATTERN_LINES = 5
WALL_SIZE = 5
FLOOR_LINE_SIZE = 7
TILES_PER_COLOR = 20
TILES_PER_FACTORY = 4


class TileColor(IntEnum):
    BLUE = 0
    YELLOW = 1
    RED = 2
    BLACK = 3
    WHITE = 4


class Phase(IntEnum):
    FACTORY_OFFER = 0
    WALL_TILING = 1


# Floor line penalties: positions 0-1 = -1, 2-4 = -2, 5-6 = -3
FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3]

# Standard wall pattern (colored side) - row, col -> color
# Each row has colors shifted by 1 position
STANDARD_WALL_PATTERN = np.array(
    [
        [0, 1, 2, 3, 4],  # Row 0: Blue, Yellow, Red, Black, White
        [4, 0, 1, 2, 3],  # Row 1: White, Blue, Yellow, Red, Black
        [3, 4, 0, 1, 2],  # Row 2: Black, White, Blue, Yellow, Red
        [2, 3, 4, 0, 1],  # Row 3: Red, Black, White, Blue, Yellow
        [1, 2, 3, 4, 0],  # Row 4: Yellow, Red, Black, White, Blue
    ],
    dtype=np.int_,
)

WALL_COLUMN_FOR_COLOR = np.array(
    [
        [0, 1, 2, 3, 4],  # row 0: color 0->col 0, color 1->col 1, ...
        [1, 2, 3, 4, 0],  # row 1: color 0->col 1, etc.
        [2, 3, 4, 0, 1],
        [3, 4, 0, 1, 2],
        [4, 0, 1, 2, 3],
    ],
    dtype=np.int8,
)

# Define color codes at module level
ANSI_COLORS = {
    0: "\033[94m",  # Blue
    1: "\033[93m",  # Yellow
    2: "\033[91m",  # Red
    3: "\033[35m",  # Black (dark gray for visibility)
    4: "\033[97m",  # White (bright white)
}
ANSI_RESET = "\033[0m"


def colorize(char: str, color_idx: int) -> str:
    """Wrap a character in ANSI color codes."""
    return f"{ANSI_COLORS[color_idx]}{char}{ANSI_RESET}"


class Azul:
    """
    Azul game environment for reinforcement learning.

    State representation:
    - Factory displays: (num_factories, NUM_COLORS) - count of each color per factory
    - Center: (NUM_COLORS,) - count of each color in center
    - First player marker in center: bool
    - Per player:
        - Pattern lines: (NUM_PATTERN_LINES, 2) - [color (-1 if empty), count]
        - Wall: (WALL_SIZE, WALL_SIZE) - bool, whether tile is placed
        - Floor line: int - number of tiles (0-7+)
        - Score: int
        - Has first player marker: bool

    Action representation for Factory Offer phase:
    - (source, color, destination)
    - source: 0 to num_factories-1 for factory, num_factories for center
    - color: 0-4 for tile colors
    - destination: 0-4 for pattern lines, 5 for floor line
    """

    def __init__(self, num_players: int = 2):
        if num_players < 2 or num_players > 4:
            raise ValueError("Azul supports 2-4 players")

        self.num_players = num_players

        # Number of factory displays based on player count
        if num_players == 2:
            self.num_factories = 5
        elif num_players == 3:
            self.num_factories = 7
        else:  # 4 players
            self.num_factories = 9
        self.legal_moves = []
        self.steps = 0
        self.reset()

    def reset(self):
        """Reset game to initial state."""
        # Tile bag: count of each color remaining
        self.bag = np.full(NUM_COLORS, TILES_PER_COLOR, dtype=np.int_)

        # Lid/discard: tiles removed from game temporarily
        self.lid = np.zeros(NUM_COLORS, dtype=np.int_)

        # Factory displays: (num_factories, NUM_COLORS)
        self.factories = np.zeros((self.num_factories, NUM_COLORS), dtype=np.int_)

        # Center of table: count of each color
        self.center = np.zeros(NUM_COLORS, dtype=np.int_)

        # Player states
        # Pattern lines: (num_players, NUM_PATTERN_LINES, 2) -> [color, count]
        # color = -1 means empty
        self.pattern_lines = np.full(
            (self.num_players, NUM_PATTERN_LINES, 2), -1, dtype=np.int_
        )
        self.pattern_lines[:, :, 1] = 0  # counts start at 0

        # Walls: (num_players, WALL_SIZE, WALL_SIZE) -> bool
        self.walls = np.zeros((self.num_players, WALL_SIZE, WALL_SIZE), dtype=np.bool_)

        # Floor lines: count of tiles per player
        self.floor_lines = np.zeros(self.num_players, dtype=np.int_)

        # Scores
        self.scores: npt.NDArray[np.int_] = np.zeros(self.num_players, dtype=np.int_)

        # Who has first player marker (-1 if in center)
        self.first_player_marker_holder = -1

        # Current player and phase
        self.current_player = 0  # Will be set properly when game starts
        self.starting_player = 0
        self.phase = Phase.FACTORY_OFFER

        # Game over flag
        self.game_over = False
        self.winner = None
        self.steps = 0

        # Fill factories to start the game
        self._fill_factories()
        self.legal_moves = self.update_legal_moves()
        return self._get_state()

    def _draw_tiles(self, count: int) -> npt.NDArray[np.int_]:
        """Draw tiles from bag, refilling from lid if necessary."""
        drawn = np.zeros(NUM_COLORS, dtype=np.int_)
        remaining = count

        while remaining > 0:
            total_in_bag = self.bag.sum()

            if total_in_bag == 0:
                # Refill bag from lid
                self.bag = self.lid.copy()
                self.lid = np.zeros(NUM_COLORS, dtype=np.int_)
                total_in_bag = self.bag.sum()

                if total_in_bag == 0:
                    # No tiles left anywhere
                    break

            # Draw one tile at a time (weighted random)
            probs = self.bag / total_in_bag
            color = np.random.choice(NUM_COLORS, p=probs)
            self.bag[color] -= 1
            drawn[color] += 1
            remaining -= 1

        return drawn

    def _fill_factories(self):
        """Fill all factory displays with 4 tiles each."""
        self.factories = np.zeros((self.num_factories, NUM_COLORS), dtype=np.int_)
        for i in range(self.num_factories):
            tiles = self._draw_tiles(TILES_PER_FACTORY)
            self.factories[i] = tiles

        # Reset center and first player marker
        self.center = np.zeros(NUM_COLORS, dtype=np.int_)
        self.first_player_marker_holder = -1

    def _get_state(self) -> dict:
        """Return current game state as a dictionary."""
        return {
            "factories": self.factories.copy(),
            "center": self.center.copy(),
            "first_player_marker_holder": self.first_player_marker_holder,
            "pattern_lines": self.pattern_lines.copy(),
            "walls": self.walls.copy(),
            "floor_lines": self.floor_lines.copy(),
            "scores": self.scores.copy(),
            "current_player": self.current_player,
            "phase": self.phase,
            "game_over": self.game_over,
            "winner": self.winner,
        }

    def get_starting_state(self) -> dict:
        """Return the starting state of a new game."""
        return self.reset()

    def get_starting_player(self) -> int:
        """Return the starting player index."""
        return self.starting_player

    def get_current_player(self, state: Optional[dict] = None) -> int:
        """Return current player to act."""
        if state is not None:
            return state["current_player"]
        return self.current_player

    def get_next_player(self, player: int, state: Optional[dict] = None) -> int:
        """Return the next player in turn order."""
        return (player + 1) % self.num_players

    def get_legal_moves(self):
        return self.legal_moves

    def update_legal_moves(self, state: Optional[dict] = None) -> list[tuple]:
        """
        Return list of legal moves for current player.

        In Factory Offer phase:
            Returns list of (source, color, destination) tuples
            - source: factory index (0 to num_factories-1) or num_factories for center
            - color: tile color (0-4)
            - destination: pattern line (0-4) or 5 for floor

        In Wall Tiling phase:
            Returns [()] - single "confirm" action (wall tiling is automatic)
        """
        if state is None:
            state = self._get_state()

        if state["game_over"]:
            return []

        if state["phase"] == Phase.WALL_TILING:
            # Wall tiling is automatic, just return a dummy action
            return [()]

        legal_moves = []
        player = state["current_player"]
        player_pattern_lines = state["pattern_lines"][player]
        player_wall = state["walls"][player]

        # Check all sources (factories + center)
        sources = []

        # Add factories that have tiles
        for factory_idx in range(self.num_factories):
            if state["factories"][factory_idx].sum() > 0:
                sources.append((factory_idx, state["factories"][factory_idx]))

        # Add center if it has tiles
        if state["center"].sum() > 0:
            sources.append((self.num_factories, state["center"]))

        for source_idx, tile_counts in sources:
            for color in range(NUM_COLORS):
                if tile_counts[color] == 0:
                    continue

                # Check each possible destination
                for dest in range(
                    NUM_PATTERN_LINES + 1
                ):  # 0-4 = pattern lines, 5 = floor
                    if dest == 5:
                        # Floor is always a legal destination
                        legal_moves.append((source_idx, color, dest))
                    else:
                        # Check if pattern line can accept this color
                        pattern_line = player_pattern_lines[dest]
                        line_color, line_count = pattern_line[0], pattern_line[1]
                        line_capacity = dest + 1

                        # Check if line is empty or has same color
                        if line_count > 0 and line_color != color:
                            continue

                        # Check if line is already full
                        if line_count >= line_capacity:
                            continue

                        # Check if this color is already on the wall in this row
                        wall_col = self._get_wall_column(dest, color)
                        if player_wall[dest, wall_col]:
                            continue

                        legal_moves.append((source_idx, color, dest))

        # If no moves are possible (shouldn't happen in normal play)
        # but just in case, allow taking any available tiles to floor
        if not legal_moves:
            for source_idx, tile_counts in sources:
                for color in range(NUM_COLORS):
                    if tile_counts[color] > 0:
                        legal_moves.append((source_idx, color, 5))

        return legal_moves

    @staticmethod
    def _get_wall_column(row: int, color: int) -> int:
        """Get the column where a color should be placed on the standard wall pattern."""
        return WALL_COLUMN_FOR_COLOR[row, color]

    def step(self, action: tuple) -> tuple[dict, bool]:
        """
        Execute an action and return (new_state, done).

        action: (source, color, destination) for Factory Offer phase
                () for Wall Tiling phase (automatic)
        """
        if self.game_over:
            self.legal_moves = self.update_legal_moves()
            return self._get_state(), True

        if self.phase == Phase.WALL_TILING:
            self._do_wall_tiling()
            self.legal_moves = self.update_legal_moves()
            return self._get_state(), self.game_over

        # Factory Offer phase
        source, color, destination = action
        player = self.current_player

        # Validate action
        if action not in self.legal_moves:
            print(
                f"Action {action} at step {self.steps} not in legal moves {self.legal_moves[:5]}"
            )
            raise ValueError(f"Illegal move: {action}")
        self.steps += 1
        # Get tiles from source
        if source < self.num_factories:
            # Taking from factory
            num_tiles = self.factories[source, color]
            self.factories[source, color] = 0

            # Move remaining tiles to center
            for c in range(NUM_COLORS):
                if c != color:
                    self.center[c] += self.factories[source, c]
                    self.factories[source, c] = 0
        else:
            # Taking from center
            num_tiles = self.center[color]
            self.center[color] = 0

            # First player to take from center gets the marker
            if self.first_player_marker_holder == -1:
                self.first_player_marker_holder = player
                # Marker goes to floor line
                self.floor_lines[player] += 1

        # Place tiles
        if destination == 5:
            # All to floor
            self.floor_lines[player] += num_tiles
        else:
            # To pattern line
            line_capacity = destination + 1
            current_count = self.pattern_lines[player, destination, 1]
            space_available = line_capacity - current_count

            tiles_to_line = min(num_tiles, space_available)
            tiles_to_floor = num_tiles - tiles_to_line

            self.pattern_lines[player, destination, 0] = color
            self.pattern_lines[player, destination, 1] = current_count + tiles_to_line
            self.floor_lines[player] += tiles_to_floor

        # Check if Factory Offer phase is over
        factories_empty = self.factories.sum() == 0
        center_empty = self.center.sum() == 0

        if factories_empty and center_empty:
            # Move to Wall Tiling phase
            self.phase = Phase.WALL_TILING
        else:
            # Next player's turn
            self.current_player = self.get_next_player(self.current_player)

        self.legal_moves = self.update_legal_moves()
        # print(f"stepped to {self.steps}, set legal moves {self.legal_moves[:5]}")
        return self._get_state(), self.game_over

    def _do_wall_tiling(self):
        """Execute wall tiling phase for all players."""
        for player in range(self.num_players):
            self._tile_wall_for_player(player)

        # Apply floor line penalties
        for player in range(self.num_players):
            penalty = 0
            floor_count = min(self.floor_lines[player], FLOOR_LINE_SIZE)
            for i in range(floor_count):
                penalty += FLOOR_PENALTIES[i]

            self.scores[player] = max(0, self.scores[player] + penalty)

            # Clear floor line (tiles to lid, except first player marker)
            # For simplicity, we don't track which specific tiles went to floor
            self.floor_lines[player] = 0

        # Check for game end
        game_ended = False
        for player in range(self.num_players):
            for row in range(WALL_SIZE):
                if self.walls[player, row, :].sum() == WALL_SIZE:
                    game_ended = True
                    break
            if game_ended:
                break

        if game_ended:
            self._end_game()
        else:
            # Prepare next round
            self._prepare_next_round()

    def _tile_wall_for_player(self, player: int):
        """Move tiles from complete pattern lines to wall and score."""
        for row in range(NUM_PATTERN_LINES):
            line_color = self.pattern_lines[player, row, 0]
            line_count = self.pattern_lines[player, row, 1]
            line_capacity = row + 1

            if line_count < line_capacity or line_color == -1:
                # Line not complete, skip
                continue

            # Line is complete - move one tile to wall
            col = self._get_wall_column(row, line_color)
            self.walls[player, row, col] = True

            # Score the placement
            points = self._score_tile_placement(player, row, col)
            self.scores[player] += points

            # Remaining tiles go to lid
            self.lid[line_color] += line_count - 1

            # Clear the pattern line
            self.pattern_lines[player, row, 0] = -1
            self.pattern_lines[player, row, 1] = 0

    def _score_tile_placement(self, player: int, row: int, col: int) -> int:
        """Calculate points for placing a tile at (row, col)."""
        wall = self.walls[player]

        # Count horizontal chain
        h_count = 1
        # Left
        for c in range(col - 1, -1, -1):
            if wall[row, c]:
                h_count += 1
            else:
                break
        # Right
        for c in range(col + 1, WALL_SIZE):
            if wall[row, c]:
                h_count += 1
            else:
                break

        # Count vertical chain
        v_count = 1
        # Up
        for r in range(row - 1, -1, -1):
            if wall[r, col]:
                v_count += 1
            else:
                break
        # Down
        for r in range(row + 1, WALL_SIZE):
            if wall[r, col]:
                v_count += 1
            else:
                break

        # Calculate points
        if h_count == 1 and v_count == 1:
            # No adjacent tiles
            return 1

        points = 0
        if h_count > 1:
            points += h_count
        if v_count > 1:
            points += v_count

        # If both are > 1, the tile is counted in both chains
        # If only one direction has adjacent tiles, we still count the placed tile
        if h_count > 1 and v_count == 1:
            points = h_count
        elif v_count > 1 and h_count == 1:
            points = v_count

        return points

    def _prepare_next_round(self):
        """Set up for the next round."""
        # Starting player for next round is whoever took the first player marker
        if self.first_player_marker_holder >= 0:
            self.starting_player = self.first_player_marker_holder
            self.current_player = self.starting_player

        # Refill factories
        self._fill_factories()

        # Back to Factory Offer phase
        self.phase = Phase.FACTORY_OFFER

    def _end_game(self):
        """Calculate final scores and determine winner."""
        # Bonus points
        for player in range(self.num_players):
            wall = self.walls[player]

            # 2 points per complete horizontal line
            for row in range(WALL_SIZE):
                if wall[row, :].sum() == WALL_SIZE:
                    self.scores[player] += 2

            # 7 points per complete vertical line
            for col in range(WALL_SIZE):
                if wall[:, col].sum() == WALL_SIZE:
                    self.scores[player] += 7

            # 10 points per complete color (all 5 tiles of one color)
            for color in range(NUM_COLORS):
                color_count = 0
                for row in range(WALL_SIZE):
                    col = self._get_wall_column(row, color)
                    if wall[row, col]:
                        color_count += 1
                if color_count == 5:
                    self.scores[player] += 10

        self.game_over = True

        # Determine winner
        max_score = self.scores.max()
        winners = np.where(self.scores == max_score)[0]

        if len(winners) == 1:
            self.winner = winners[0]
        else:
            # Tiebreaker: most complete horizontal lines
            best_h_lines = -1
            tied_winner = None
            for player in winners:
                h_lines = 0
                for row in range(WALL_SIZE):
                    if self.walls[player, row, :].sum() == WALL_SIZE:
                        h_lines += 1
                if h_lines > best_h_lines:
                    best_h_lines = h_lines
                    tied_winner = player
                elif h_lines == best_h_lines:
                    tied_winner = None  # Still tied

            self.winner = tied_winner  # None if still tied (shared victory)

    def check_winner(self, state: Optional[dict] = None) -> Optional[int]:
        """Return winner if game is over, None otherwise."""
        if state is None:
            state = self._get_state()

        if state["game_over"]:
            return state["winner"]
        return None

    def print_game(self, state: Optional[dict] = None, player: Optional[int] = None):
        """Print the current game state."""
        if state is None:
            state = self._get_state()

        if player is not None:
            self.print_player_board(state, player)
            return

        color_chars = ["B", "Y", "R", "K", "W"]

        print("=" * 50)
        print(
            f"Phase: {'Factory Offer' if state['phase'] == Phase.FACTORY_OFFER else 'Wall Tiling'}"
        )
        print(f"Current Player: {state['current_player']}")
        print()
        self.print_boards_side_by_side(state)
        # for p in range(self.num_players):
        #     self.print_player_board(state, p)

        print("Factories:")
        for i, factory in enumerate(state["factories"]):
            tiles = ", ".join(
                filter(
                    len,
                    [
                        colorize(color_chars[c], c) * factory[c]
                        for c in range(NUM_COLORS)
                    ],
                )
            )
            print(f"  Factory {i}: {tiles if tiles else '(empty)'}")

        center_tiles = "".join(
            [
                colorize(color_chars[c], c) * state["center"][c]
                for c in range(NUM_COLORS)
            ]
        )
        print(
            f"  Center: {center_tiles} {'-1' if self.first_player_marker_holder == -1 else ''}"
        )
        print()

    def print_boards_side_by_side(self, state: dict):
        """Print all player boards side-by-side like physical Azul boards."""
        color_chars = ["B", "Y", "R", "K", "W"]
        empty_slot = "☐"  # or '□' if this doesn't render well
        board_width = 22  # Width of each player's board section

        # Header row with player names and scores
        header = ""
        for p in range(self.num_players):
            player_header = f"Player {p} (Score: {state['scores'][p]})"
            header += player_header.ljust(board_width) + "   "
        print(header)
        print("-" * (board_width * self.num_players + 3 * (self.num_players - 1)))

        # Pattern lines label
        label_row = ""
        for p in range(self.num_players):
            label_row += "Pattern Lines  Wall".ljust(board_width) + "   "
        print(label_row)

        # Print each row (pattern line + wall row combined)
        for row in range(WALL_SIZE):
            line = ""
            for p in range(self.num_players):
                # Build pattern line string (right-aligned, dots on left)
                color = state["pattern_lines"][p, row, 0]
                count = state["pattern_lines"][p, row, 1]
                capacity = row + 1

                # Pad to 5 characters (max pattern line size)
                padding = 5 - capacity
                if color >= 0:
                    pattern_str = (
                        " " * padding
                        + "☐" * (capacity - count)
                        + colorize(color_chars[color], color) * count
                    )
                else:
                    pattern_str = " " * padding + "☐" * capacity

                # Build wall row string
                wall_str = ""
                for col in range(WALL_SIZE):
                    wall_color = STANDARD_WALL_PATTERN[row, col]
                    if state["walls"][p, row, col]:
                        # Filled slot: solid colored letter
                        wall_str += colorize(color_chars[wall_color], wall_color)
                    else:
                        # Empty slot: colored box outline
                        wall_str += colorize(empty_slot, wall_color)

                # Combine: pattern line (5) + separator (4) + wall (5) + padding
                player_section = f"{pattern_str}    {wall_str}"
                line += player_section.ljust(board_width) + "   "

            print(f"  {row + 1}: {line}")

        # Floor lines
        print()
        floor_row = ""
        for p in range(self.num_players):
            floor_count = state["floor_lines"][p]
            floor_str = "Floor: " + "X" * min(floor_count, FLOOR_LINE_SIZE)
            if floor_count > FLOOR_LINE_SIZE:
                floor_str += f"(+{floor_count - FLOOR_LINE_SIZE})"
            floor_row += floor_str.ljust(board_width) + "   "
        print(floor_row)
        print()

    @staticmethod
    def print_player_board(state: dict, player: int):
        """Print a single player's board (fallback for single-player view)."""
        color_chars = ["B", "Y", "R", "K", "W"]
        empty_slot = "☐"
        print(f"Player {player} (Score: {state['scores'][player]}):")

        print("  Pattern Lines    Wall")
        for row in range(NUM_PATTERN_LINES):
            color = state["pattern_lines"][player, row, 0]
            count = state["pattern_lines"][player, row, 1]
            capacity = row + 1

            padding = 5 - capacity
            if color >= 0:
                pattern_str = (
                    " " * padding
                    + "." * (capacity - count)
                    + colorize(color_chars[color], color) * count
                )
            else:
                pattern_str = " " * padding + "." * capacity

            wall_str = ""
            for col in range(WALL_SIZE):
                wall_color = STANDARD_WALL_PATTERN[row, col]
                if state["walls"][player, row, col]:
                    wall_str += colorize(color_chars[wall_color], wall_color)
                else:
                    wall_str += colorize(empty_slot, wall_color)

            print(f"    {row + 1}: {pattern_str}    {wall_str}")

        floor_count = state["floor_lines"][player]
        floor_str = "X" * min(floor_count, FLOOR_LINE_SIZE)
        if floor_count > FLOOR_LINE_SIZE:
            floor_str += f"(+{floor_count - FLOOR_LINE_SIZE})"
        print(f"  Floor: {floor_str}")
        print()

    def get_observation(self, player: Optional[int] = None) -> npt.NDArray[np.float32]:
        """
        Return a flattened observation array suitable for RL.

        The observation includes:
        - Factories: (num_factories * NUM_COLORS) = varies by player count
        - Center: (NUM_COLORS) = 5
        - First player marker in center: 1
        - Current player's pattern lines: (NUM_PATTERN_LINES * 2) = 10
        - Current player's wall: (WALL_SIZE * WALL_SIZE) = 25
        - Current player's floor line: 1
        - Current player's score: 1
        - Opponent pattern lines: ((num_players-1) * NUM_PATTERN_LINES * 2)
        - Opponent walls: ((num_players-1) * WALL_SIZE * WALL_SIZE)
        - Opponent floor lines: (num_players-1)
        - Opponent scores: (num_players-1)

        For a 2-player game: 5*5 + 5 + 1 + 10 + 25 + 1 + 1 + 10 + 25 + 1 + 1 = 105
        """
        if player is None:
            player = self.current_player

        obs_parts = []

        # Factories (flattened)
        obs_parts.append(self.factories.flatten().astype(np.float32))

        # Center
        obs_parts.append(self.center.astype(np.float32))

        # First player marker in center
        obs_parts.append(
            np.array([float(self.first_player_marker_holder)], dtype=np.float32)
        )

        # Current player's state
        obs_parts.append(self.pattern_lines[player].flatten().astype(np.float32))
        obs_parts.append(self.walls[player].flatten().astype(np.float32))
        obs_parts.append(np.array([self.floor_lines[player]], dtype=np.float32))
        obs_parts.append(np.array([self.scores[player]], dtype=np.float32))

        # Other players' states (in order after current player)
        for i in range(1, self.num_players):
            other = (player + i) % self.num_players
            obs_parts.append(self.pattern_lines[other].flatten().astype(np.float32))
            obs_parts.append(self.walls[other].flatten().astype(np.float32))
            obs_parts.append(np.array([self.floor_lines[other]], dtype=np.float32))
            obs_parts.append(np.array([self.scores[other]], dtype=np.float32))

        return np.concatenate(obs_parts)

    def get_observation_shape(self) -> tuple[int]:
        """Return the shape of the observation array."""
        # Calculate based on num_players
        factory_size = self.num_factories * NUM_COLORS
        center_size = NUM_COLORS
        fpm_size = 1
        per_player = (
            NUM_PATTERN_LINES * 2 + WALL_SIZE * WALL_SIZE + 1 + 1
        )  # pattern + wall + floor + score

        total = factory_size + center_size + fpm_size + per_player * self.num_players
        return (total,)

    def get_action_space_size(self) -> int:
        """
        Return the size of the action space.

        Actions: (source, color, destination)
        - source: num_factories + 1 (factories + center)
        - color: NUM_COLORS
        - destination: NUM_PATTERN_LINES + 1 (lines + floor)

        Plus 1 for the wall-tiling "confirm" action.
        """
        return (self.num_factories + 1) * NUM_COLORS * (NUM_PATTERN_LINES + 1) + 1

    def action_to_index(self, action: tuple) -> int:
        """Convert an action tuple to a flat index."""
        if action == ():
            # Wall tiling confirm action
            return self.get_action_space_size() - 1

        source, color, destination = action
        # Encode as: source * (NUM_COLORS * 6) + color * 6 + destination
        return (
            source * (NUM_COLORS * (NUM_PATTERN_LINES + 1))
            + color * (NUM_PATTERN_LINES + 1)
            + destination
        )

    def index_to_action(self, index: int) -> tuple:
        """Convert a flat index to an action tuple."""
        if index == self.get_action_space_size() - 1:
            return ()

        destination = index % (NUM_PATTERN_LINES + 1)
        index //= NUM_PATTERN_LINES + 1
        color = index % NUM_COLORS
        source = index // NUM_COLORS

        return source, color, destination

    def get_action_mask(self) -> npt.NDArray[np.bool_]:
        """Return a boolean mask of legal actions."""
        mask = np.zeros(self.get_action_space_size(), dtype=np.bool_)

        for action in self.get_legal_moves():
            mask[self.action_to_index(action)] = True

        return mask

    def copy(self) -> "Azul":
        """Fast shallow copy with numpy array copies."""
        new_game = object.__new__(Azul)
        new_game.num_players = self.num_players
        new_game.num_factories = self.num_factories
        new_game.bag = self.bag.copy()
        new_game.lid = self.lid.copy()
        new_game.factories = self.factories.copy()
        new_game.center = self.center.copy()
        new_game.pattern_lines = self.pattern_lines.copy()
        new_game.walls = self.walls.copy()
        new_game.floor_lines = self.floor_lines.copy()
        new_game.scores = self.scores.copy()
        new_game.first_player_marker_holder = self.first_player_marker_holder
        new_game.current_player = self.current_player
        new_game.starting_player = self.starting_player
        new_game.phase = self.phase
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.legal_moves = self.legal_moves.copy()
        new_game.steps = self.steps
        # new_game = deepcopy(self)
        return new_game
