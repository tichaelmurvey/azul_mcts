import numpy as np

from AzulEnvHeuristic import Azul, Phase, FLOOR_PENALTIES
from AzulMCScorePolicies import HeuristicScore


def random_mover(game: Azul):
    moves = game.get_legal_moves()
    if not moves:
        return ()
    return moves[np.random.randint(len(moves))]

def heuristic_mover(game: Azul):
    if game.game_over or game.phase == Phase.WALL_TILING:
        return ()
    moves = game.get_legal_moves()
    if not moves:
        return ()
    best_move = moves[0]
    best_score = 0
    for move in moves:
        local_game = game.copy()
        local_game.step(move)
        score = HeuristicScore.heuristic_score(local_game, local_game.get_current_player())
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def heuristic_mover_fast(game: Azul):
    moves = game.get_legal_moves()
    moves = filter_bad_moves(moves)
    if not moves:
        return ()

    player = game.current_player
    best_move = moves[0]
    best_delta = float('-inf')

    for move in moves:
        delta = estimate_move_value(game, move, player)
        if delta > best_delta:
            best_delta = delta
            best_move = move
    return best_move


def estimate_move_value(game: Azul, move: tuple, player: int) -> float:
    """Estimate value without copying game state."""
    source, color, dest = move

    # How many tiles are we taking?
    if source < game.num_factories:
        num_tiles = game.factories[source, color]
    else:
        num_tiles = game.center[color]

    if dest == 5:  # Floor
        return sum(FLOOR_PENALTIES[:min(num_tiles, 7)])

    # Pattern line destination
    line_capacity = dest + 1
    current_count = game.pattern_lines[player, dest, 1]
    space = line_capacity - current_count
    tiles_to_line = min(num_tiles, space)
    tiles_to_floor = num_tiles - tiles_to_line

    # Estimate: progress toward completion + adjacency bonus - floor penalty
    value = tiles_to_line * 1.5  # Progress value
    if current_count + tiles_to_line == line_capacity:
        value += 3.0  # Completion bonus

    if tiles_to_floor > 0:
        value -= sum(FLOOR_PENALTIES[:min(tiles_to_floor, 7)])

    return value


def filter_bad_moves(moves):
    has_line_option = [False] * 5

    for source, color, dest in moves:
        if dest < 5:
            has_line_option[color] = True

    # Keep if: NOT a floor move, OR no line alternative exists
    filtered_moves = [m for m in moves if m[2] < 5 or not has_line_option[m[1]]]
    return filtered_moves
