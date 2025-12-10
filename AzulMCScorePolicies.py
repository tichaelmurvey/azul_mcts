import numpy as np

from AzulEnvHeuristic import Azul, FLOOR_LINE_SIZE, FLOOR_PENALTIES, NUM_PATTERN_LINES, WALL_SIZE, NUM_COLORS, \
    STANDARD_WALL_PATTERN
type Wall = np.ndarray[tuple[int, int], np.dtype[np.bool_]]


class HeuristicScore:
    @staticmethod
    def heuristic_score(game: Azul, player: int) -> float:
        """
        Calculate heuristic score for a player based on:
        - Current score
        - Progress toward end-game bonuses (rows, columns, colors)
        - Pattern line progress (tiles that will score next wall-tiling)
        - Floor line penalty
        """
        score = float(game.scores[player])
        wall: np.ndarray[tuple[int, int], np.dtype[np.bool_]] = game.walls[player]
        pattern_lines = game.pattern_lines[player]

        # === Progress toward horizontal line bonus (2 points each) ===
        # Value partial rows based on completion percentage squared
        # (completing a row is more valuable than spreading tiles)
        horizontal_bonus = ScoreCalcs.get_horizontal_bonus(wall)

        # === Progress toward vertical line bonus (7 points each) ===
        vertical_bonus = ScoreCalcs.get_vertical_bonus(wall)

        # === Progress toward color bonus (10 points each) ===
        color_bonus = ScoreCalcs.get_color_bonus(wall)

        # === Pattern line progress (potential future points) ===
        # Estimate points for complete and incomplete lines
        pattern_potential = ScoreCalcs.get_pattern_potential(wall, pattern_lines)

        # === Floor line penalty ===
        floor_penalty = 0.0
        floor_count = min(game.floor_lines[player], FLOOR_LINE_SIZE)
        for i in range(floor_count):
            floor_penalty += FLOOR_PENALTIES[i]  # These are already negative

        # Combine all factors
        # Weight the bonus progress less since they're speculative
        total = (
            score
            + 0.5 * horizontal_bonus
            + 0.5 * vertical_bonus
            + 0.5 * color_bonus
            + 0.8 * pattern_potential  # Pattern lines are more certain
            + floor_penalty  # Redundant when resolving at scoring stage
        )

        return total


class ScoreCalcs:
    @staticmethod
    def get_pattern_potential(wall, pattern_lines):
        pattern_potential = 0.0
        for row in range(NUM_PATTERN_LINES):
            pattern_potential += ScoreCalcs.row_pattern_potential(wall, pattern_lines, row)
        return pattern_potential
    @staticmethod
    def row_pattern_potential(wall, pattern_lines, row):
        line_color = pattern_lines[row, 0]
        line_count = pattern_lines[row, 1]
        line_capacity = row + 1
        row_pattern_potential = 0.0

        if line_color >= 0:
            # subtracted because total tile capacity doesn't impact score. 4/5 filled and 1/2 filled present the same opportunity for future actions. Note, it's possible 2 or 3 spots remaining could be better than 1 due to factory amts, worth experimenting.
            line_progress = line_count - line_capacity

            # This line will score - estimate the points
            col = ScoreCalcs._get_wall_column_static(row, line_color)

            # Count adjacent tiles for scoring estimate
            h_count = 1
            for c in range(col - 1, -1, -1):
                if wall[row, c]:
                    h_count += 1
                else:
                    break
            for c in range(col + 1, WALL_SIZE):
                if wall[row, c]:
                    h_count += 1
                else:
                    break

            v_count = 1
            for r in range(row - 1, -1, -1):
                if wall[r, col]:
                    v_count += 1
                else:
                    break
            for r in range(row + 1, WALL_SIZE):
                if wall[r, col]:
                    v_count += 1
                else:
                    break

            if h_count == 1 and v_count == 1:
                row_pattern_potential += 1
            else:
                if h_count > 1:
                    row_pattern_potential += h_count
                if v_count > 1:
                    row_pattern_potential += v_count
            row_pattern_potential *= line_progress  # weights value by progress
        return row_pattern_potential

    @staticmethod
    def get_color_bonus(wall: Wall):
        color_bonus = 0.0
        for color in range(NUM_COLORS):
            tiles_of_color = 0
            for row in range(WALL_SIZE):
                col = ScoreCalcs._get_wall_column_static(row, color)
                if wall[row, col]:
                    tiles_of_color += 1
            if tiles_of_color == 5:
                color_bonus += 10.0  # Already complete
            else:
                color_bonus += 10.0 * (tiles_of_color / 5) ** 2
        return color_bonus

    @staticmethod
    def get_vertical_bonus(wall: Wall):
        vertical_bonus = 0.0
        for col in range(WALL_SIZE):
            tiles_in_col = wall[:, col].sum()
            if tiles_in_col == WALL_SIZE:
                vertical_bonus += 7.0  # Already complete
            else:
                vertical_bonus += 7.0 * (tiles_in_col / WALL_SIZE) ** 2
        return vertical_bonus

    @staticmethod
    def get_horizontal_bonus(wall: Wall):
        horizontal_bonus = 0.0
        for row in range(WALL_SIZE):
            tiles_in_row = wall[row, :].sum()
            if tiles_in_row == WALL_SIZE:
                horizontal_bonus += 2.0  # Already complete
            else:
                # Partial credit: quadratic to reward near-completion
                horizontal_bonus += 2.0 * (tiles_in_row / WALL_SIZE) ** 2
        return horizontal_bonus

    @staticmethod
    def _get_wall_column_static(row: int, color: int) -> int:
        """Get the column where a color should be placed (static version)."""
        for col in range(WALL_SIZE):
            if STANDARD_WALL_PATTERN[row, col] == color:
                return col
        return 0
