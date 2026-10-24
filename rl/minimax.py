"""
Optimal Minimax Agent with Transposition Table Memoization for Tic-Tac-Toe.
Used as an evaluation baseline and teacher opponent.
"""
import random
from typing import Dict, List, Optional, Tuple
import numpy as np


class MinimaxAgent:
    """
    Minimax agent with transposition table memoization for instant decision making.
    """

    def __init__(self, epsilon_error: float = 0.0):
        self.epsilon_error = epsilon_error
        self.memo: Dict[Tuple[int, ...], float] = {}

    def select_action(self, board: np.ndarray, current_player: int) -> int:
        valid_actions = list(np.where(board == 0)[0])
        if not valid_actions:
            raise ValueError("No valid actions on board.")

        # Optional exploration/random mistake rate
        if self.epsilon_error > 0 and random.random() < self.epsilon_error:
            return random.choice(valid_actions)

        best_score = -float("inf")
        best_actions = []

        for action in valid_actions:
            board[action] = current_player
            score = -self._minimax(board, -current_player, -float("inf"), float("inf"))
            board[action] = 0

            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)

    def _minimax(self, board: np.ndarray, player: int, alpha: float, beta: float) -> float:
        state_key = (tuple(board), player)
        if state_key in self.memo:
            return self.memo[state_key]

        winner = self._check_winner(board)
        if winner is not None:
            if winner == 0:
                score = 0.0
            else:
                score = 1.0 if winner == player else -1.0
            self.memo[state_key] = score
            return score

        best_val = -float("inf")
        for action in np.where(board == 0)[0]:
            board[action] = player
            val = -self._minimax(board, -player, -beta, -alpha)
            board[action] = 0

            best_val = max(best_val, val)
            alpha = max(alpha, val)
            if alpha >= beta:
                break

        self.memo[state_key] = best_val
        return best_val

    def _check_winner(self, board: np.ndarray) -> Optional[int]:
        b = board.reshape(3, 3)
        for i in range(3):
            if abs(np.sum(b[i, :])) == 3:
                return int(b[i, 0])
            if abs(np.sum(b[:, i])) == 3:
                return int(b[0, i])

        diag1 = b[0, 0] + b[1, 1] + b[2, 2]
        diag2 = b[0, 2] + b[1, 1] + b[2, 0]
        if abs(diag1) == 3 or abs(diag2) == 3:
            return int(b[1, 1])

        if np.all(board != 0):
            return 0
        return None
