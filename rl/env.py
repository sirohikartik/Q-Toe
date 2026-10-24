"""
Tic-Tac-Toe Environment for Reinforcement Learning.
Supports 2-player interaction, self-play, and canonical state representations.
"""
from typing import List, Optional, Tuple
import numpy as np


class TicTacToeEnv:
    """
    Standard 3x3 Tic-Tac-Toe Environment.
    State representation:
        0: Empty
        1: Player 1 (X)
       -1: Player 2 (O)
    """

    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1  # 1 for X, -1 for O
        self.done = False
        self.winner = None

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.current_player = 1
        self.done = False
        self.winner = None
        return self.get_canonical_state()

    def get_valid_actions(self) -> List[int]:
        if self.done:
            return []
        return list(np.where(self.board == 0)[0])

    def get_canonical_state(self) -> np.ndarray:
        """
        Returns the board from the perspective of the current player:
        +1: current player
        -1: opponent
         0: empty
        """
        return (self.board * self.current_player).astype(np.float32)

    def check_winner(self) -> Optional[int]:
        """
        Returns 1 if Player 1 won, -1 if Player 2 won, 0 for draw, None if game continues.
        """
        b = self.board.reshape(3, 3)

        # Check rows & columns
        for i in range(3):
            if abs(np.sum(b[i, :])) == 3:
                return int(b[i, 0])
            if abs(np.sum(b[:, i])) == 3:
                return int(b[0, i])

        # Check diagonals
        diag1 = b[0, 0] + b[1, 1] + b[2, 2]
        diag2 = b[0, 2] + b[1, 1] + b[2, 0]
        if abs(diag1) == 3:
            return int(b[1, 1])
        if abs(diag2) == 3:
            return int(b[1, 1])

        # Draw check
        if len(self.get_valid_actions()) == 0:
            return 0

        return None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Executes an action for the current player.
        Returns:
            canonical_next_state, reward_for_player_who_moved, done, info
        """
        if self.done:
            raise ValueError("Game is already terminated. Call reset() first.")
        if self.board[action] != 0:
            raise ValueError(f"Invalid action {action}; cell already taken.")

        player_who_moved = self.current_player
        self.board[action] = player_who_moved

        winner = self.check_winner()

        if winner is not None:
            self.done = True
            self.winner = winner
            if winner == player_who_moved:
                reward = 1.0
            elif winner == 0:
                reward = 0.5  # Draw is positive in zero-sum game
            else:
                reward = -1.0
        else:
            reward = 0.0
            self.current_player *= -1

        info = {"winner": self.winner, "current_player": self.current_player}
        return self.get_canonical_state(), reward, self.done, info

    def render(self) -> str:
        symbols = {0: "·", 1: "X", -1: "O"}
        rows = []
        for i in range(3):
            row = [symbols[self.board[i * 3 + j]] for j in range(3)]
            rows.append(" " + " | ".join(row) + " ")
        return "\n---+---+---\n".join(rows)
