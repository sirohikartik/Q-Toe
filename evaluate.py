"""
Evaluation and Interactive CLI Match for Tic-Tac-Toe RL Agent.
"""
import argparse
import random
import torch
import numpy as np

from rl.env import TicTacToeEnv
from rl.dqn import DQNAgent, get_device
from rl.minimax import MinimaxAgent


def play_cli_game(agent_path: str = "models/dqn_tictactoe.pt", human_turn: str = "X"):
    device = get_device()
    agent = DQNAgent(device=device)
    try:
        agent.load(agent_path)
        print(f"Loaded trained DQN agent from {agent_path}")
    except Exception as e:
        print(f"Could not load checkpoint ({e}). Using untrained agent.")

    env = TicTacToeEnv()
    env.reset()
    human_player = 1 if human_turn.upper() == "X" else -1

    print("\n--- Tic-Tac-Toe: Human vs. RL Agent ---")
    print("Cells are numbered 0 through 8:")
    print(" 0 | 1 | 2 \n---+---+---\n 3 | 4 | 5 \n---+---+---\n 6 | 7 | 8 \n")

    while not env.done:
        print(env.render())
        print()
        valid_actions = env.get_valid_actions()

        if env.current_player == human_player:
            while True:
                try:
                    move = int(input(f"Your move ({valid_actions}): "))
                    if move in valid_actions:
                        break
                    print("Invalid move, try again.")
                except ValueError:
                    print("Please enter a valid integer.")
            env.step(move)
        else:
            print("RL Agent is thinking...")
            canonical = env.get_canonical_state()
            action = agent.select_action(canonical, valid_actions, greedy=True)
            print(f"RL Agent chooses cell {action}")
            env.step(action)

    print("\nFinal Board:")
    print(env.render())
    winner = env.winner
    if winner == human_player:
        print("\n🎉 You win!")
    elif winner == 0:
        print("\n🤝 Draw game!")
    else:
        print("\n🤖 RL Agent wins!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate or play against trained Tic-Tac-Toe RL Agent.")
    parser.add_argument("--play", action="store_true", help="Play interactively in the terminal against RL Agent")
    parser.add_argument("--player", type=str, default="X", choices=["X", "O"], help="Play as X (first) or O (second)")
    parser.add_argument("--model", type=str, default="models/dqn_tictactoe.pt", help="Path to model weights")
    args = parser.parse_args()

    if args.play:
        play_cli_game(agent_path=args.model, human_turn=args.player)
    else:
        print("Run with --play to challenge the RL agent in your terminal!")
