"""
Training pipeline for DQN Tic-Tac-Toe Agent using Apple Silicon MPS acceleration.
Mathematically consistent MDP transitions for turn-based zero-sum self-play.
"""
import json
import os
import random
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl.dqn import DQNAgent, get_device
from rl.env import TicTacToeEnv
from rl.minimax import MinimaxAgent


def evaluate_agent(
    agent: DQNAgent,
    minimax_agent: MinimaxAgent,
    opponent_type: str = "random",
    num_games: int = 150,
) -> Dict[str, float]:
    """
    Evaluates agent against specified opponent over num_games (half as X, half as O).
    Returns win, draw, and loss percentages.
    """
    env = TicTacToeEnv()
    wins, draws, losses = 0, 0, 0

    for game in range(num_games):
        env.reset()
        agent_player = 1 if (game % 2 == 0) else -1

        while not env.done:
            current_player = env.current_player
            valid_actions = env.get_valid_actions()

            if current_player == agent_player:
                action = agent.select_action(
                    env.get_canonical_state(), valid_actions, greedy=True
                )
            else:
                if opponent_type == "random":
                    action = random.choice(valid_actions)
                else:
                    action = minimax_agent.select_action(env.board.copy(), current_player)

            env.step(action)

        winner = env.winner
        if winner == agent_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    return {
        "win_rate": wins / num_games,
        "draw_rate": draws / num_games,
        "loss_rate": losses / num_games,
        "undefeated_rate": (wins + draws) / num_games,
    }


def export_policy_table(agent: DQNAgent, output_path: str):
    """
    Traverses all reachable legal Tic-Tac-Toe states and maps each state string
    to the optimal action chosen by the trained DQN.
    State format: string of 9 chars ('0', '1', '2' where 0=empty, 1=X, 2=O).
    """
    policy = {}
    visited = set()

    def dfs(board: np.ndarray, current_player: int):
        board_key = "".join("0" if v == 0 else ("1" if v == 1 else "2") for v in board)
        if board_key in visited:
            return
        visited.add(board_key)

        # Check terminal
        env = TicTacToeEnv()
        env.board = board.copy()
        if env.check_winner() is not None:
            return

        valid_actions = env.get_valid_actions()
        if not valid_actions:
            return

        canonical_state = (board * current_player).astype(np.float32)
        best_action = agent.select_action(canonical_state, valid_actions, greedy=True)
        policy[board_key] = best_action

        for action in valid_actions:
            next_board = board.copy()
            next_board[action] = current_player
            dfs(next_board, -current_player)

    initial_board = np.zeros(9, dtype=np.int8)
    dfs(initial_board, 1)

    with open(output_path, "w") as f:
        json.dump(policy, f)
    print(f"Exported policy table ({len(policy)} states) to {output_path}")


def main():
    device = get_device()
    print(f"🚀 Training DQN Agent on Device: {device} (MPS is {'ACTIVE' if device.type == 'mps' else 'INACTIVE'})")

    env = TicTacToeEnv()
    agent = DQNAgent(
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.02,
        epsilon_decay=0.9993,
        target_update_freq=200,
        buffer_capacity=40000,
        batch_size=128,
        device=device,
    )
    minimax_teacher = MinimaxAgent(epsilon_error=0.20)
    minimax_eval = MinimaxAgent(epsilon_error=0.0)

    num_episodes = 6000
    eval_freq = 400

    episodes_history = []
    losses_history = []
    epsilons_history = []
    vs_random_win = []
    vs_random_undefeated = []
    vs_minimax_draw_or_win = []

    start_time = time.time()
    recent_losses = []

    for episode in range(1, num_episodes + 1):
        env.reset()

        # Decide curriculum opponent
        p = random.random()
        if episode < 1500:
            opp_type = "random" if p < 0.75 else "self"
        elif episode < 3500:
            opp_type = "random" if p < 0.35 else ("self" if p < 0.70 else "minimax")
        else:
            opp_type = "minimax" if p < 0.45 else ("self" if p < 0.80 else "random")

        # Track state-action history for each player (1 and -1)
        last_state = {1: None, -1: None}
        last_action = {1: None, -1: None}

        # In non-self-play, assign which side agent plays
        agent_side = 1 if random.random() < 0.5 else -1

        while not env.done:
            player = env.current_player
            valid_actions = env.get_valid_actions()
            canonical = env.get_canonical_state()

            is_agent_turn = (opp_type == "self") or (player == agent_side)

            if is_agent_turn:
                action = agent.select_action(canonical, valid_actions, greedy=False)
            else:
                if opp_type == "random":
                    action = random.choice(valid_actions)
                else:
                    action = minimax_teacher.select_action(env.board.copy(), player)

            # Record intermediate transition for current player if they made a previous move
            if is_agent_turn:
                if last_state[player] is not None:
                    agent.memory.push(last_state[player], last_action[player], 0.0, canonical, False)
                last_state[player] = canonical
                last_action[player] = action

            env.step(action)

        # Game finished: record terminal transitions
        winner = env.winner
        dummy_state = np.zeros(9, dtype=np.float32)

        for p_id in [1, -1]:
            is_agent = (opp_type == "self") or (p_id == agent_side)
            if is_agent and last_state[p_id] is not None:
                if winner == p_id:
                    reward = 1.0
                elif winner == 0:
                    reward = 0.5
                else:
                    reward = -1.0
                agent.memory.push(last_state[p_id], last_action[p_id], reward, dummy_state, True)

        # DQN Training Step (2 gradient steps per episode)
        for _ in range(2):
            loss = agent.update()
            if loss is not None:
                recent_losses.append(loss)

        # Periodic Evaluation
        if episode % eval_freq == 0 or episode == 1:
            eval_rand = evaluate_agent(agent, minimax_eval, opponent_type="random", num_games=150)
            eval_mini = evaluate_agent(agent, minimax_eval, opponent_type="minimax", num_games=100)

            avg_loss = float(np.mean(recent_losses)) if recent_losses else 0.0
            recent_losses = []

            episodes_history.append(episode)
            losses_history.append(avg_loss)
            epsilons_history.append(agent.epsilon)
            vs_random_win.append(eval_rand["win_rate"] * 100)
            vs_random_undefeated.append(eval_rand["undefeated_rate"] * 100)
            vs_minimax_draw_or_win.append(eval_mini["undefeated_rate"] * 100)

            print(
                f"Ep {episode:4d}/{num_episodes} | "
                f"ε: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"vs Random: Win {eval_rand['win_rate']*100:.1f}%, Undefeated {eval_rand['undefeated_rate']*100:.1f}% | "
                f"vs Minimax: Undefeated {eval_mini['undefeated_rate']*100:.1f}%"
            )

    elapsed = time.time() - start_time
    print(f"\n✅ Training completed in {elapsed:.1f}s ({elapsed/num_episodes*1000:.2f} ms/ep)")

    # Save model
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    agent.save("models/dqn_tictactoe.pt")
    print("Saved PyTorch model to models/dqn_tictactoe.pt")

    # Export policy JSON for Web UI
    export_policy_table(agent, "trained_policy.json")

    # Generate Figures
    generate_plots(
        episodes_history,
        losses_history,
        epsilons_history,
        vs_random_win,
        vs_random_undefeated,
        vs_minimax_draw_or_win,
    )


def generate_plots(episodes, losses, epsilons, vs_rand_win, vs_rand_undefeated, vs_mini_undefeated):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Training Loss and Epsilon Decay
    fig, ax1 = plt.subplots(figsize=(9, 5))
    color = "#1f77b4"
    ax1.set_xlabel("Training Episodes", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Huber Loss", color=color, fontsize=12, fontweight="bold")
    line1 = ax1.plot(episodes, losses, color=color, linewidth=2, label="DQN Huber Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#ff7f0e"
    ax2.set_ylabel("Exploration Rate (ε)", color=color, fontsize=12, fontweight="bold")
    line2 = ax2.plot(episodes, epsilons, color=color, linestyle="--", linewidth=2, label="Epsilon (ε)")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True)
    plt.title("DQN Training Loss & Epsilon Decay (Apple Silicon MPS)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    loss_path = "assets/training_loss_epsilon.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"Saved plot: {loss_path}")

    # Figure 2: Evaluation Performance (Win & Undefeated Rates)
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, vs_rand_win, color="#2ca02c", linewidth=2.5, marker="o", markersize=4, label="Win Rate vs Random (%)")
    plt.plot(episodes, vs_rand_undefeated, color="#17becf", linewidth=2, linestyle="-.", label="Undefeated Rate vs Random (%)")
    plt.plot(episodes, vs_mini_undefeated, color="#d62728", linewidth=2.5, marker="s", markersize=4, label="Undefeated Rate vs Minimax (%)")

    plt.axhline(y=100, color="gray", linestyle=":", alpha=0.6)
    plt.xlabel("Training Episodes", fontsize=12, fontweight="bold")
    plt.ylabel("Success Rate (%)", fontsize=12, fontweight="bold")
    plt.title("RL Agent Performance vs. Random & Minimax Opponents", fontsize=14, fontweight="bold", pad=12)
    plt.ylim(-5, 105)
    plt.legend(loc="lower right", frameon=True, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    perf_path = "assets/eval_performance.png"
    plt.savefig(perf_path, dpi=200)
    plt.close()
    print(f"Saved plot: {perf_path}")


if __name__ == "__main__":
    main()
