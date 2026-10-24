"""
Deep Q-Network (DQN) Agent for Tic-Tac-Toe with Apple Silicon MPS acceleration.
Includes replay buffer, epsilon-greedy action selection, and policy export.
"""
from collections import deque
import random
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_device() -> torch.device:
    """Returns Apple Silicon MPS if available, otherwise CUDA or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def encode_state_np(board_state: np.ndarray) -> np.ndarray:
    """Vectorized state encoding to 18-element binary features."""
    my_marks = (board_state == 1).astype(np.float32)
    opp_marks = (board_state == -1).astype(np.float32)
    return np.concatenate([my_marks, opp_marks])


class QNetwork(nn.Module):
    """Deep Q-Network evaluating expected return for all 9 actions."""

    def __init__(self, input_dim: int = 18, hidden_dim: int = 128, output_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Experience replay buffer for off-policy Q-learning."""

    def __init__(self, capacity: int = 30000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_np = np.stack([encode_state_np(s) for s in states])
        ns_np = np.stack([encode_state_np(ns) for ns in next_states])

        s_t = torch.tensor(s_np, dtype=torch.float32, device=device)
        a_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        r_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        ns_t = torch.tensor(ns_np, dtype=torch.float32, device=device)
        d_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

        return s_t, a_t, r_t, ns_t, d_t

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN Agent with target network and Apple Silicon MPS support."""

    def __init__(
        self,
        lr: float = 7e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.03,
        epsilon_decay: float = 0.9996,
        target_update_freq: int = 250,
        buffer_capacity: int = 30000,
        batch_size: int = 128,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.train_steps = 0

    def select_action(
        self, state: np.ndarray, valid_actions: List[int], greedy: bool = False
    ) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")

        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        self.policy_net.eval()
        with torch.no_grad():
            feat = encode_state_np(state)
            s_tensor = torch.tensor(feat, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s_tensor).squeeze(0)

            # Mask invalid actions with large negative value
            mask = torch.full_like(q_values, -1e9)
            mask[valid_actions] = q_values[valid_actions]
            best_action = int(torch.argmax(mask).item())

        return best_action

    def update(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        self.policy_net.train()
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        current_q = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def save(self, filepath: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
            },
            filepath,
        )

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["policy_net"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.train_steps = checkpoint.get("train_steps", 0)
