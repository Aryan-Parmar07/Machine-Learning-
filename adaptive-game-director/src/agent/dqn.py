"""Deep Q-Network (DQN) Agent Skeleton.

ARCHITECTURE NOTE:
This RL agent serves as the AI Game Director. It observes the human player's
aggregated state vector and periodically selects difficulty adjustment actions.
Training will be implemented in future milestones once human gameplay telemetry
and baseline performance data are verified.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from src.agent.network import DDA_QNetwork, TORCH_AVAILABLE
from src.agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

if TORCH_AVAILABLE:
    import torch
    import torch.optim as optim
    import torch.nn.functional as F


class DQNAgent:
    """Dueling Double DQN Agent acting as the Adaptive Game Director.

    Features:
      - Dueling Network Architecture: Decouples Flow state value V(s) from action advantage A(s, a)
      - Double DQN: Eliminates positive maximization bias in Q-value estimation
      - Prioritized Experience Replay: Prioritizes learning from critical distress/rescue moments
      - Soft Polyak Target Updates (tau=0.005): Smoother convergence without abrupt policy shifts
    """

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 7,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        lr: Optional[float] = None,
        tau: float = 0.005,
    ) -> None:
        if lr is not None:
            learning_rate = lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        # Prioritized Replay Buffer prioritizing critical pacing crises & recoveries
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_capacity, alpha=0.6, beta=0.4)
        self.train_steps: int = 0

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DDA_QNetwork(state_dim, action_dim).to(self.device)
            self.target_network = DDA_QNetwork(state_dim, action_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        else:
            self.device = None
            self.q_network = None
            self.target_network = None
            self.optimizer = None

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Selects an action using epsilon-greedy exploration.

        Args:
            state: Normalized player state vector (15-dim).
            epsilon: Probability of choosing a random exploratory action [0.0, 1.0].

        Returns:
            Discrete action index (0 to 6).
        """
        if np.random.rand() < epsilon or not TORCH_AVAILABLE or self.q_network is None:
            return int(np.random.randint(0, self.action_dim))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores a gameplay transition in the prioritized replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """Performs one Double DQN update with Prioritized Replay and soft target tracking."""
        if len(self.replay_buffer) < self.batch_size or not TORCH_AVAILABLE:
            return None

        # Prioritized transition sampling
        states, actions, rewards, next_states, dones, is_weights, tree_indices = \
            self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        is_weights_t = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # Current predicted Q-values Q(s, a)
        current_q = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # DOUBLE DQN: Online network chooses best action, target network evaluates it
            best_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_network(next_states_t).gather(1, best_actions)
            target_q = rewards_t + (self.gamma * next_q * (~dones_t).float())

        # TD-errors for priority update
        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()

        # Importance-sampling weighted Huber loss
        element_losses = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (element_losses * is_weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities in SumTree
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        # Soft Polyak target network update
        self.train_steps += 1
        for target_param, online_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copies policy network weights to the target network."""
        if TORCH_AVAILABLE and self.q_network and self.target_network:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save_checkpoint(self, filepath: Path) -> None:
        """Saves agent weights and training state to disk."""
        if not TORCH_AVAILABLE or self.q_network is None:
            return
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
        }, filepath)

    def load_checkpoint(self, filepath: Path) -> None:
        """Loads agent weights from disk."""
        if not TORCH_AVAILABLE or self.q_network is None:
            return
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = checkpoint.get("train_steps", 0)
