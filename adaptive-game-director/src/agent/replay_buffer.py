"""Experience Replay Buffer skeleton for off-policy DQN training."""

import random
from collections import deque
from typing import Tuple, List, Any
import numpy as np


class ReplayBuffer:
    """Fixed-capacity buffer storing experience transitions (s, a, r, s', done).

    Decouples data collection from gradient optimization and breaks temporal
    correlation between consecutive gameplay transitions.
    """

    def __init__(self, capacity: int = 50000) -> None:
        """
        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends a new transition to the buffer."""
        self.buffer.append((
            np.array(state, copy=True, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, copy=True, dtype=np.float32),
            bool(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniformly samples a mini-batch of transitions for training.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class SumTree:
    """Binary sum-tree for O(log n) proportional priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write_ptr = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Any) -> None:
        idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay — samples transitions proportional to TD-error.

    Transitions that surprise the network (high TD-error) are replayed more often,
    dramatically accelerating learning on critical combat events (kills, deaths, dodges).
    """

    def __init__(self, capacity: int = 50000, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001) -> None:
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha        # Priority exponent: 0 = uniform, 1 = full prioritization
        self.beta = beta          # Importance sampling exponent (annealed to 1.0)
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.epsilon = 1e-6       # Small constant to prevent zero-priority

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Adds transition with maximum priority (will be properly prioritized after first training)."""
        transition = (
            np.array(state, copy=True, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, copy=True, dtype=np.float32),
            bool(done),
        )
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """Samples a prioritized mini-batch.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, importance_weights, tree_indices).
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices: List[int] = []
        priorities: List[float] = []

        segment = self.tree.total / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = random.uniform(lo, hi)
            idx, priority, data = self.tree.get(s)
            if data is None:
                # Fallback: resample from beginning
                s = random.uniform(0, self.tree.total * 0.5)
                idx, priority, data = self.tree.get(s)
                if data is None:
                    continue

            indices.append(idx)
            priorities.append(priority)
            st, a, r, ns, d = data
            states.append(st)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        if len(states) == 0:
            # Extreme fallback
            return (np.zeros((1, 14), dtype=np.float32), np.zeros(1, dtype=np.int64),
                    np.zeros(1, dtype=np.float32), np.zeros((1, 14), dtype=np.float32),
                    np.zeros(1, dtype=np.bool_), np.ones(1, dtype=np.float32), [0])

        # Importance sampling weights for unbiased gradient updates
        sampling_probs = np.array(priorities, dtype=np.float64) / (self.tree.total + 1e-8)
        sampling_probs = np.clip(sampling_probs, 1e-8, 1.0)
        is_weights = (self.tree.size * sampling_probs) ** (-self.beta)
        is_weights /= is_weights.max()  # Normalize to [0, 1]

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            np.array(is_weights, dtype=np.float32),
            indices,
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """Updates priorities based on TD-errors from the latest training step."""
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.size
