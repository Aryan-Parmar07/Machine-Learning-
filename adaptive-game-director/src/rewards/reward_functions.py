"""Reward functions for the AI Game Director RL agent.

The objective of an AI Game Director differs fundamentally from standard game-playing agents:
- A game-playing agent maximizes kills or wins.
- A Game Director agent maximizes player engagement and maintains the psychological
  "Flow State" (balancing challenge and skill to avoid boredom and frustration).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from src.player_model.features import RawTelemetryFrame, DerivedMetrics


class BaseRewardFunction(ABC):
    """Abstract base class for Game Director reward formulations."""

    @abstractmethod
    def calculate_reward(
        self,
        raw: RawTelemetryFrame,
        derived: DerivedMetrics,
        step_duration_sec: float,
    ) -> float:
        """Computes scalar reward for the Game Director at the current decision step."""
        pass


class FlowStateReward(BaseRewardFunction):
    """Flow State reward function balancing tension, challenge, and survival.

    Rewards:
    + Positive reward when player health is kept in the optimal tension zone [35, 75].
    + Positive reward for active combat engagement (balanced kill rate and action).
    - Penalty for player death (indicates excessive frustration / spike difficulty).
    - Penalty for boredom (player at 100% health for prolonged duration without challenge).
    """

    def __init__(
        self,
        target_health_min: float = 35.0,
        target_health_max: float = 75.0,
        death_penalty: float = 10.0,
        boredom_penalty: float = 1.0,
    ) -> None:
        self.target_health_min = target_health_min
        self.target_health_max = target_health_max
        self.death_penalty = death_penalty
        self.boredom_penalty = boredom_penalty

    def calculate_reward(
        self,
        raw: RawTelemetryFrame,
        derived: DerivedMetrics,
        step_duration_sec: float,
    ) -> float:
        """Calculates scalar flow-state reward for the director.

        Formula:
            - If dead: -death_penalty (-10.0)
            - If HP < target_health_min (35 HP): Negative distress penalty [-1.0, -4.0]
            - If in Flow State [35, 75 HP]: Positive reward [0.6, 1.0] + combat engagement bonus (+0.5)
            - If HP > 75 HP: Comfort zone [0.0, 0.6] with boredom penalty if idle
        """
        # 1. Death Penalty (Severe failure of game director pacing)
        if raw.is_dead:
            return -self.death_penalty

        # 2. Severe Distress Penalty: Player is critically injured / overwhelmed (<35 HP)
        # Prevents agent from farming combat reward by keeping player near death
        if raw.health < self.target_health_min:
            distress_severity = (self.target_health_min - max(0.0, raw.health)) / self.target_health_min
            return float(-1.0 - 3.0 * distress_severity)

        # 3. Health Balance in Flow State / Comfort
        target_mid = (self.target_health_min + self.target_health_max) / 2.0  # 55.0 HP
        health_dist = abs(raw.health - target_mid)
        norm_dist = np.clip(health_dist / 50.0, 0.0, 1.0)
        r_health = 1.0 - norm_dist  # [0.0, 1.0]

        # 4. Combat Engagement: Player is actively fighting (only rewarded if NOT in distress)
        has_combat = (derived.damage_dealt_rate > 0.0 or raw.killcount > 0)
        r_engagement = 0.5 if has_combat else 0.0

        # 5. Boredom Penalty: Health is full and no damage taken
        p_boredom = 0.0
        if raw.health >= 99.0 and derived.damage_received_rate < 0.1:
            p_boredom = self.boredom_penalty * 0.5

        total_reward = float(r_health + r_engagement - p_boredom)
        return total_reward
