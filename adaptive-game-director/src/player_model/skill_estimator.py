"""Player skill estimation module.

Provides algorithmic skill modeling based on ongoing combat performance,
accuracy, and survival rates.
"""

from enum import Enum
from typing import Optional
import numpy as np


class SkillTier(str, Enum):
    """Categorical skill tier for human player."""
    NOVICE = "NOVICE"
    INTERMEDIATE = "INTERMEDIATE"
    EXPERT = "EXPERT"


class SkillEstimator:
    """Estimates continuous player skill score in [0.0, 1.0].

    Uses an Exponential Moving Average (EMA) over composite combat performance:
    - Damage Ratio (dealt / (dealt + taken))
    - Kill / Death efficiency
    - Aim accuracy
    - Movement dynamism

    The resulting score informs the DDA Director when deciding whether the player
    is struggling (novice), comfortable, or breezing through (expert).
    """

    def __init__(
        self,
        initial_skill: float = 0.5,
        alpha: float = 0.05,
        novice_threshold: float = 0.35,
        expert_threshold: float = 0.75,
    ) -> None:
        """
        Args:
            initial_skill: Prior skill estimate (default: 0.5).
            alpha: EMA smoothing factor per update step (0 < alpha <= 1).
            novice_threshold: Cutoff score below which player is considered Novice.
            expert_threshold: Cutoff score above which player is considered Expert.
        """
        self.initial_skill = initial_skill
        self.alpha = alpha
        self.novice_threshold = novice_threshold
        self.expert_threshold = expert_threshold
        self._current_skill = initial_skill

    def reset(self) -> None:
        """Resets the skill score to the initial prior."""
        self._current_skill = self.initial_skill

    def update(
        self,
        health: float,
        damage_dealt: float,
        damage_taken: float,
        killcount: float,
        deathcount: float,
        accuracy: float,
    ) -> float:
        """Updates the skill estimate with new gameplay evidence.

        Returns:
            Updated continuous skill score in [0.0, 1.0].
        """
        # 1. Combat efficiency: damage dealt relative to taken
        total_dmg = damage_dealt + damage_taken
        if total_dmg > 0:
            combat_score = damage_dealt / (total_dmg + 1e-5)
        else:
            combat_score = 0.5

        # 2. Kill-to-death ratio score (clamped)
        kd_ratio = killcount / max(1.0, deathcount)
        kd_score = float(np.clip(kd_ratio / 3.0, 0.0, 1.0))

        # 3. Health maintenance score
        health_score = float(np.clip(health / 100.0, 0.0, 1.0))

        # 4. Composite performance snapshot
        instant_score = (
            0.40 * combat_score +
            0.30 * kd_score +
            0.15 * health_score +
            0.15 * accuracy
        )

        # Smooth update via EMA
        self._current_skill = (
            (1.0 - self.alpha) * self._current_skill + self.alpha * instant_score
        )
        self._current_skill = float(np.clip(self._current_skill, 0.0, 1.0))
        return self._current_skill

    @property
    def skill_score(self) -> float:
        """Current continuous skill score in [0.0, 1.0]."""
        return self._current_skill

    @property
    def tier(self) -> SkillTier:
        """Returns the categorical skill tier."""
        if self._current_skill < self.novice_threshold:
            return SkillTier.NOVICE
        elif self._current_skill > self.expert_threshold:
            return SkillTier.EXPERT
        return SkillTier.INTERMEDIATE
