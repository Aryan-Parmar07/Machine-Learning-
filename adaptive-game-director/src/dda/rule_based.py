"""Rule-based Dynamic Difficulty Adjustment (DDA) Controller.

This serves as our baseline heuristic AI Game Director. It monitors key player
state indicators (health, kill rate, ammo) and selects adjustments using deterministic
threshold rules designed to keep the player in a target flow channel.
"""

from typing import Optional, Union, Dict, Any
import numpy as np

from src.dda.difficulty import DDAAction, DifficultyState
from src.player_model.features import RawTelemetryFrame, DerivedMetrics


class RuleBasedDDAController:
    """Heuristic DDA controller providing baseline comparison against RL agents.

    Rules evaluate whether the player is in Distress (frustrated), Comfort (flow),
    or Dominance (bored):
    - Low Health (< 30) or High Death Rate -> Decrease difficulty / Increase resources
    - Low Ammo (< 15) -> Increase resources
    - High Health (> 80) and High Kill Rate -> Increase difficulty / Increase enemies
    - Otherwise -> No change
    """

    def __init__(
        self,
        target_health_low: float = 30.0,
        target_health_high: float = 80.0,
        low_ammo_threshold: float = 15.0,
        high_kill_rate_threshold: float = 0.5,
    ) -> None:
        self.target_health_low = target_health_low
        self.target_health_high = target_health_high
        self.low_ammo_threshold = low_ammo_threshold
        self.high_kill_rate_threshold = high_kill_rate_threshold

    def select_action_from_telemetry(
        self,
        raw: RawTelemetryFrame,
        derived: DerivedMetrics,
        current_state: DifficultyState,
    ) -> DDAAction:
        """Selects a DDA action directly from raw and derived telemetry frames.

        Args:
            raw: Latest raw telemetry frame.
            derived: Latest derived metrics frame.
            current_state: Current DifficultyState parameters.

        Returns:
            DDAAction decision.
        """
        # -------------------------------------------------------------
        # TIER 1: CRITICAL DISTRESS PROTECTION (HIGHEST PRIORITY)
        # If player is struggling or dying (HP < 35), protect them immediately.
        # NEVER escalate challenge when the player is hurting.
        # Priority 1: Ammo Depletion Protection (Cannot defend without ammo)
        if raw.ammo <= 8.0:
            return DDAAction.INCREASE_RESOURCES

        # Priority 2: Critical Health Distress (<35 HP)
        if raw.health < self.target_health_low or raw.health < 35.0:
            return DDAAction.INCREASE_RESOURCES

        # -------------------------------------------------------------
        # TIER 2: FLOW CHANNEL [35, 75 HP] (OPTIMAL TENSION)
        # Player is actively fighting in the sweet spot. Maintain flow.
        # -------------------------------------------------------------
        if 35.0 <= raw.health <= self.target_health_high:
            if raw.enemy_count <= 1:
                return DDAAction.INCREASE_ENEMY_SPAWN
            return DDAAction.NO_CHANGE

        # -------------------------------------------------------------
        # TIER 3: DOMINANCE / BOREDOM ESCALATION (ONLY IF HP > 75)
        # Player is comfortable at high health and eliminating threats.
        # -------------------------------------------------------------
        if raw.health > self.target_health_high:
            if raw.enemy_count < 3:
                return DDAAction.INCREASE_ENEMY_SPAWN
            elif derived.player_skill_estimate > 0.65 or derived.kill_rate > 0.30:
                return DDAAction.INCREASE_DIFFICULTY

        return DDAAction.NO_CHANGE

    def select_action_from_vector(self, feature_vector: np.ndarray) -> DDAAction:
        """Selects a DDA action from the normalized 15-dimensional observation vector.

        Index mapping matches FeatureExtractor:
        [0]: normalized health (health / 200)
        [2]: normalized ammo (ammo / 200)
        [5]: normalized kill rate
        """
        norm_health = feature_vector[0] * 200.0  # de-normalize to ~health
        norm_ammo = feature_vector[2] * 200.0    # de-normalize to ~ammo
        kill_rate = feature_vector[5] * 2.0      # de-normalize to ~kill rate

        if norm_health < self.target_health_low:
            return DDAAction.DECREASE_DIFFICULTY
        if norm_ammo < self.low_ammo_threshold:
            return DDAAction.INCREASE_RESOURCES
        if norm_health > self.target_health_high and kill_rate > self.high_kill_rate_threshold:
            return DDAAction.INCREASE_DIFFICULTY

        return DDAAction.NO_CHANGE
