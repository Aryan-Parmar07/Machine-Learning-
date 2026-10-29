"""Player model feature extraction and telemetry transformation.

This module is responsible for:
1. Defining the schema of raw telemetry captured directly from ViZDoom.
2. Deriving gameplay metrics not directly exposed by the ViZDoom engine.
3. Converting telemetry sequences into fixed-length normalized feature vectors
   ready for the RL DDA Director Agent.

CRITICAL DISTINCTION:
- DIRECT VARIABLES: Provided natively by the ViZDoom game state (e.g., HEALTH, KILLCOUNT).
- DERIVED METRICS: Computed algorithmically over time (e.g., kill_rate, accuracy, aggression).
"""

from dataclasses import dataclass, asdict
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# Standardized columns recorded in data/telemetry/*.csv
TELEMETRY_COLUMNS: List[str] = [
    # Metadata
    "episode",
    "tic",
    "timestamp",
    "elapsed_time_sec",
    
    # Direct ViZDoom Engine Variables
    "health",
    "armor",
    "ammo",
    "killcount",
    "deathcount",
    "hitcount",
    "hits_taken",
    "damagecount",
    "damage_taken",
    "is_dead",
    "selected_weapon",
    "position_x",
    "position_y",
    "position_z",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "attack_ready",
    "current_doom_skill",
    "last_reward",
    
    # Level Entities (from state.objects)
    "enemy_count",
    "item_count",
    
    # Derived Metrics (Calculated)
    "survival_time_sec",
    "kill_rate",
    "death_rate",
    "damage_dealt_rate",
    "damage_received_rate",
    "shots_fired_estimate",
    "accuracy_estimate",
    "player_speed",
    "recent_performance_trend",
    "aggression_score",
    "player_skill_estimate",
]


@dataclass
class RawTelemetryFrame:
    """Snapshot of raw state variables extracted directly from ViZDoom."""

    episode: int
    tic: int
    timestamp: float
    health: float
    armor: float
    ammo: float
    killcount: float
    deathcount: float
    hitcount: float
    hits_taken: float
    damagecount: float
    damage_taken: float
    is_dead: bool
    selected_weapon: float
    pos_x: float
    pos_y: float
    pos_z: float
    vel_x: float
    vel_y: float
    vel_z: float
    attack_ready: bool
    current_doom_skill: int
    last_reward: float
    enemy_count: int
    item_count: int


@dataclass
class DerivedMetrics:
    """Calculated metrics derived across gameplay time or deltas."""

    elapsed_time_sec: float
    survival_time_sec: float
    kill_rate: float            # kills per second
    death_rate: float           # deaths per second
    damage_dealt_rate: float    # damage dealt per second
    damage_received_rate: float # damage taken per second
    shots_fired_estimate: int   # estimated shots fired
    accuracy_estimate: float    # hitcount / shots_fired [0.0, 1.0]
    player_speed: float         # euclidean velocity magnitude
    recent_performance_trend: float  # performance trend [-1.0, 1.0]
    aggression_score: float     # movement towards enemies & firing [0.0, 1.0]
    player_skill_estimate: float # normalized estimated skill [0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureExtractor:
    """Transforms raw telemetry data and history into normalized feature vectors.

    The RL Game Director agent requires fixed-dimension normalized observations
    representing the player's current status and performance trajectory.
    """

    def __init__(self, ticrate: int = 35, window_seconds: float = 10.0) -> None:
        """
        Args:
            ticrate: ViZDoom internal ticks per second (default: 35).
            window_seconds: Duration of sliding window for rolling stats.
        """
        self.ticrate = ticrate
        self.window_seconds = window_seconds
        self.window_tics = int(ticrate * window_seconds)

        # State tracking for delta and rate calculations
        self.reset()

    def reset(self) -> None:
        """Resets episode-specific state accumulators."""
        self._episode_start_tic: int = 0
        self._last_death_tic: int = 0
        self._prev_ammo: Optional[float] = None
        self._prev_weapon: Optional[float] = None
        self._total_shots_fired: int = 0
        self._history: List[Tuple[RawTelemetryFrame, DerivedMetrics]] = []

    def compute_derived_metrics(
        self,
        raw: RawTelemetryFrame,
        skill_estimate: float = 0.5,
    ) -> DerivedMetrics:
        """Computes all derived metrics for the given raw frame.

        Args:
            raw: Current RawTelemetryFrame from ViZDoom.
            skill_estimate: Current skill score from SkillEstimator.

        Returns:
            DerivedMetrics instance with calculated values.
        """
        elapsed_time = raw.tic / float(self.ticrate)

        # Survival time tracks duration since last death or episode start
        if raw.is_dead:
            self._last_death_tic = raw.tic
        survival_time = (raw.tic - self._last_death_tic) / float(self.ticrate)

        # Rates computed over total elapsed time
        time_denom = max(1.0, elapsed_time)
        kill_rate = raw.killcount / time_denom
        death_rate = raw.deathcount / time_denom
        damage_dealt_rate = raw.damagecount / time_denom
        damage_received_rate = raw.damage_taken / time_denom

        # Shot estimation: track drops in ammo when weapon didn't change
        if self._prev_ammo is not None and self._prev_weapon is not None:
            if raw.selected_weapon == self._prev_weapon and raw.ammo < self._prev_ammo:
                ammo_delta = int(self._prev_ammo - raw.ammo)
                self._total_shots_fired += max(1, ammo_delta)
        self._prev_ammo = raw.ammo
        self._prev_weapon = raw.selected_weapon

        # Accuracy estimate: hitcount vs estimated shots
        if self._total_shots_fired > 0:
            accuracy = min(1.0, raw.hitcount / float(self._total_shots_fired))
        else:
            accuracy = 0.0

        # Player speed (2D Euclidean)
        speed = math.sqrt(raw.vel_x**2 + raw.vel_y**2)

        # Performance trend over recent sliding window
        recent_trend = self._calculate_recent_trend(raw)

        # Aggression score heuristic: combination of speed and damage ratio
        damage_ratio = (raw.damagecount + 1.0) / (raw.damagecount + raw.damage_taken + 2.0)
        norm_speed = min(1.0, speed / 25.0)  # ~25 units/tic is fast running speed in Doom
        aggression = float(np.clip(0.6 * damage_ratio + 0.4 * norm_speed, 0.0, 1.0))

        return DerivedMetrics(
            elapsed_time_sec=elapsed_time,
            survival_time_sec=survival_time,
            kill_rate=kill_rate,
            death_rate=death_rate,
            damage_dealt_rate=damage_dealt_rate,
            damage_received_rate=damage_received_rate,
            shots_fired_estimate=self._total_shots_fired,
            accuracy_estimate=accuracy,
            player_speed=speed,
            recent_performance_trend=recent_trend,
            aggression_score=aggression,
            player_skill_estimate=skill_estimate,
        )

    def _calculate_recent_trend(self, raw: RawTelemetryFrame) -> float:
        """Calculates short-term health delta trend in [-1.0, 1.0]."""
        if not self._history:
            return 0.0

        cutoff_tic = raw.tic - self.window_tics
        # Find oldest entry within the window
        past_raw = None
        for past_r, _ in reversed(self._history):
            if past_r.tic <= cutoff_tic:
                past_raw = past_r
                break
        if past_raw is None:
            past_raw = self._history[0][0]

        delta_health = (raw.health - past_raw.health) / 100.0
        delta_kills = (raw.killcount - past_raw.killcount) / 5.0
        return float(np.clip(delta_health * 0.5 + delta_kills * 0.5, -1.0, 1.0))

    def update_history(self, raw: RawTelemetryFrame, derived: DerivedMetrics) -> None:
        """Appends frame to rolling history and prunes old entries."""
        self._history.append((raw, derived))
        # Keep maximum 2 * window_tics frames
        max_keep = 2 * self.window_tics
        if len(self._history) > max_keep:
            self._history = self._history[-max_keep:]

    def extract_normalized_state_vector(
        self,
        raw: RawTelemetryFrame,
        derived: DerivedMetrics,
    ) -> np.ndarray:
        """Converts raw and derived telemetry into a fixed-length normalized vector.

        Output Feature Vector (Dimension: 15):
        [0]  Normalized Health: [0, 1] (health / 100 clamped)
        [1]  Normalized Armor: [0, 1] (armor / 100 clamped)
        [2]  Normalized Ammo: [0, 1] (ammo / 100 clamped)
        [3]  Normalized Kill Count: [0, 1] (killcount / 20 clamped)
        [4]  Normalized Death Count: [0, 1] (deathcount / 10 clamped)
        [5]  Kill Rate: [0, 1] (kills/sec / 2.0 clamped)
        [6]  Death Rate: [0, 1] (deaths/sec / 1.0 clamped)
        [7]  Damage Dealt Rate: [0, 1] (dmg/sec / 100.0 clamped)
        [8]  Damage Received Rate: [0, 1] (dmg/sec / 50.0 clamped)
        [9]  Accuracy Estimate: [0, 1]
        [10] Player Speed: [0, 1] (speed / 25.0 clamped)
        [11] Enemy Count: [0, 1] (enemy_count / 15.0 clamped)
        [12] Recent Performance Trend: [-1, 1]
        [13] Aggression Score: [0, 1]
        [14] Player Skill Estimate: [0, 1]

        Returns:
            np.ndarray of shape (15,), dtype=np.float32.
        """
        features = np.array(
            [
                np.clip(raw.health / 100.0, 0.0, 2.0) / 2.0,
                np.clip(raw.armor / 100.0, 0.0, 2.0) / 2.0,
                np.clip(raw.ammo / 100.0, 0.0, 2.0) / 2.0,
                np.clip(raw.killcount / 20.0, 0.0, 1.0),
                np.clip(raw.deathcount / 10.0, 0.0, 1.0),
                np.clip(derived.kill_rate / 2.0, 0.0, 1.0),
                np.clip(derived.death_rate / 1.0, 0.0, 1.0),
                np.clip(derived.damage_dealt_rate / 100.0, 0.0, 1.0),
                np.clip(derived.damage_received_rate / 50.0, 0.0, 1.0),
                np.clip(derived.accuracy_estimate, 0.0, 1.0),
                np.clip(derived.player_speed / 25.0, 0.0, 1.0),
                np.clip(raw.enemy_count / 15.0, 0.0, 1.0),
                np.clip(derived.recent_performance_trend, -1.0, 1.0),
                np.clip(derived.aggression_score, 0.0, 1.0),
                np.clip(derived.player_skill_estimate, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        return features
