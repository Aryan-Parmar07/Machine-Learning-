"""Gymnasium Environment for the RL Dynamic Difficulty Adjustment (DDA) Game Director.

ARCHITECTURE NOTE:
This environment does NOT control the player character.
- The HUMAN player (in interactive mode) or SIMULATED BOT (in training mode)
  controls the Doom marine at 35 FPS (Inner Game Loop).
- The RL AGENT acts as the high-level AI Game Director at a lower decision frequency
  (e.g., every 5 seconds / 175 tics) (Outer Director Loop).
- Observation space: 15-dimensional normalized player performance vector.
- Action space: Discrete(7) difficulty adjustments.
- Reward function: FlowStateReward (rewards optimal tension [35, 75 HP]; penalizes death & boredom).
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd

from src.dda.difficulty import DDAAction, DifficultyState, EngineActuator
from src.player_model.features import (
    FeatureExtractor,
    RawTelemetryFrame,
    DerivedMetrics,
)
from src.player_model.skill_estimator import SkillEstimator
from src.rewards.reward_functions import FlowStateReward, BaseRewardFunction
from scripts.play_human import resolve_scenario_path, count_level_entities
from scripts.collect_telemetry import SimulatedPlayerAgent


class ViZDoomGameDirectorEnv(gym.Env):
    """Gymnasium environment for training and evaluating an RL AI Game Director."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 35}

    def __init__(
        self,
        scenario: str = "defend_the_center.cfg",
        decision_interval_sec: float = 5.0,
        ticrate: int = 35,
        player_mode: str = "bot",
        persona: str = "intermediate",
        initial_skill: int = 3,
        reward_fn: Optional[BaseRewardFunction] = None,
        render_mode: Optional[str] = None,
        sound_enabled: bool = False,
        render_fps: int = 35,
    ) -> None:
        """
        Args:
            scenario: ViZDoom scenario filename (e.g. defend_the_center.cfg, deathmatch.cfg).
            decision_interval_sec: Seconds between RL director actions (default: 5.0).
            ticrate: ViZDoom internal ticks per second (default: 35).
            player_mode: 'bot' (automated simulated combatant) or 'human' (interactive spectator).
            persona: Simulated player style ('novice', 'intermediate', 'aggressive', 'random').
            initial_skill: Starting Doom engine skill (1 to 5).
            reward_fn: Reward function instance (defaults to FlowStateReward).
            render_mode: Optional render mode ('human' or None).
            sound_enabled: Enable game audio during live visual rendering. Default: True.
            render_fps: Target playback speed when rendered visually. Default: 35.
        """
        super().__init__()
        self.scenario_name = scenario
        self.scenario_path = resolve_scenario_path(scenario)
        self.decision_interval_sec = decision_interval_sec
        self.ticrate = ticrate
        self.decision_interval_tics = int(decision_interval_sec * ticrate)
        self.player_mode = player_mode
        self.persona = persona
        self.initial_skill = initial_skill
        self.render_mode = render_mode
        self.sound_enabled = sound_enabled
        self.render_fps = render_fps

        # Reward model
        self.reward_fn = reward_fn or FlowStateReward()

        # Observation Space: 15-dimensional normalized player performance vector
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32,
        )

        # Action Space: 7 discrete difficulty adjustments
        # 0: No change, 1: +Difficulty, 2: -Difficulty, 3: +Enemies, 4: -Enemies, 5: +Resources, 6: -Resources
        self.action_space = spaces.Discrete(7)

        # Internal state trackers
        self.difficulty_state = DifficultyState(doom_skill=initial_skill)
        self.feature_extractor = FeatureExtractor(ticrate=ticrate)
        self.skill_estimator = SkillEstimator()
        self._game: Optional[vzd.DoomGame] = None
        self._actuator: Optional[EngineActuator] = None
        self._bot_agent: Optional[SimulatedPlayerAgent] = None
        self._available_buttons: List[vzd.Button] = []
        self._current_tic = 0

    def _init_game(self) -> None:
        """Initializes the ViZDoom engine instance."""
        if self._game is not None:
            return

        self._game = vzd.DoomGame()
        self._game.load_config(self.scenario_path)

        show_window = (self.render_mode == "human" or self.player_mode == "human")
        self._game.set_window_visible(show_window)
        self._game.set_doom_skill(self.initial_skill)

        if self.player_mode == "human":
            self._game.set_mode(vzd.Mode.SPECTATOR)
            self._game.add_game_args("+freelook 1")
            self._game.set_sound_enabled(True)
            self._game.set_render_hud(True)
            self._game.set_render_crosshair(True)
        else:
            self._game.set_mode(vzd.Mode.PLAYER)
            self._game.set_sound_enabled(show_window and self.sound_enabled)
            self._game.set_render_hud(show_window)
            if show_window:
                self._game.set_render_crosshair(True)

        self._game.set_objects_info_enabled(True)
        self._game.set_sectors_info_enabled(False)

        # Configure direct game variables
        self._game.clear_available_game_variables()
        game_vars = [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.DEATHCOUNT,
            vzd.GameVariable.HITCOUNT,
            vzd.GameVariable.HITS_TAKEN,
            vzd.GameVariable.DAMAGECOUNT,
            vzd.GameVariable.DAMAGE_TAKEN,
            vzd.GameVariable.DEAD,
            vzd.GameVariable.SELECTED_WEAPON,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.POSITION_Z,
            vzd.GameVariable.VELOCITY_X,
            vzd.GameVariable.VELOCITY_Y,
            vzd.GameVariable.VELOCITY_Z,
            vzd.GameVariable.ATTACK_READY,
            vzd.GameVariable.ANGLE,
        ]
        for gv in game_vars:
            self._game.add_available_game_variable(gv)

        self._game.init()
        try:
            self._game.send_game_command("sv_cheats 1")
        except Exception:
            pass
        self._actuator = EngineActuator(self._game)
        self._available_buttons = self._game.get_available_buttons()

        if self.player_mode == "bot":
            self._bot_agent = SimulatedPlayerAgent(self.persona, self._available_buttons)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the game episode and returns the initial player observation vector."""
        super().reset(seed=seed)
        self._init_game()

        # Allow overriding persona on reset
        if options and "persona" in options and self.player_mode == "bot":
            self.persona = options["persona"]
            self._bot_agent = SimulatedPlayerAgent(self.persona, self._available_buttons)

        self.difficulty_state = DifficultyState(doom_skill=self.initial_skill)
        self.feature_extractor.reset()
        self.skill_estimator.reset()
        self._current_tic = 0

        self._game.new_episode()
        try:
            self._game.send_game_command("sv_cheats 1")
        except Exception:
            pass

        # Step forward a few tics to stabilize initial state
        initial_raw, initial_derived = self._advance_simulation(advance_tics=10)
        obs = self.feature_extractor.extract_normalized_state_vector(initial_raw, initial_derived)

        info = {
            "difficulty": self.difficulty_state.to_dict(),
            "health": initial_raw.health,
            "ammo": initial_raw.ammo,
            "skill_estimate": self.skill_estimator.skill_score,
            "skill_tier": self.skill_estimator.tier.value,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one Director decision step.

        1. Applies discrete DDA action (e.g. spawns items/enemies) via EngineActuator.
        2. Simulates inner game loop for `decision_interval_tics` (e.g. 5 seconds) while player fights.
        3. Computes FlowStateReward and extracts 15-dim normalized observation vector.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        dda_action = DDAAction(action)

        # 1. Engine Actuation with safety guardrail
        curr_state = self._game.get_state()
        curr_hp = float(curr_state.game_variables[0]) if (curr_state and len(curr_state.game_variables) > 0) else 100.0
        actuation_msg = self._actuator.execute(dda_action, self.difficulty_state, player_health=curr_hp)

        # 2. Advance inner loop for decision_interval_tics
        raw_frame, derived_metrics = self._advance_simulation(advance_tics=self.decision_interval_tics)

        # 3. Compute Flow State Reward
        reward = self.reward_fn.calculate_reward(
            raw=raw_frame,
            derived=derived_metrics,
            step_duration_sec=self.decision_interval_sec,
        )

        # Heavy penalty if director tried to escalate challenge on a player in distress (<35 HP)
        if curr_hp < 35.0 and dda_action in (DDAAction.INCREASE_ENEMY_SPAWN, DDAAction.INCREASE_DIFFICULTY):
            reward -= 2.0

        # 4. Extract 15-dimensional observation vector
        obs = self.feature_extractor.extract_normalized_state_vector(raw_frame, derived_metrics)

        # 5. Termination checks
        terminated = self._game.is_episode_finished() or raw_frame.is_dead
        truncated = False

        info = {
            "applied_action": dda_action.name,
            "actuation_msg": actuation_msg,
            "difficulty": self.difficulty_state.to_dict(),
            "health": raw_frame.health,
            "ammo": raw_frame.ammo,
            "killcount": raw_frame.killcount,
            "skill_estimate": derived_metrics.player_skill_estimate,
            "skill_tier": self.skill_estimator.tier.value,
            "elapsed_time_sec": derived_metrics.elapsed_time_sec,
        }

        return obs, float(reward), terminated, truncated, info

    def _advance_simulation(self, advance_tics: int) -> Tuple[RawTelemetryFrame, DerivedMetrics]:
        """Runs the inner combat loop (human or bot) for a specified number of tics."""
        for _ in range(advance_tics):
            if self._game.is_episode_finished():
                break

            state = self._game.get_state()
            if state is None:
                break

            g_vars = state.game_variables

            if self.player_mode == "human":
                self._game.advance_action()
            else:
                action_vec = self._bot_agent.compute_action(state, g_vars, Path(self.scenario_path).name)
                self._game.make_action(action_vec)
                if self.render_mode == "human" and self.render_fps > 0:
                    time.sleep(1.0 / self.render_fps)

            self._current_tic += 1

        state = self._game.get_state()
        if state is None:
            # Fallback frame if episode closed
            raw = RawTelemetryFrame(
                episode=1, tic=self._current_tic, timestamp=0.0, health=0.0, armor=0.0, ammo=0.0,
                killcount=0.0, deathcount=1.0, hitcount=0.0, hits_taken=1.0, damagecount=0.0,
                damage_taken=100.0, is_dead=True, selected_weapon=0.0, pos_x=0.0, pos_y=0.0,
                pos_z=0.0, vel_x=0.0, vel_y=0.0, vel_z=0.0, attack_ready=False,
                current_doom_skill=self.difficulty_state.doom_skill, last_reward=0.0,
                enemy_count=0, item_count=0,
            )
            derived = self.feature_extractor.compute_derived_metrics(raw, self.skill_estimator.skill_score)
            return raw, derived

        g_vars = state.game_variables
        health = float(g_vars[0]) if len(g_vars) > 0 else 0.0
        armor = float(g_vars[1]) if len(g_vars) > 1 else 0.0
        ammo = float(g_vars[2]) if len(g_vars) > 2 else 0.0
        killcount = float(g_vars[3]) if len(g_vars) > 3 else 0.0
        deathcount = float(g_vars[4]) if len(g_vars) > 4 else 0.0
        hitcount = float(g_vars[5]) if len(g_vars) > 5 else 0.0
        hits_taken = float(g_vars[6]) if len(g_vars) > 6 else 0.0
        damagecount = float(g_vars[7]) if len(g_vars) > 7 else 0.0
        damage_taken = float(g_vars[8]) if len(g_vars) > 8 else 0.0
        is_dead = bool(g_vars[9] > 0.0) if len(g_vars) > 9 else False
        selected_weapon = float(g_vars[10]) if len(g_vars) > 10 else 0.0
        pos_x = float(g_vars[11]) if len(g_vars) > 11 else 0.0
        pos_y = float(g_vars[12]) if len(g_vars) > 12 else 0.0
        pos_z = float(g_vars[13]) if len(g_vars) > 13 else 0.0
        vel_x = float(g_vars[14]) if len(g_vars) > 14 else 0.0
        vel_y = float(g_vars[15]) if len(g_vars) > 15 else 0.0
        vel_z = float(g_vars[16]) if len(g_vars) > 16 else 0.0
        attack_ready = bool(g_vars[17] > 0.0) if len(g_vars) > 17 else False

        enemy_count, item_count = count_level_entities(state)

        raw = RawTelemetryFrame(
            episode=1,
            tic=state.number,
            timestamp=0.0,
            health=health,
            armor=armor,
            ammo=ammo,
            killcount=killcount,
            deathcount=deathcount,
            hitcount=hitcount,
            hits_taken=hits_taken,
            damagecount=damagecount,
            damage_taken=damage_taken,
            is_dead=is_dead,
            selected_weapon=selected_weapon,
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            vel_x=vel_x,
            vel_y=vel_y,
            vel_z=vel_z,
            attack_ready=attack_ready,
            current_doom_skill=self.difficulty_state.doom_skill,
            last_reward=self._game.get_last_reward(),
            enemy_count=enemy_count,
            item_count=item_count,
        )

        derived = self.feature_extractor.compute_derived_metrics(raw, self.skill_estimator.skill_score)
        self.skill_estimator.update(
            health=health,
            damage_dealt=damagecount,
            damage_taken=damage_taken,
            killcount=killcount,
            deathcount=deathcount,
            accuracy=derived.accuracy_estimate,
        )
        derived.player_skill_estimate = self.skill_estimator.skill_score
        self.feature_extractor.update_history(raw, derived)

        return raw, derived

    def render(self) -> None:
        """Renders the game window if in human/visual mode."""
        pass

    def close(self) -> None:
        """Safely closes the ViZDoom engine instance."""
        if self._game is not None:
            try:
                self._game.close()
            except Exception:
                pass
            self._game = None
            self._actuator = None
