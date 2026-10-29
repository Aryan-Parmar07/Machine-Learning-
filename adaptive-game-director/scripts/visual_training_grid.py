#!/usr/bin/env python3
"""Multi-Window Split-Screen Surveillance Grid for Parallel AI Training.

Renders a composite multi-tile dashboard (e.g. 2x2 or 2x3 matrix) using Pygame,
allowing you to watch 4 to 6 AI games running side-by-side in real-time while
the neural networks train and improve!

Supports Three Modes:
  1. `co_evolution` (Default): Trains BOTH the Combat Marine (aiming, shooting, dodging)
     and the AI Game Director (pacing, supply drops, flow state retention).
  2. `director_only`: Trains the Game Director while scripted bots fight across all windows.
  3. `combat_only`: Trains the Combat Marine while difficulty is held static.

Usage:
    # 4-Window Grid across diverse scenarios (defend_the_center, deadly_corridor, deathmatch, defend_the_line):
    python scripts/visual_training_grid.py --grid-size 4 --mode co_evolution

    # 4-Window Grid on a single scenario:
    python scripts/visual_training_grid.py --grid-size 4 --scenario defend_the_center.cfg

    # 6-Window Grid (2x3 Matrix):
    python scripts/visual_training_grid.py --grid-size 6 --episodes 20
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pygame
import torch
import vizdoom as vzd

from src.agent.combat_agent import CombatAgent, CombatStateExtractor, COMBAT_ACTIONS
from src.agent.dqn import DQNAgent
from src.dda.difficulty import DifficultyState, DDAAction, EngineActuator
from src.dda.overlay import DOOM_SKILL_META, SKILL_TIER_META
from src.player_model.features import (
    FeatureExtractor,
    RawTelemetryFrame,
    DerivedMetrics,
)
from src.player_model.skill_estimator import SkillEstimator
from src.rewards.reward_functions import FlowStateReward
from src.utils.logger import get_logger
from scripts.play_human import resolve_scenario_path, count_level_entities
from scripts.collect_telemetry import SimulatedPlayerAgent

logger = get_logger("VisualGrid")

DEFAULT_SCENARIOS = [
    "defend_the_center.cfg",
    "deadly_corridor.cfg",
    "deathmatch.cfg",
    "defend_the_line.cfg",
    "deadly_corridor.cfg",
    "defend_the_center.cfg",
]

CURRICULUM_STAGES = [
    {"scenario": "defend_the_center.cfg", "tier_name": "T1: CENTER", "threshold": 3.0, "desc": "Stationary 360-degree aiming & shooting"},
    {"scenario": "defend_the_line.cfg", "tier_name": "T2: LINE", "threshold": 3.5, "desc": "Linear corridor defense & projectile dodging"},
    {"scenario": "deadly_corridor.cfg", "tier_name": "T3: CORRIDOR", "threshold": 2.5, "desc": "Aggressive advance & evasion"},
    {"scenario": "deathmatch.cfg", "tier_name": "T4: DEATHMATCH", "threshold": 999.0, "desc": "Full open-arena tactical combat"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Window Parallel AI Training Surveillance Grid")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=4,
        choices=[4, 6],
        help="Number of concurrent game windows tiled in the grid (4 for 2x2, 6 for 2x3). Default: 4",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="co_evolution",
        choices=["co_evolution", "director_only", "combat_only"],
        help="Training mode: 'co_evolution' (train both Marine & Director), 'director_only', or 'combat_only'.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="curriculum",
        help="Scenario config ('curriculum' for auto-graduating curriculum training, 'mixed' for fixed mixed tiles, or specify e.g. 'defend_the_center.cfg'). Default: curriculum",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=True,
        help="Enable curriculum learning: all tiles start on defend_the_center and graduate to harder scenarios as avg kills improve. Default: True",
    )
    parser.add_argument(
        "--no-curriculum",
        dest="curriculum",
        action="store_false",
        help="Disable curriculum learning and use static scenarios specified by --scenario.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Total training episodes per tile. Default: 30",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=35,
        help="Target display framerate (35 is native Doom speed). Default: 35",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=4,
        help="Action repeat tics per step (4 tics ~ 114ms provides physics momentum & responsive aiming). Default: 4",
    )
    parser.add_argument(
        "--reset-marine",
        action="store_true",
        default=False,
        help="Reset Marine weights and warm up with fresh tactical bootstrap.",
    )
    parser.add_argument(
        "--director-interval",
        type=float,
        default=3.0,
        help="Director decision interval in seconds. Default: 3.0",
    )
    parser.add_argument(
        "--director-model",
        type=str,
        default=str(PROJECT_ROOT / "models" / "final" / "director_dqn.pt"),
        help="Path to load/save Director model",
    )
    parser.add_argument(
        "--combat-model",
        type=str,
        default=str(PROJECT_ROOT / "models" / "final" / "combat_marine_dqn.pt"),
        help="Path to load/save Combat Marine model",
    )
    return parser.parse_args()


class ParallelTileInstance:
    """Manages an individual ViZDoom game instance within the visual grid."""

    def __init__(
        self,
        tile_id: int,
        scenario_name: str,
        director_interval_sec: float,
        curriculum_mode: bool = False,
    ) -> None:
        self.tile_id = tile_id
        self.curriculum_mode = curriculum_mode
        self.curriculum_tier = 0
        if curriculum_mode:
            self.scenario_name = CURRICULUM_STAGES[0]["scenario"]
        else:
            self.scenario_name = scenario_name
        self.scenario_path = resolve_scenario_path(self.scenario_name)
        self.director_interval_tics = int(director_interval_sec * vzd.DEFAULT_TICRATE)

        # ViZDoom Engine Setup
        self._init_engine(self.scenario_name)

        # Telemetry & Pacing Trackers
        self.feature_extractor = FeatureExtractor(ticrate=vzd.DEFAULT_TICRATE)
        self.skill_estimator = SkillEstimator()
        self.difficulty_state = DifficultyState(doom_skill=3)
        self.actuator = EngineActuator(self.game)
        self.reward_fn = FlowStateReward()

        # Heuristic scripted bot fallback (used in director_only mode)
        self.scripted_bot = SimulatedPlayerAgent("intermediate", self.available_buttons)

        # State cache
        self.tic_counter = 0
        self.last_director_decision_tic = 0
        self.prev_combat_state: Optional[np.ndarray] = None
        self.prev_combat_action: int = 0
        self.prev_game_vars: List[float] = []
        self.prev_director_state: Optional[np.ndarray] = None
        self.prev_director_action: int = 0

        self.latest_director_action = DDAAction.NO_CHANGE
        self.latest_director_msg = "Initialized"
        self.latest_combat_action_name = "IDLE"
        self.latest_flow_reward = 0.0

        self.episodes_completed = 0
        self.is_dead = False
        self.health = 100.0
        self.ammo = 50.0
        self.killcount = 0
        self.last_pos = (0.0, 0.0)
        self.last_angle = 0.0
        self.last_killcount = 0
        self.stuck_counter = 0
        self.idle_counter = 0  # For stationary combat scenarios (no movement expected)

        # Per-tile rolling performance metrics (visible in HUD)
        self.episode_kill_history: List[int] = []
        self.episode_reward_accum = 0.0
        self.episode_reward_history: List[float] = []
        self.episode_survival_tics = 0
        self.episode_survival_history: List[int] = []
        self.avg_kills = 0.0
        self.avg_reward = 0.0
        self.avg_survival = 0.0
        self.peak_kills = 0

        # Curriculum tier metrics
        self.tier_episodes = 0
        self.tier_kill_history: List[int] = []

        self.reset_episode()

    def _init_engine(self, scenario_name: str) -> None:
        """Initializes or reinitializes the ViZDoom engine instance for a given scenario."""
        self.scenario_path = resolve_scenario_path(scenario_name)
        self.game = vzd.DoomGame()
        self.game.load_config(self.scenario_path)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)  # Completely silent by default
        self.game.set_render_hud(True)
        self.game.set_render_crosshair(True)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_doom_skill(3)
        self.game.set_objects_info_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_sectors_info_enabled(False)

        # Standard game variables
        self.game.clear_available_game_variables()
        for gv in [
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
        ]:
            self.game.add_available_game_variable(gv)

        self.game.init()
        try:
            self.game.send_game_command("sv_cheats 1")
            self.game.send_game_command("cl_autoweapon 1")
        except Exception:
            pass
        self.available_buttons = self.game.get_available_buttons()

    def switch_scenario(self, new_scenario_name: str) -> None:
        """Dynamically hot-swaps the ViZDoom engine to a new scenario for curriculum progression."""
        self.scenario_name = new_scenario_name
        try:
            self.game.close()
        except Exception:
            pass
        self._init_engine(new_scenario_name)
        self.actuator = EngineActuator(self.game)
        self.scripted_bot = SimulatedPlayerAgent("intermediate", self.available_buttons)
        self.reset_episode()

    def reset_episode(self, corridor_done: bool = False) -> None:
        """Starts a new episode on this tile and records metrics from the finished episode."""
        # Record metrics from the episode that just ended (skip the very first init call)
        if self.episodes_completed > 0 or self.tic_counter > 0:
            self.episode_kill_history.append(self.killcount)
            self.episode_reward_history.append(self.episode_reward_accum)
            self.episode_survival_history.append(self.episode_survival_tics)
            if self.killcount > self.peak_kills:
                self.peak_kills = self.killcount
            # Rolling averages over last 10 episodes
            recent_k = self.episode_kill_history[-10:]
            recent_r = self.episode_reward_history[-10:]
            recent_s = self.episode_survival_history[-10:]
            self.avg_kills = sum(recent_k) / len(recent_k) if recent_k else 0.0
            self.avg_reward = sum(recent_r) / len(recent_r) if recent_r else 0.0
            self.avg_survival = sum(recent_s) / len(recent_s) if recent_s else 0.0

            # Check curriculum graduation
            if self.curriculum_mode and self.curriculum_tier < len(CURRICULUM_STAGES) - 1:
                self.tier_kill_history.append(self.killcount)
                self.tier_episodes += 1
                recent_tier_k = self.tier_kill_history[-5:]
                avg_tier_k = sum(recent_tier_k) / len(recent_tier_k) if recent_tier_k else 0.0
                curr_stage = CURRICULUM_STAGES[self.curriculum_tier]

                should_graduate = (
                    (self.tier_episodes >= 3 and avg_tier_k >= curr_stage["threshold"])
                    or (corridor_done and "corridor" in self.scenario_name)
                    or (self.killcount >= curr_stage["threshold"] + 3)
                )
                if should_graduate:
                    self.curriculum_tier += 1
                    next_stage = CURRICULUM_STAGES[self.curriculum_tier]
                    logger.info(
                        f"🎓 [CURRICULUM GRADUATION] Tile #{self.tile_id + 1} PROMOTED to "
                        f"Tier {self.curriculum_tier + 1} ({next_stage['tier_name']}: {next_stage['scenario']})! "
                        f"(Tier Avg Kills: {avg_tier_k:.1f}, Target: {curr_stage['threshold']:.1f})"
                    )
                    self.tier_episodes = 0
                    self.tier_kill_history.clear()
                    self.switch_scenario(next_stage["scenario"])
                    return

        self.game.new_episode()
        try:
            self.game.send_game_command("sv_cheats 1")
            self.game.send_game_command("cl_autoweapon 1")
        except Exception:
            pass
        self.feature_extractor.reset()
        self.skill_estimator.reset()
        self.difficulty_state = DifficultyState(doom_skill=3)
        self.tic_counter = 0
        self.last_director_decision_tic = 0
        self.prev_combat_state = None
        self.prev_game_vars = []
        self.prev_director_state = None
        self.prev_director_action = 0
        self.is_dead = False
        self.killcount = 0
        self.last_pos = (0.0, 0.0)
        self.last_angle = 0.0
        self.last_killcount = 0
        self.stuck_counter = 0
        self.idle_counter = 0
        self.episode_reward_accum = 0.0
        self.episode_survival_tics = 0

    def close(self) -> None:
        self.game.close()


def warmup_combat_agent(combat_agent: CombatAgent, steps: int = 1200) -> None:
    """Rapid headless tactical warm-up to bootstrap combat aiming, tracking, and firing."""
    logger.info(f"Initiating rapid tactical warm-up ({steps} steps in headless engine)...")
    scenarios = ["defend_the_center.cfg", "deadly_corridor.cfg"]
    for scen_name in scenarios:
        try:
            game = vzd.DoomGame()
            game.load_config(resolve_scenario_path(scen_name))
            game.set_window_visible(False)
            game.set_sound_enabled(False)
            game.set_objects_info_enabled(True)
            game.set_labels_buffer_enabled(True)
            game.clear_available_game_variables()
            for gv in [
                vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR, vzd.GameVariable.SELECTED_WEAPON_AMMO,
                vzd.GameVariable.KILLCOUNT, vzd.GameVariable.DEATHCOUNT, vzd.GameVariable.HITCOUNT,
                vzd.GameVariable.HITS_TAKEN, vzd.GameVariable.DAMAGECOUNT, vzd.GameVariable.DAMAGE_TAKEN,
                vzd.GameVariable.DEAD, vzd.GameVariable.SELECTED_WEAPON, vzd.GameVariable.POSITION_X,
                vzd.GameVariable.POSITION_Y, vzd.GameVariable.POSITION_Z, vzd.GameVariable.VELOCITY_X,
                vzd.GameVariable.VELOCITY_Y, vzd.GameVariable.VELOCITY_Z, vzd.GameVariable.ATTACK_READY,
                vzd.GameVariable.ANGLE,
            ]:
                game.add_available_game_variable(gv)
            game.init()
            game.new_episode()
            bot = SimulatedPlayerAgent("aggressive", game.get_available_buttons())

            prev_s = None
            prev_act = 0
            prev_vars = []

            sub_steps = steps // len(scenarios)
            for _ in range(sub_steps):
                if game.is_episode_finished() or game.is_player_dead():
                    game.new_episode()
                    prev_s = None
                    prev_vars = []
                s = game.get_state()
                if s is None:
                    continue
                g_vars = list(s.game_variables)
                c_state = CombatStateExtractor.extract_state(s)

                bot_vec = bot.compute_action(s, g_vars, scen_name)

                tl, tr, atk, mf = False, False, False, False
                for btn_idx, b in enumerate(game.get_available_buttons()):
                    if b == vzd.Button.TURN_LEFT and bot_vec[btn_idx]: tl = True
                    elif b == vzd.Button.TURN_RIGHT and bot_vec[btn_idx]: tr = True
                    elif b == vzd.Button.ATTACK and bot_vec[btn_idx]: atk = True
                    elif b == vzd.Button.MOVE_FORWARD and bot_vec[btn_idx]: mf = True

                if mf and atk: act_idx = 4
                elif mf and tl: act_idx = 5
                elif mf and tr: act_idx = 6
                elif atk: act_idx = 2
                elif tl: act_idx = 0
                elif tr: act_idx = 1
                else: act_idx = 3

                game.make_action(bot_vec, 4)

                if prev_s is not None:
                    r = combat_agent.calculate_combat_reward(prev_vars, g_vars, prev_act, c_state, prev_s)
                    combat_agent.replay_buffer.push(prev_s, prev_act, r, c_state, game.is_player_dead())
                    combat_agent.train_step()

                prev_s = c_state
                prev_act = act_idx
                prev_vars = g_vars

            game.close()
        except Exception as e:
            logger.warning(f"Tactical warm-up on {scen_name} encountered note: {e}")
    logger.info(f"Tactical warm-up complete! Marine trained on {len(combat_agent.replay_buffer)} transitions.")


def run_visual_grid(args: argparse.Namespace) -> None:
    # 1. Initialize Pygame Master Display Window
    pygame.init()
    pygame.font.init()

    # Determine Grid Geometry (Rows, Columns, Dimensions)
    if args.grid_size == 6:
        cols, rows = 3, 2
        tile_w, tile_h = 426, 320
        win_w, win_h = 1280, 680
    else:  # 4 tiles (2x2)
        cols, rows = 2, 2
        tile_w, tile_h = 512, 384
        win_w, win_h = 1024, 808

    header_h = 40  # Top dashboard status bar
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Adaptive Game Director: Parallel AI Surveillance Grid ({cols}x{rows} Matrix)")
    clock = pygame.time.Clock()

    font_small = pygame.font.SysFont("Consolas", 13, bold=True)
    font_large = pygame.font.SysFont("Arial", 16, bold=True)
    font_badge = pygame.font.SysFont("Arial", 12, bold=True)

    # 2. Initialize Shared AI Agents
    director_agent = DQNAgent(state_dim=15, action_dim=7, learning_rate=3e-4)
    combat_agent = CombatAgent(state_dim=14, action_dim=7, lr=3e-4)

    # Load existing weights if present
    dir_path = Path(args.director_model)
    if dir_path.exists():
        try:
            director_agent.load_checkpoint(dir_path)
            logger.info(f"Loaded existing Director model from: {dir_path}")
        except Exception as e:
            logger.warning(f"Could not load Director checkpoint: {e}")

    comb_path = Path(args.combat_model)
    if comb_path.exists() and not args.reset_marine:
        try:
            combat_agent.load_checkpoint(comb_path)
            logger.info(f"Loaded existing Combat Marine model from: {comb_path}")
        except Exception as e:
            logger.warning(f"Could not load Combat Marine checkpoint: {e}")
            warmup_combat_agent(combat_agent, steps=1000)
    else:
        logger.info("Initializing Combat Marine with fresh tactical bootstrap...")
        warmup_combat_agent(combat_agent, steps=1000)

    # 3. Create Parallel Game Instances
    tiles: List[ParallelTileInstance] = []
    logger.info(f"Spawning {args.grid_size} parallel ViZDoom game instances...")
    use_curriculum = bool(args.curriculum or args.scenario == "curriculum")

    for i in range(args.grid_size):
        if use_curriculum:
            scen = CURRICULUM_STAGES[0]["scenario"]
        elif args.scenario == "mixed":
            scen = DEFAULT_SCENARIOS[i % len(DEFAULT_SCENARIOS)]
        else:
            scen = args.scenario
        tile = ParallelTileInstance(
            tile_id=i,
            scenario_name=scen,
            director_interval_sec=args.director_interval,
            curriculum_mode=use_curriculum,
        )
        tiles.append(tile)
        tier_tag = f" [Curriculum Tier 1: {CURRICULUM_STAGES[0]['tier_name']}]" if use_curriculum else ""
        logger.info(f"  Tile {i}: Initialized in scenario '{scen}'{tier_tag}")

    logger.info("=" * 80)
    logger.info(f"SURVEILLANCE GRID ACTIVE | Mode: {args.mode.upper()} | Target FPS: {args.fps}")
    logger.info("Press 'Q' or 'Esc' in the Pygame window anytime to exit and save weights.")
    logger.info("=" * 80)

    # Epsilon decay schedules — start lower and decay faster for visible improvement
    director_epsilon = 0.30 if dir_path.exists() else 0.80
    combat_epsilon = 0.25 if comb_path.exists() else 0.70
    total_director_transitions = 0
    total_combat_transitions = 0
    last_log_time = time.time()
    training_start_time = time.time()
    running = True

    try:
        while running:
            # Handle Pygame UI events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                        running = False

            # Draw master background
            screen.fill((14, 17, 23))

            # -------------------------------------------------------------
            # STEP ALL PARALLEL TILES
            # -------------------------------------------------------------
            for idx, tile in enumerate(tiles):
                state = tile.game.get_state()
                if state is None or tile.game.is_episode_finished() or tile.is_dead:
                    tile.episodes_completed += 1
                    tile.reset_episode()
                    continue

                g_vars = list(state.game_variables)
                tile.health = float(g_vars[0]) if len(g_vars) > 0 else 0.0
                tile.ammo = float(g_vars[2]) if len(g_vars) > 2 else 0.0
                tile.killcount = int(g_vars[3]) if len(g_vars) > 3 else 0
                tile.is_dead = bool(g_vars[9] > 0 or tile.health <= 0)

                pos_x = float(g_vars[11]) if len(g_vars) > 11 else 0.0
                pos_y = float(g_vars[12]) if len(g_vars) > 12 else 0.0
                armor = float(g_vars[1]) if len(g_vars) > 1 else 0.0

                # 1. Level Completion: Deadly corridor reached the end (X >= 1150 or green armor collected)
                corridor_done = "deadly_corridor" in tile.scenario_name.lower() and (pos_x >= 1150.0 or armor > 0)

                # 2. Smart Anti-Stuck Watchdog (scenario-aware)
                is_stationary_arena = any(s in tile.scenario_name.lower() for s in ["defend_the_center", "defend_the_line"])
                player_angle = float(g_vars[18]) if len(g_vars) > 18 else 0.0

                if is_stationary_arena:
                    # Stationary arenas: player stands on a platform and rotates to shoot.
                    # "Stuck" = not rotating AND not getting kills for a long time.
                    angle_delta = abs(player_angle - tile.last_angle)
                    if angle_delta > 180.0:
                        angle_delta = 360.0 - angle_delta
                    kills_changed = tile.killcount > tile.last_killcount

                    if angle_delta < 2.0 and not kills_changed:
                        tile.idle_counter += args.frame_skip
                    else:
                        tile.idle_counter = 0
                    tile.last_angle = player_angle
                    tile.last_killcount = tile.killcount
                    is_stuck = tile.idle_counter >= 700  # ~20 seconds of zero activity
                else:
                    # Movement-based scenarios (corridor, deathmatch): check position changes
                    dist_moved = math.hypot(pos_x - tile.last_pos[0], pos_y - tile.last_pos[1])
                    if dist_moved < 5.0:
                        tile.stuck_counter += args.frame_skip
                    else:
                        tile.stuck_counter = 0
                        tile.last_pos = (pos_x, pos_y)
                    is_stuck = tile.stuck_counter >= 700  # ~20 seconds stuck at a wall

                if tile.game.is_episode_finished() or tile.is_dead or corridor_done or is_stuck:
                    if corridor_done:
                        logger.info(f"Tile #{idx + 1} ({tile.scenario_name}): Reached end of corridor! Level completed. Restarting...")
                    elif is_stuck:
                        logger.info(f"Tile #{idx + 1} ({tile.scenario_name}): Stuck at wall/corner for 200 tics. Resetting...")
                    tile.episodes_completed += 1
                    tile.reset_episode(corridor_done=corridor_done)
                    continue

                # A. COMBAT MARINE DECISION & ACTION (Inner Loop: 35 FPS)
                combat_state = CombatStateExtractor.extract_state(state)
                is_on_target = bool(combat_state[12] > 0.5)

                if args.mode in ["co_evolution", "combat_only"]:
                    combat_action = combat_agent.select_action(combat_state, epsilon=combat_epsilon)
                    tile.latest_combat_action_name = COMBAT_ACTIONS[combat_action]
                    btn_vec = combat_agent.convert_action_to_buttons(combat_action, tile.available_buttons, on_target=is_on_target)

                    # Auto-equip heavier artillery if player is on pistol (weapon 2) and has picked up a better gun
                    sel_w = float(g_vars[10]) if len(g_vars) > 10 else 2.0
                    if sel_w <= 2.0:
                        for w_btn in [vzd.Button.SELECT_WEAPON6, vzd.Button.SELECT_WEAPON5, vzd.Button.SELECT_WEAPON4, vzd.Button.SELECT_WEAPON3, vzd.Button.SELECT_NEXT_WEAPON]:
                            if w_btn in tile.available_buttons:
                                b_idx = tile.available_buttons.index(w_btn)
                                btn_vec[b_idx] = True
                                break
                else:
                    # Scripted bot fallback
                    btn_vec = tile.scripted_bot.compute_action(state, g_vars, tile.scenario_name)
                    tile.latest_combat_action_name = "SCRIPTED_BOT"

                # Advance engine by frame_skip tics (provides momentum & responsive aiming)
                tile.game.make_action(btn_vec, args.frame_skip)
                tile.tic_counter += args.frame_skip

                # Track survival time
                tile.episode_survival_tics += args.frame_skip

                # Train Combat Agent via RL transition replay (4x gradient updates for faster learning)
                if args.mode in ["co_evolution", "combat_only"] and tile.prev_combat_state is not None:
                    c_reward = combat_agent.calculate_combat_reward(
                        prev_vars=tile.prev_game_vars,
                        curr_vars=g_vars,
                        action_idx=tile.prev_combat_action,
                        state_vec=combat_state,
                        prev_state_vec=tile.prev_combat_state,
                    )
                    tile.episode_reward_accum += c_reward
                    combat_agent.replay_buffer.push(
                        tile.prev_combat_state,
                        tile.prev_combat_action,
                        c_reward,
                        combat_state,
                        tile.is_dead,
                    )
                    total_combat_transitions += 1
                    # 4x gradient updates per transition for accelerated learning
                    for _ in range(4):
                        combat_agent.train_step()

                tile.prev_combat_state = combat_state
                tile.prev_combat_action = combat_action if args.mode in ["co_evolution", "combat_only"] else 0
                tile.prev_game_vars = g_vars

                # B. AI GAME DIRECTOR DECISION LOOP (Outer Loop: every interval_sec)
                if tile.tic_counter - tile.last_director_decision_tic >= tile.director_interval_tics:
                    tile.last_director_decision_tic = tile.tic_counter

                    enemy_count, item_count = count_level_entities(state.objects)
                    raw_frame = RawTelemetryFrame(
                        episode=tile.episodes_completed + 1,
                        tic=tile.tic_counter,
                        timestamp=time.time(),
                        health=tile.health,
                        armor=float(g_vars[1]) if len(g_vars) > 1 else 0.0,
                        ammo=tile.ammo,
                        killcount=tile.killcount,
                        deathcount=float(g_vars[4]) if len(g_vars) > 4 else 0.0,
                        hitcount=float(g_vars[5]) if len(g_vars) > 5 else 0.0,
                        hits_taken=float(g_vars[6]) if len(g_vars) > 6 else 0.0,
                        damagecount=float(g_vars[7]) if len(g_vars) > 7 else 0.0,
                        damage_taken=float(g_vars[8]) if len(g_vars) > 8 else 0.0,
                        is_dead=tile.is_dead,
                        selected_weapon=float(g_vars[10]) if len(g_vars) > 10 else 0.0,
                        pos_x=float(g_vars[11]) if len(g_vars) > 11 else 0.0,
                        pos_y=float(g_vars[12]) if len(g_vars) > 12 else 0.0,
                        pos_z=float(g_vars[13]) if len(g_vars) > 13 else 0.0,
                        vel_x=float(g_vars[14]) if len(g_vars) > 14 else 0.0,
                        vel_y=float(g_vars[15]) if len(g_vars) > 15 else 0.0,
                        vel_z=float(g_vars[16]) if len(g_vars) > 16 else 0.0,
                        attack_ready=bool(len(g_vars) > 17 and g_vars[17] > 0),
                        current_doom_skill=tile.difficulty_state.doom_skill,
                        last_reward=tile.game.get_last_reward(),
                        enemy_count=enemy_count,
                        item_count=item_count,
                    )
                    derived = tile.feature_extractor.compute_derived_metrics(raw_frame, tile.skill_estimator.skill_score)
                    tile.skill_estimator.update(
                        health=tile.health,
                        damage_dealt=raw_frame.damagecount,
                        damage_taken=raw_frame.damage_taken,
                        killcount=raw_frame.killcount,
                        deathcount=raw_frame.deathcount,
                        accuracy=derived.accuracy_estimate,
                    )

                    # Compute Flow State reward
                    d_reward = tile.reward_fn.calculate_reward(raw_frame, derived, args.director_interval)
                    tile.latest_flow_reward = d_reward

                    # Extract 15-dim state and select DDA action
                    state_vec = tile.feature_extractor.extract_normalized_state_vector(raw_frame, derived)

                    if args.mode in ["co_evolution", "director_only"]:
                        # 1. Temporal credit assignment: store previous decision evaluated against outcome observed 3s later
                        if tile.prev_director_state is not None:
                            director_agent.store_transition(
                                tile.prev_director_state,
                                tile.prev_director_action,
                                d_reward,
                                state_vec,
                                tile.is_dead,
                            )
                            total_director_transitions += 1
                            # 4x gradient updates per transition for accelerated Director learning
                            for _ in range(4):
                                director_agent.train_step()

                        # 2. Select and execute next DDA action
                        d_action_idx = director_agent.select_action(state_vec, epsilon=director_epsilon)
                        tile.latest_director_action = DDAAction(d_action_idx)
                        # Actuate into engine with safety envelope
                        tile.latest_director_msg = tile.actuator.execute(
                            tile.latest_director_action,
                            tile.difficulty_state,
                            player_health=tile.health,
                            player_ammo=tile.ammo,
                        )

                        tile.prev_director_state = state_vec
                        tile.prev_director_action = d_action_idx

                # C. RENDER TILE TO PYGAME DISPLAY
                screen_buf = state.screen_buffer  # (240, 320, 3) uint8
                surf = pygame.surfarray.make_surface(np.transpose(screen_buf, (1, 0, 2)))
                scaled_surf = pygame.transform.scale(surf, (tile_w - 4, tile_h - 4))

                col_idx = idx % cols
                row_idx = idx // cols
                pos_x = col_idx * tile_w + 2
                pos_y = header_h + row_idx * tile_h + 2

                screen.blit(scaled_surf, (pos_x, pos_y))

                # Overlay 1: Top Bar (Vitals & Scenario)
                overlay_top = pygame.Surface((tile_w - 4, 28), pygame.SRCALPHA)
                overlay_top.fill((15, 23, 42, 210))
                screen.blit(overlay_top, (pos_x, pos_y))

                if tile.curriculum_mode:
                    tier_str = CURRICULUM_STAGES[tile.curriculum_tier]["tier_name"]
                    vitals_text = f"Tile #{idx + 1} | [{tier_str}] | HP: {tile.health:3.0f} | Ammo: {tile.ammo:2.0f} | K: {tile.killcount:2d}"
                else:
                    vitals_text = f"Tile #{idx + 1} | {tile.scenario_name[:18]} | HP: {tile.health:3.0f} | Ammo: {tile.ammo:2.0f} | K: {tile.killcount:2d}"
                text_surf = font_small.render(vitals_text, True, (241, 245, 249))
                screen.blit(text_surf, (pos_x + 8, pos_y + 6))

                # Status Badge: Flow Channel [35, 75]
                in_flow = 35.0 <= tile.health <= 75.0
                badge_text = "FLOW" if in_flow else ("DISTRESS" if tile.health < 35 else "COMFORT")
                badge_color = (52, 211, 153) if in_flow else ((248, 113, 113) if tile.health < 35 else (96, 165, 250))
                badge_surf = font_badge.render(f"[{badge_text}]", True, badge_color)
                screen.blit(badge_surf, (pos_x + tile_w - 85, pos_y + 6))

                # Overlay 2: LIVE METRICS & DIRECTOR TREATMENT BAR
                overlay_mid = pygame.Surface((tile_w - 4, 22), pygame.SRCALPHA)
                overlay_mid.fill((15, 23, 42, 180))
                mid_y = pos_y + 28
                screen.blit(overlay_mid, (pos_x, mid_y))

                # Extract live Director state
                diff_meta = DOOM_SKILL_META.get(tile.difficulty_state.doom_skill, DOOM_SKILL_META[3])
                tier_name = getattr(tile.skill_estimator.tier, "value", str(tile.skill_estimator.tier))
                tier_meta = SKILL_TIER_META.get(tier_name.upper(), SKILL_TIER_META["INTERMEDIATE"])

                # Metrics text displaying exact difficulty state and player skill assessment
                metrics_text = (
                    f"DIFF: {diff_meta['name']} | "
                    f"SEES: {tier_meta['label'][:6]} ({tile.skill_estimator.skill_score:.2f}) | "
                    f"AvgK: {tile.avg_kills:.1f} | Ep: {tile.episodes_completed}"
                )
                metrics_surf = font_badge.render(metrics_text, True, diff_meta["color"])
                screen.blit(metrics_surf, (pos_x + 8, mid_y + 4))

                # Overlay 3: Bottom Bar (Director & Combat Action)
                overlay_bot = pygame.Surface((tile_w - 4, 30), pygame.SRCALPHA)
                overlay_bot.fill((15, 23, 42, 220))
                bot_y = pos_y + tile_h - 34
                screen.blit(overlay_bot, (pos_x, bot_y))

                # Format Director intervention with clear color-coded feedback
                if "Dropped" in tile.latest_director_msg:
                    item_type = "Medikit+Ammo" if "Medikit" in tile.latest_director_msg else "Stimpack"
                    dir_tag = f"[AID] Dropped {item_type}"
                    dir_color = (52, 211, 153)  # Bright emerald green
                elif "Spawned" in tile.latest_director_msg:
                    threat = tile.latest_director_msg.split("Spawned ")[-1].replace("!", "")[:12]
                    dir_tag = f"[SPAWN] {threat}"
                    dir_color = (248, 113, 113)  # Alert coral red
                elif "Thinned" in tile.latest_director_msg:
                    dir_tag = "[EASE] Thinned Foes"
                    dir_color = (96, 165, 250)  # Calm blue
                else:
                    dir_tag = f"[DIR] {tile.latest_director_action.name[:16]}"
                    dir_color = (251, 191, 36)  # Amber gold

                dir_text = f"{dir_tag:22s} | [MARINE] {tile.latest_combat_action_name:14s}"
                dir_surf = font_small.render(dir_text, True, dir_color)
                screen.blit(dir_surf, (pos_x + 8, bot_y + 8))

            # Draw Master Header Bar
            header_rect = pygame.Surface((win_w, header_h), pygame.SRCALPHA)
            header_rect.fill((10, 14, 26, 255))
            screen.blit(header_rect, (0, 0))

            elapsed = time.time() - training_start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            avg_k_all = sum(t.avg_kills for t in tiles) / max(1, len(tiles))
            curric_tag = "[CURRICULUM GRADUATION] | " if use_curriculum else ""
            header_info = (
                f"CO-EVOLUTION | {curric_tag}{mins}m{secs:02d}s | "
                f"Combat: {total_combat_transitions} (eps={combat_epsilon:.2f}) | "
                f"Director: {total_director_transitions} (eps={director_epsilon:.2f}) | "
                f"AvgK: {avg_k_all:.1f} | FPS: {clock.get_fps():.0f}"
            )
            head_surf = font_large.render(header_info, True, (56, 189, 248))
            screen.blit(head_surf, (15, 10))

            # Decay exploration epsilons faster for visible improvement
            if total_combat_transitions % 500 == 0 and total_combat_transitions > 0:
                combat_epsilon = max(0.05, combat_epsilon * 0.96)
            if total_director_transitions % 30 == 0 and total_director_transitions > 0:
                director_epsilon = max(0.05, director_epsilon * 0.96)

            # Periodic console stats logging (every 30 seconds)
            now = time.time()
            if now - last_log_time >= 30.0:
                elapsed = now - training_start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                avg_k_all = sum(t.avg_kills for t in tiles) / len(tiles)
                avg_r_all = sum(t.avg_reward for t in tiles) / len(tiles)
                total_eps = sum(t.episodes_completed for t in tiles)
                logger.info(
                    f"[{mins}m{secs:02d}s] Combat: {total_combat_transitions} steps (eps={combat_epsilon:.3f}) | "
                    f"Director: {total_director_transitions} steps (eps={director_epsilon:.3f}) | "
                    f"Avg Kills/Ep: {avg_k_all:.1f} | Avg Reward/Ep: {avg_r_all:.1f} | Total Episodes: {total_eps}"
                )
                last_log_time = now

            pygame.display.flip()
            clock.tick(args.fps)

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user.")
    finally:
        # Save checkpoints
        logger.info("\nSaving trained model weights...")
        director_agent.save_checkpoint(Path(args.director_model))
        combat_agent.save_checkpoint(Path(args.combat_model))
        logger.info(f"Saved Director model to: {args.director_model}")
        logger.info(f"Saved Combat Marine model to: {args.combat_model}")

        for t in tiles:
            t.close()
        pygame.quit()
        logger.info("Visual grid training session finished cleanly.")


def main():
    args = parse_args()
    run_visual_grid(args)


if __name__ == "__main__":
    main()
