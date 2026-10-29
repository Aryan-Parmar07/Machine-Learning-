#!/usr/bin/env python3
"""Automated Telemetry Collection Pipeline with Simulated Player Personas.

Simulates human player personas (Novice, Intermediate, Aggressive, Random)
in ViZDoom at high speed (300+ FPS) to generate comprehensive baseline datasets
for player modeling, feature analysis, and RL DDA training.

Usage:
    # Collect 3 episodes across all personas in defend_the_center
    python scripts/collect_telemetry.py --scenario defend_the_center.cfg --episodes 3 --persona all

    # Run aggressive persona in deathmatch
    python scripts/collect_telemetry.py --scenario deathmatch.cfg --episodes 5 --persona aggressive

    # Watch the bot play visually in a window
    python scripts/collect_telemetry.py --render
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import vizdoom as vzd

from src.player_model.features import (
    TELEMETRY_COLUMNS,
    RawTelemetryFrame,
    DerivedMetrics,
    FeatureExtractor,
)
from src.player_model.skill_estimator import SkillEstimator
from src.utils.logger import get_logger, TelemetryRecorder
from scripts.play_human import resolve_scenario_path, count_level_entities

logger = get_logger("CollectTelemetry")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated ViZDoom Telemetry Collector using Simulated Player Personas"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="defend_the_center.cfg",
        help="ViZDoom scenario filename (e.g., defend_the_center.cfg, deathmatch.cfg). Default: defend_the_center.cfg",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="all",
        choices=["novice", "intermediate", "aggressive", "random", "all"],
        help="Player persona to simulate: novice, intermediate, aggressive, random, or all. Default: all",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes per persona. Default: 3",
    )
    parser.add_argument(
        "--interval-tics",
        type=int,
        default=5,
        help="Sampling interval in game tics (~7Hz at 5 tics). Default: 5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "telemetry" / "human_session.csv"),
        help="Output CSV file path. Default: data/telemetry/human_session.csv",
    )
    parser.add_argument(
        "--skill",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Doom skill level (1 to 5). Default: 3",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Show visual game window (runs in real-time). Default: False (headless high-speed ~350 FPS)",
    )
    return parser.parse_args()


class SimulatedPlayerAgent:
    """Simulates realistic human combat actions based on player persona parameters."""

    def __init__(self, persona: str, available_buttons: List[vzd.Button]) -> None:
        self.persona = persona
        self.buttons = available_buttons

        # Persona configuration parameters
        if persona == "novice":
            self.aim_noise_deg = 25.0       # Higher aim inaccuracy
            self.aim_tolerance_deg = 20.0   # Shoots even when poorly aligned
            self.fire_probability = 0.35    # Hesitant trigger finger
            self.turn_speed = 6.0           # Sluggish turning
            self.reaction_delay_tics = 8    # Slower reactions
        elif persona == "intermediate":
            self.aim_noise_deg = 8.0
            self.aim_tolerance_deg = 12.0
            self.fire_probability = 0.70
            self.turn_speed = 12.0
            self.reaction_delay_tics = 3
        elif persona == "aggressive":
            self.aim_noise_deg = 3.0        # Sharp precision
            self.aim_tolerance_deg = 8.0
            self.fire_probability = 0.95    # High rate of fire
            self.turn_speed = 20.0          # Fast snap turns
            self.reaction_delay_tics = 1
        else:  # random
            self.aim_noise_deg = 180.0
            self.aim_tolerance_deg = 180.0
            self.fire_probability = 0.50
            self.turn_speed = 15.0
            self.reaction_delay_tics = 0

        self._tic_counter = 0

    def compute_action(
        self,
        state: vzd.GameState,
        game_vars: np.ndarray,
        scenario_name: str,
    ) -> List[Any]:
        """Computes discrete/delta action vector based on available scenario buttons."""
        self._tic_counter += 1

        # Check if scenario is defend_the_center (buttons: [TURN_LEFT, TURN_RIGHT, ATTACK])
        is_defend_scenario = len(self.buttons) == 3 and self.buttons[0] == vzd.Button.TURN_LEFT

        if self.persona == "random":
            if is_defend_scenario:
                return random.choice([
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [False, False, False],
                ])
            else:
                return [random.random() > 0.8 for _ in self.buttons]

        # Extract player position and angle from state
        # In our configuration: pos_x = var[11], pos_y = var[12], angle = var[18] if available
        # Or look at state.objects directly
        player_obj = None
        enemies: List[Tuple[float, float, float]] = []

        if hasattr(state, "objects") and state.objects:
            for obj in state.objects:
                name_l = obj.name.lower()
                if obj.id == 0 and "player" in name_l:
                    player_obj = obj
                elif obj.id != 0 and ("demon" in name_l or "marine" in name_l or "guy" in name_l or "player" in name_l):
                    enemies.append((obj.position_x, obj.position_y, obj.position_z))

        if not player_obj and hasattr(state, "game_variables") and len(state.game_variables) >= 14:
            px = float(state.game_variables[11])
            py = float(state.game_variables[12])
            # Default forward angle approximation if angle not in vars
            p_angle = float(state.game_variables[18]) if len(state.game_variables) > 18 else 0.0
        elif player_obj:
            px, py, p_angle = player_obj.position_x, player_obj.position_y, player_obj.angle
        else:
            px, py, p_angle = 0.0, 0.0, 0.0

        # Find nearest threat
        nearest_dx, nearest_dy = None, None
        min_dist = float("inf")
        for ex, ey, _ in enemies:
            dx = ex - px
            dy = ey - py
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
                nearest_dx, nearest_dy = dx, dy

        turn_left, turn_right, attack = False, False, False
        move_forward = False

        if nearest_dx is not None and nearest_dy is not None:
            # Target heading in degrees
            target_deg = math.degrees(math.atan2(nearest_dy, nearest_dx))
            # Add persona-specific aiming noise
            if self.aim_noise_deg > 0:
                target_deg += random.gauss(0, self.aim_noise_deg)

            diff = (target_deg - p_angle + 180.0) % 360.0 - 180.0

            if abs(diff) <= self.aim_tolerance_deg:
                attack = random.random() < self.fire_probability
                move_forward = self.persona in ["intermediate", "aggressive"]
            elif diff > 0:
                turn_left = True
            else:
                turn_right = True
        else:
            # Search & explore: scan while advancing forward through the environment
            turn_left = random.random() < 0.4
            turn_right = random.random() < 0.2 if not turn_left else False
            move_forward = not is_defend_scenario and (random.random() < 0.7)
            attack = random.random() < 0.05

        # Format action vector according to scenario button list
        if is_defend_scenario:
            return [turn_left, turn_right, attack]

        # Multi-button deathmatch layout
        action_vector = []
        for btn in self.buttons:
            if btn == vzd.Button.ATTACK:
                action_vector.append(attack)
            elif btn == vzd.Button.TURN_LEFT:
                action_vector.append(turn_left)
            elif btn == vzd.Button.TURN_RIGHT:
                action_vector.append(turn_right)
            elif btn == vzd.Button.MOVE_FORWARD:
                action_vector.append(move_forward or self.persona == "aggressive")
            elif btn == vzd.Button.MOVE_LEFT_RIGHT_DELTA:
                action_vector.append(0.0)
            elif btn == vzd.Button.TURN_LEFT_RIGHT_DELTA:
                turn_delta = 0.0
                if turn_left:
                    turn_delta = -self.turn_speed
                elif turn_right:
                    turn_delta = self.turn_speed
                action_vector.append(turn_delta)
            elif btn == vzd.Button.LOOK_UP_DOWN_DELTA:
                action_vector.append(0.0)
            else:
                action_vector.append(False)

        return action_vector


def configure_game(args: argparse.Namespace, scenario_path: str) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(scenario_path)

    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(args.render)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_doom_skill(args.skill)
    game.set_sound_enabled(False)  # keep sound off during high-speed batch collection

    # Enable entity inspection
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(False)

    # Exhaustive direct game variables
    game.clear_available_game_variables()
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
        game.add_available_game_variable(gv)

    game.init()
    return game


def run_batch_collection(args: argparse.Namespace) -> None:
    scenario_path = resolve_scenario_path(args.scenario)
    logger.info(f"Target Scenario: {scenario_path}")
    logger.info(f"Execution Mode: {'Interactive Visual Window' if args.render else 'Headless High-Speed Simulation (~350 FPS)'}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recorder = TelemetryRecorder(filepath=output_path, fieldnames=TELEMETRY_COLUMNS)
    recorder.open(mode="w")

    personas_to_run = (
        ["novice", "intermediate", "aggressive"] if args.persona == "all" else [args.persona]
    )

    game = configure_game(args, scenario_path)
    available_buttons = game.get_available_buttons()

    feature_extractor = FeatureExtractor(ticrate=vzd.DEFAULT_TICRATE)
    skill_estimator = SkillEstimator()

    total_episodes_run = 0
    start_wall_time = time.time()
    total_tics_simulated = 0

    print("=" * 80)
    logger.info(f"Starting batch simulation across personas: {personas_to_run}")

    try:
        episode_global_index = 0
        for persona in personas_to_run:
            logger.info(f">>> Simulating Persona: [{persona.upper()}] ({args.episodes} episodes) <<<")
            agent = SimulatedPlayerAgent(persona, available_buttons)

            for ep in range(1, args.episodes + 1):
                episode_global_index += 1
                game.new_episode()
                feature_extractor.reset()
                skill_estimator.reset()

                ep_tics = 0
                ep_start_t = time.time()

                while not game.is_episode_finished():
                    state = game.get_state()
                    if state is None:
                        break

                    g_vars = state.game_variables
                    # Agent computes next action based on world state
                    action = agent.compute_action(state, g_vars, Path(scenario_path).name)
                    game.make_action(action)
                    ep_tics += 1
                    total_tics_simulated += 1

                    reward = game.get_last_reward()

                    # Sample telemetry at configured interval
                    if ep_tics % args.interval_tics == 0:
                        now_timestamp = time.time()
                        current_tic = state.number

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

                        raw_frame = RawTelemetryFrame(
                            episode=episode_global_index,
                            tic=current_tic,
                            timestamp=now_timestamp,
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
                            current_doom_skill=game.get_doom_skill(),
                            last_reward=reward,
                            enemy_count=enemy_count,
                            item_count=item_count,
                        )

                        derived = feature_extractor.compute_derived_metrics(
                            raw=raw_frame,
                            skill_estimate=skill_estimator.skill_score,
                        )

                        skill_estimator.update(
                            health=health,
                            damage_dealt=damagecount,
                            damage_taken=damage_taken,
                            killcount=killcount,
                            deathcount=deathcount,
                            accuracy=derived.accuracy_estimate,
                        )
                        derived.player_skill_estimate = skill_estimator.skill_score
                        feature_extractor.update_history(raw_frame, derived)

                        row_data: Dict[str, Any] = {
                            "episode": episode_global_index,
                            "tic": current_tic,
                            "timestamp": round(now_timestamp, 4),
                            "elapsed_time_sec": round(derived.elapsed_time_sec, 2),
                            "health": round(health, 1),
                            "armor": round(armor, 1),
                            "ammo": round(ammo, 1),
                            "killcount": int(killcount),
                            "deathcount": int(deathcount),
                            "hitcount": int(hitcount),
                            "hits_taken": int(hits_taken),
                            "damagecount": int(damagecount),
                            "damage_taken": int(damage_taken),
                            "is_dead": int(is_dead),
                            "selected_weapon": int(selected_weapon),
                            "position_x": round(pos_x, 2),
                            "position_y": round(pos_y, 2),
                            "position_z": round(pos_z, 2),
                            "velocity_x": round(vel_x, 2),
                            "velocity_y": round(vel_y, 2),
                            "velocity_z": round(vel_z, 2),
                            "attack_ready": int(attack_ready),
                            "current_doom_skill": game.get_doom_skill(),
                            "last_reward": round(reward, 3),
                            "enemy_count": enemy_count,
                            "item_count": item_count,
                            "survival_time_sec": round(derived.survival_time_sec, 2),
                            "kill_rate": round(derived.kill_rate, 3),
                            "death_rate": round(derived.death_rate, 3),
                            "damage_dealt_rate": round(derived.damage_dealt_rate, 2),
                            "damage_received_rate": round(derived.damage_received_rate, 2),
                            "shots_fired_estimate": derived.shots_fired_estimate,
                            "accuracy_estimate": round(derived.accuracy_estimate, 3),
                            "player_speed": round(derived.player_speed, 2),
                            "recent_performance_trend": round(derived.recent_performance_trend, 3),
                            "aggression_score": round(derived.aggression_score, 3),
                            "player_skill_estimate": round(derived.player_skill_estimate, 3),
                        }
                        recorder.record_row(row_data)

                ep_duration = time.time() - ep_start_t
                sim_speed = ep_tics / max(0.001, ep_duration)
                final_kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                final_hp = float(game.get_game_variable(vzd.GameVariable.HEALTH))
                logger.info(
                    f"[{persona.upper()}] Ep {ep}/{args.episodes} completed | "
                    f"Tics: {ep_tics} ({sim_speed:.0f} FPS) | "
                    f"Health: {final_hp:.1f} | Kills: {final_kills} | "
                    f"Skill Tier: {skill_estimator.tier.value} ({skill_estimator.skill_score:.2f})"
                )
                total_episodes_run += 1

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user. Finalizing data...")
    finally:
        recorder.close()
        game.close()
        total_time = time.time() - start_wall_time
        overall_fps = total_tics_simulated / max(0.001, total_time)
        print("=" * 80)
        logger.info(f"Batch collection finished successfully!")
        logger.info(f"Simulated {total_episodes_run} episodes ({total_tics_simulated} tics) in {total_time:.2f}s ({overall_fps:.1f} FPS)!")
        logger.info(f"Recorded {recorder.row_count} telemetry rows to:")
        logger.info(f"  --> {output_path.resolve()}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_batch_collection(cli_args)
