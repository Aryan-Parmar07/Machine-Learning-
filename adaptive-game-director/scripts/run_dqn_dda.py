#!/usr/bin/env python3
"""Run ViZDoom with the Trained DQN Reinforcement Learning AI Game Director active.

Serves as Experimental Condition C (Learned Neural Network DDA) to compare against
Fixed Difficulty (Condition A) and Rule-Based DDA (Condition B).

Unlike static if/else heuristics, the Deep Q-Network evaluates 15 continuous player
telemetry dimensions (health velocity, kill rate, damage intake, accuracy, enemy pressure)
and dynamically issues pacing and resource commands to maximize the player's time
in Csikszentmihalyi's Flow Channel (35-75 HP).

Usage:
    # Interactive human play with DQN AI Director (spectator mode with freelook, HUD, and audio):
    python scripts/run_dqn_dda.py --mode human --scenario defend_the_center.cfg

    # Play in Deathmatch scenario:
    python scripts/run_dqn_dda.py --mode human --scenario deathmatch.cfg

    # Automated bot evaluation with DQN AI Director:
    python scripts/run_dqn_dda.py --mode bot --scenario defend_the_center.cfg --persona intermediate --episodes 3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import vizdoom as vzd

from src.agent.dqn import DQNAgent
from src.dda.difficulty import DifficultyState, DDAAction, EngineActuator
from src.player_model.features import (
    TELEMETRY_COLUMNS,
    RawTelemetryFrame,
    DerivedMetrics,
    FeatureExtractor,
)
from src.player_model.skill_estimator import SkillEstimator
from src.rewards.reward_functions import FlowStateReward
from src.utils.logger import get_logger, TelemetryRecorder
from scripts.play_human import resolve_scenario_path, count_level_entities
from scripts.collect_telemetry import SimulatedPlayerAgent

logger = get_logger("DQNGameDirector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViZDoom with Trained DQN AI Game Director active (Learned DDA)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "bot"],
        help="Player control mode: 'human' (interactive spectator) or 'bot' (automated bot). Default: human",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="defend_the_center.cfg",
        help="ViZDoom scenario (e.g., defend_the_center.cfg, deathmatch.cfg). Default: defend_the_center.cfg",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "final" / "director_dqn.pt"),
        help="Path to trained PyTorch DQN checkpoint. Default: models/final/director_dqn.pt",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to play. Default: 1",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=3.0,
        help="Director decision interval in seconds. Default: 3.0",
    )
    parser.add_argument(
        "--sample-tics",
        type=int,
        default=5,
        help="Telemetry logging frequency in game tics. Default: 5",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "telemetry" / "dqn_session.csv"),
        help="Output CSV path. Default: data/telemetry/dqn_session.csv",
    )
    parser.add_argument(
        "--skill",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Initial Doom skill level (1 to 5). Default: 3",
    )
    parser.add_argument(
        "--persona",
        type=str,
        default="intermediate",
        choices=["novice", "intermediate", "aggressive", "random"],
        help="Bot persona (only used if --mode bot). Default: intermediate",
    )
    parser.add_argument(
        "--sound",
        action="store_true",
        default=True,
        help="Enable game audio. Default: True",
    )
    parser.add_argument(
        "--no-sound",
        dest="sound",
        action="store_false",
        help="Disable game audio",
    )
    return parser.parse_args()


def configure_game(args: argparse.Namespace, scenario_path: str) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(scenario_path)

    game.set_screen_resolution(vzd.ScreenResolution.RES_800X600)
    game.set_window_visible(True)
    game.set_doom_skill(args.skill)

    if args.mode == "human":
        game.set_mode(vzd.Mode.SPECTATOR)
        game.add_game_args("+freelook 1")
        game.set_sound_enabled(args.sound)
        game.set_render_hud(True)
        game.set_render_crosshair(True)
        game.set_render_messages(True)
        game.set_render_screen_flashes(True)

        # Inject mouse freelook & movement controls if omitted by scenario config
        existing_buttons = set(game.get_available_buttons())
        for btn in [
            vzd.Button.TURN_LEFT_RIGHT_DELTA,
            vzd.Button.LOOK_UP_DOWN_DELTA,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
        ]:
            if btn not in existing_buttons:
                game.add_available_button(btn)
    else:
        game.set_mode(vzd.Mode.PLAYER)
        game.set_sound_enabled(False)
        game.set_render_hud(False)

    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(False)

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


def run_dqn_session(args: argparse.Namespace) -> None:
    scenario_path = resolve_scenario_path(args.scenario)
    logger.info("=" * 80)
    logger.info("Starting Learned DQN Reinforcement Learning AI Game Director Session")
    logger.info(f"Scenario: {scenario_path} | Mode: {args.mode.upper()}")
    logger.info(f"Model Checkpoint: {args.model_path}")
    logger.info(f"Director Interval: {args.interval_sec}s (~{int(args.interval_sec * vzd.DEFAULT_TICRATE)} tics)")

    # 1. Initialize RL DQN Agent and load learned weights
    agent = DQNAgent(state_dim=15, action_dim=7)
    model_path = Path(args.model_path)
    if not model_path.exists():
        fallback_path = PROJECT_ROOT / "models" / "checkpoints" / "best_director_dqn.pt"
        if fallback_path.exists():
            logger.warning(f"Specified model not found at {model_path}. Using best checkpoint at {fallback_path}")
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"DQN weights not found at {model_path} or {fallback_path}")

    agent.load_checkpoint(model_path)
    logger.info(f"Successfully loaded trained DQN Director weights from {model_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    session_columns = TELEMETRY_COLUMNS + ["director_action", "director_actuation", "flow_reward"]
    recorder = TelemetryRecorder(filepath=output_path, fieldnames=session_columns)
    recorder.open(mode="w")

    game = configure_game(args, scenario_path)
    available_buttons = game.get_available_buttons()

    feature_extractor = FeatureExtractor(ticrate=vzd.DEFAULT_TICRATE)
    skill_estimator = SkillEstimator()
    reward_fn = FlowStateReward()
    difficulty_state = DifficultyState(doom_skill=args.skill)
    actuator = EngineActuator(game)

    bot_agent = SimulatedPlayerAgent(args.persona, available_buttons) if args.mode == "bot" else None

    decision_interval_tics = int(args.interval_sec * vzd.DEFAULT_TICRATE)
    latest_action = DDAAction.NO_CHANGE
    latest_actuation_msg = "Initialized"
    latest_reward = 0.0

    flow_state_frames = 0
    total_director_frames = 0

    try:
        for ep in range(1, args.episodes + 1):
            logger.info(f"\n--- Episode {ep}/{args.episodes} ---")
            game.new_episode()
            try:
                game.send_game_command("sv_cheats 1")
            except Exception:
                pass
            feature_extractor.reset()
            skill_estimator.reset()
            difficulty_state = DifficultyState(doom_skill=args.skill)

            tic_counter = 0
            ep_start_time = time.time()
            last_director_decision_tic = 0

            while not game.is_episode_finished():
                state = game.get_state()
                if state is None:
                    break

                g_vars = state.game_variables

                # Human plays freely via Spectator mode; Bot acts automatically
                if args.mode == "human":
                    game.advance_action()
                else:
                    action_vec = bot_agent.compute_action(state, g_vars, Path(scenario_path).name)
                    game.make_action(action_vec)

                tic_counter += 1
                reward = game.get_last_reward()

                current_tic = state.number
                now_timestamp = time.time()

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
                    episode=ep,
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
                    current_doom_skill=difficulty_state.doom_skill,
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

                # -------------------------------------------------------------
                # LEARNED DQN NEURAL NETWORK AI DIRECTOR DECISION LOOP
                # -------------------------------------------------------------
                # Immediate interrupt if player is in critical danger (<35 HP) or out of ammo (<=5)
                is_emergency = (health < 35.0 or ammo <= 5.0) and (tic_counter - last_director_decision_tic >= 35)
                is_scheduled = (tic_counter - last_director_decision_tic >= decision_interval_tics)

                if is_emergency or is_scheduled:
                    last_director_decision_tic = tic_counter

                    # Calculate Flow Reward for telemetry metrics
                    latest_reward = reward_fn.calculate_reward(
                        raw=raw_frame,
                        derived=derived,
                        step_duration_sec=args.interval_sec,
                    )

                    # 1. Build 15-dimensional state representation
                    state_vec = feature_extractor.extract_normalized_state_vector(
                        raw=raw_frame,
                        derived=derived,
                    )

                    # 2. DQN forward pass (greedy policy, epsilon=0.0)
                    action_idx = agent.select_action(state_vec, epsilon=0.0)
                    latest_action = DDAAction(action_idx)

                    # 3. Actuate decision into ViZDoom engine with safety guardrail
                    latest_actuation_msg = actuator.execute(
                        latest_action,
                        difficulty_state,
                        player_health=health,
                        player_ammo=ammo,
                    )

                    total_director_frames += 1
                    in_flow = 35.0 <= health <= 75.0
                    if in_flow:
                        flow_state_frames += 1

                    flow_tag = "[FLOW CHANNEL]" if in_flow else ("[DISTRESS]" if health < 35 else "[COMFORT]")
                    prefix = ">>> [EMERGENCY AID TRIGGERED] " if is_emergency else "[DQN DIRECTOR] "
                    logger.info(
                        f"{prefix}@{derived.elapsed_time_sec:5.1f}s | "
                        f"Action: {latest_action.name:22s} | {latest_actuation_msg} | "
                        f"HP: {health:4.0f} ({flow_tag:14s}) | Ammo: {ammo:3.0f} | "
                        f"Kills: {killcount:2.0f} | Flow Reward: {latest_reward:+.2f}"
                    )

                # Periodic telemetry logging to CSV
                if tic_counter % args.sample_tics == 0:
                    feature_extractor.update_history(raw_frame, derived)

                    row_data: Dict[str, Any] = {
                        "episode": ep,
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
                        "current_doom_skill": difficulty_state.doom_skill,
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
                        "director_action": latest_action.name,
                        "director_actuation": latest_actuation_msg,
                        "flow_reward": round(latest_reward, 3),
                    }
                    recorder.record_row(row_data)

            # End of Episode summary
            ep_time = time.time() - ep_start_time
            flow_pct = (flow_state_frames / max(1, total_director_frames)) * 100.0
            logger.info(f"Episode {ep} Finished | Duration: {ep_time:.1f}s | Flow Channel Ratio: {flow_pct:.1f}%")

    except KeyboardInterrupt:
        logger.info("\nSession interrupted by user.")
    finally:
        recorder.close()
        game.close()
        logger.info(f"Session telemetry and DQN decisions saved to: {output_path}")


def main():
    args = parse_args()
    run_dqn_session(args)


if __name__ == "__main__":
    main()
