#!/usr/bin/env python3
"""Run ViZDoom with the Rule-Based Dynamic Difficulty Adjustment (DDA) Director active.

Serves as Experimental Condition B (Rule-Based DDA Baseline) to compare against
Fixed Difficulty (Condition A) and the upcoming DQN Game Director (Condition C).

The Rule-Based Director monitors player vitals (health, ammo, kill rate, accuracy)
every K seconds and issues real-time game modulations (spawning resources, pacing enemies).

Usage:
    # Human interactive play with Rule-Based DDA active (spectator mode with HUD & sound)
    python scripts/run_rule_based_dda.py --mode human --scenario defend_the_center.cfg

    # Automated bot test with Rule-Based DDA active
    python scripts/run_rule_based_dda.py --mode bot --scenario defend_the_center.cfg --episodes 2

    # Run in Deathmatch
    python scripts/run_rule_based_dda.py --mode human --scenario deathmatch.cfg
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import vizdoom as vzd

from src.dda.difficulty import DifficultyState, DDAAction, EngineActuator
from src.dda.rule_based import RuleBasedDDAController
from src.dda.overlay import DirectorHUDOverlay
from src.player_model.features import (
    TELEMETRY_COLUMNS,
    RawTelemetryFrame,
    DerivedMetrics,
    FeatureExtractor,
)
from src.player_model.skill_estimator import SkillEstimator
from src.utils.logger import get_logger, TelemetryRecorder
from scripts.play_human import resolve_scenario_path, count_level_entities
from scripts.collect_telemetry import SimulatedPlayerAgent

logger = get_logger("RuleBasedDDA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ViZDoom with Rule-Based Dynamic Difficulty Adjustment active"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="human",
        choices=["human", "bot"],
        help="Player control mode: 'human' (interactive spectator) or 'bot' (simulated player). Default: human",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="defend_the_center.cfg",
        help="Scenario config (e.g., defend_the_center.cfg, deathmatch.cfg). Default: defend_the_center.cfg",
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
        default=5.0,
        help="Director evaluation interval in seconds. Default: 5.0",
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
        default=str(PROJECT_ROOT / "data" / "telemetry" / "rule_based_session.csv"),
        help="Output CSV path. Default: data/telemetry/rule_based_session.csv",
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
    parser.add_argument(
        "--overlay",
        action="store_true",
        default=True,
        help="Display real-time AI Director HUD overlay showing difficulty treatment (Easy/Medium/Hard), player skill assessment, and pacing actions. Default: True",
    )
    parser.add_argument(
        "--no-overlay",
        dest="overlay",
        action="store_false",
        help="Disable AI Director HUD overlay",
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

        # Ensure mouse aiming/look and standard FPS controls are available regardless of scenario config
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


def run_rule_based_session(args: argparse.Namespace) -> None:
    scenario_path = resolve_scenario_path(args.scenario)
    logger.info("=" * 80)
    logger.info("Starting Rule-Based Dynamic Difficulty Adjustment Session")
    logger.info(f"Scenario: {scenario_path} | Mode: {args.mode.upper()}")
    logger.info(f"Director Decision Interval: {args.interval_sec}s (~{int(args.interval_sec * vzd.DEFAULT_TICRATE)} tics)")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Columns include standard telemetry plus Director actions
    session_columns = TELEMETRY_COLUMNS + ["director_action", "director_actuation"]
    recorder = TelemetryRecorder(filepath=output_path, fieldnames=session_columns)
    recorder.open(mode="w")

    game = configure_game(args, scenario_path)
    available_buttons = game.get_available_buttons()

    feature_extractor = FeatureExtractor(ticrate=vzd.DEFAULT_TICRATE)
    skill_estimator = SkillEstimator()
    controller = RuleBasedDDAController()
    difficulty_state = DifficultyState(doom_skill=args.skill)
    actuator = EngineActuator(game)

    # Initialize Live Director HUD Overlay Banner
    overlay = (
        DirectorHUDOverlay(
            standalone=True,
            game=game,
            title=f"ViZDoom Rule-Based Director - {'Human' if args.mode == 'human' else 'Bot AI'} Play HUD",
        )
        if args.overlay
        else None
    )
    if overlay:
        logger.info(f"Live Rule-Based Director HUD overlay banner active (Mode: {args.mode.upper()}).")

    bot_agent = SimulatedPlayerAgent(args.persona, available_buttons) if args.mode == "bot" else None

    decision_interval_tics = int(args.interval_sec * vzd.DEFAULT_TICRATE)
    latest_action = DDAAction.NO_CHANGE
    latest_actuation_msg = "Initialized"

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

                # Player combat step (Human in spectator mode vs Automated Bot)
                if args.mode == "human":
                    game.advance_action()
                else:
                    action_vec = bot_agent.compute_action(state, g_vars, Path(scenario_path).name)
                    game.make_action(action_vec)

                tic_counter += 1
                reward = game.get_last_reward()

                # Extract telemetry snapshot
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
                # DDA AI DIRECTOR DECISION LOOP (Immediate emergency or periodic)
                # -------------------------------------------------------------
                is_emergency = (health < 35.0 or ammo <= 5.0) and (tic_counter - last_director_decision_tic >= 35)
                is_scheduled = (tic_counter - last_director_decision_tic >= decision_interval_tics)

                if is_emergency or is_scheduled:
                    last_director_decision_tic = tic_counter
                    latest_action = controller.select_action_from_telemetry(
                        raw=raw_frame,
                        derived=derived,
                        current_state=difficulty_state,
                    )
                    latest_actuation_msg = actuator.execute(
                        latest_action,
                        difficulty_state,
                        player_health=health,
                        player_ammo=ammo,
                    )

                    prefix = ">>> [EMERGENCY AID TRIGGERED] " if is_emergency else "[DIRECTOR] "
                    logger.info(
                        f"{prefix}@{derived.elapsed_time_sec:5.1f}s | "
                        f"Action: {latest_action.name:22s} | {latest_actuation_msg} | "
                        f"HP: {health:4.0f} | Ammo: {ammo:3.0f} | "
                        f"Skill: {skill_estimator.skill_score:0.2f} ({skill_estimator.tier.value})"
                    )

                # Record telemetry sample and update live HUD overlay
                if tic_counter % args.sample_tics == 0:
                    feature_extractor.update_history(raw_frame, derived)

                    if overlay is not None:
                        overlay.update(
                            health=health,
                            ammo=ammo,
                            killcount=int(killcount),
                            doom_skill=difficulty_state.doom_skill,
                            skill_score=skill_estimator.skill_score,
                            skill_tier=skill_estimator.tier.value,
                            latest_action=latest_action.name,
                            latest_actuation=latest_actuation_msg,
                            flow_reward=reward,
                            is_emergency=is_emergency if "is_emergency" in locals() else False,
                            game=game,
                        )

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
                    }
                    recorder.record_row(row_data)

            ep_duration = time.time() - ep_start_time
            logger.info(
                f"Episode {ep} finished | Duration: {ep_duration:.1f}s | "
                f"Final HP: {health} | Final Kills: {int(killcount)} | "
                f"Total Director Interventions Recorded: {recorder.row_count}"
            )

    except KeyboardInterrupt:
        logger.info("Session interrupted by user.")
    finally:
        if overlay is not None:
            overlay.close()
        recorder.close()
        game.close()
        logger.info("=" * 80)
        logger.info(f"Rule-Based DDA session finished. Telemetry and Director actions saved to:")
        logger.info(f"  --> {output_path.resolve()}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_rule_based_session(cli_args)
