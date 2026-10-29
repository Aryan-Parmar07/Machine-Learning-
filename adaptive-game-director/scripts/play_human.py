#!/usr/bin/env python3
"""Milestone 1: Human-Play Telemetry Pipeline for ViZDoom.

Launches ViZDoom in SPECTATOR mode to allow human keyboard and mouse play,
records native and derived telemetry at regular intervals, stream-writes to CSV,
and displays a live console dashboard.

Usage:
    python scripts/play_human.py
    python scripts/play_human.py --scenario deathmatch.cfg --episodes 2 --interval-tics 5
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to sys.path so 'src' imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import vizdoom as vzd
import numpy as np

from src.player_model.features import (
    TELEMETRY_COLUMNS,
    RawTelemetryFrame,
    DerivedMetrics,
    FeatureExtractor,
)
from src.player_model.skill_estimator import SkillEstimator
from src.dda.difficulty import DifficultyState, DDAAction, EngineActuator
from src.dda.rule_based import RuleBasedDDAController
from src.dda.overlay import DirectorHUDOverlay
from src.agent.dqn import DQNAgent
from src.utils.logger import get_logger, TelemetryRecorder


logger = get_logger("PlayHuman")

# Known monster actor names across standard ViZDoom / Doom scenarios
KNOWN_MONSTER_NAMES = {
    "demon", "marinechainsawvzd", "chaingunguy", "zombieman", "shotgunguy",
    "imp", "doomimp", "cacodemon", "hellknight", "baronofhell", "lostsoul",
    "revenant", "mancubus", "arachnotron", "archvile", "cyberdemon", "spidermastermind"
}

# Known pickup/item names
KNOWN_ITEM_NAMES = {
    "medikit", "stimpack", "healthbonus", "armorbonus", "greenarmor", "bluearmor",
    "clipbox", "shellbox", "rocketbox", "cellpack", "clip", "shells", "rocketammo", "cell",
    "chainsaw", "shotgun", "supershotgun", "chaingun", "rocketlauncher", "plasmarifle", "bfg9000",
    "backpack", "soulsphere", "megasphere", "invulnerabilitysphere", "blur-sphere"
}


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for human gameplay session."""
    parser = argparse.ArgumentParser(
        description="ViZDoom Human Gameplay & Telemetry Collector (SPECTATOR Mode)"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="deathmatch.cfg",
        help="ViZDoom scenario filename (in vzd.scenarios_path) or absolute path to a .cfg file. "
             "Default: deathmatch.cfg",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to play. Default: 1",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "telemetry" / "human_session.csv"),
        help="Output CSV file path. Default: data/telemetry/human_session.csv",
    )
    parser.add_argument(
        "--interval-tics",
        type=int,
        default=5,
        help="Sampling interval in game tics (35 tics = 1 sec; 5 tics ~ 7Hz). Default: 5",
    )
    parser.add_argument(
        "--skill",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Initial Doom skill level (1: Very Easy to 5: Nightmare). Default: 3",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="RES_800X600",
        help="Screen resolution (e.g., RES_640X480, RES_800X600). Default: RES_800X600",
    )
    parser.add_argument(
        "--sound",
        action="store_true",
        default=True,
        help="Enable in-game audio effects and sound. Default: True",
    )
    parser.add_argument(
        "--no-sound",
        dest="sound",
        action="store_false",
        help="Disable in-game audio/sound effects",
    )
    parser.add_argument(
        "--hud",
        action="store_true",
        default=True,
        help="Render in-game HUD (health, armor, ammo bar). Default: True",
    )
    parser.add_argument(
        "--no-hud",
        dest="hud",
        action="store_false",
        help="Disable in-game HUD display",
    )
    parser.add_argument(
        "--crosshair",
        action="store_true",
        default=True,
        help="Render aiming crosshair. Default: True",
    )
    parser.add_argument(
        "--no-crosshair",
        dest="crosshair",
        action="store_false",
        help="Disable aiming crosshair",
    )
    parser.add_argument(
        "--director",
        type=str,
        default="dqn",
        choices=["dqn", "rule", "none"],
        help="AI Game Director controller: 'dqn' (learned RL), 'rule' (heuristic DDA), or 'none' (passive recording). Default: dqn",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "final" / "director_dqn.pt"),
        help="Path to trained PyTorch DQN checkpoint. Default: models/final/director_dqn.pt",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=3.0,
        help="Director decision interval in seconds. Default: 3.0",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        default=True,
        help="Display real-time AI Director HUD overlay showing difficulty treatment (Easy/Medium/Hard), player skill tier, and pacing actions. Default: True",
    )
    parser.add_argument(
        "--no-overlay",
        dest="overlay",
        action="store_false",
        help="Disable AI Director HUD overlay",
    )
    return parser.parse_args()


def resolve_scenario_path(scenario_arg: str) -> str:
    """Finds scenario file either in working directory, project, or vzd.scenarios_path."""
    direct_path = Path(scenario_arg)
    if direct_path.exists():
        return str(direct_path.resolve())

    vzd_path = Path(vzd.scenarios_path) / scenario_arg
    if vzd_path.exists():
        return str(vzd_path.resolve())

    raise FileNotFoundError(
        f"Could not locate ViZDoom scenario '{scenario_arg}'. "
        f"Checked direct path and ViZDoom scenarios folder: {vzd.scenarios_path}"
    )


def count_level_entities(state: vzd.GameState) -> Tuple[int, int]:
    """Inspects state.objects to count active enemies and items.

    Args:
        state: ViZDoom GameState.

    Returns:
        Tuple of (enemy_count, item_count).
    """
    if not hasattr(state, "objects") or state.objects is None:
        return 0, 0

    enemy_count = 0
    item_count = 0

    # In Deathmatch, bots are named 'DoomPlayer' with id > 0
    # In single-player monster scenarios, enemies have monster class names
    for obj in state.objects:
        obj_name_lower = obj.name.lower()
        if obj.id == 0 and "player" in obj_name_lower:
            # Human player character
            continue

        if "player" in obj_name_lower:
            # Bot opponent in deathmatch
            enemy_count += 1
        elif obj_name_lower in KNOWN_MONSTER_NAMES or "vzd" in obj_name_lower:
            enemy_count += 1
        elif obj_name_lower in KNOWN_ITEM_NAMES or "bonus" in obj_name_lower or "box" in obj_name_lower:
            item_count += 1
        elif "marine" in obj_name_lower or "demon" in obj_name_lower or "guy" in obj_name_lower:
            enemy_count += 1

    return enemy_count, item_count


def configure_game(args: argparse.Namespace) -> vzd.DoomGame:
    """Initializes and configures the ViZDoom engine in SPECTATOR mode."""
    scenario_path = resolve_scenario_path(args.scenario)
    logger.info(f"Loading ViZDoom scenario: {scenario_path}")

    game = vzd.DoomGame()
    game.load_config(scenario_path)

    # Resolution
    res_attr = getattr(vzd.ScreenResolution, args.resolution, vzd.ScreenResolution.RES_800X600)
    game.set_screen_resolution(res_attr)

    # Human gameplay configuration
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.add_game_args("+freelook 1")
    game.set_sound_enabled(args.sound)
    game.set_doom_skill(args.skill)
    logger.info(f"Audio/Sound enabled: {args.sound}")

    # Visual rendering options for human player feedback
    game.set_render_hud(args.hud)
    game.set_render_crosshair(args.crosshair)
    game.set_render_messages(True)
    game.set_render_screen_flashes(True)
    logger.info(f"HUD enabled: {args.hud} | Crosshair enabled: {args.crosshair}")

    # Ensure mouse look and standard FPS controls are available regardless of scenario config
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

    # Enable full object inspection for live entity counting
    game.set_objects_info_enabled(True)
    game.set_sectors_info_enabled(False)

    # Configure exhaustive set of direct GameVariables
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
    ]
    for gv in game_vars:
        game.add_available_game_variable(gv)

    game.init()
    logger.info("ViZDoom initialized successfully in SPECTATOR mode.")
    return game


def run_session(args: argparse.Namespace) -> None:
    """Runs the human gameplay loop and collects telemetry."""
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    game = configure_game(args)
    feature_extractor = FeatureExtractor(ticrate=vzd.DEFAULT_TICRATE)
    skill_estimator = SkillEstimator()
    session_columns = TELEMETRY_COLUMNS + ["director_action", "director_actuation"]
    recorder = TelemetryRecorder(filepath=output_path, fieldnames=session_columns)
    recorder.open(mode="w")

    # Initialize AI Game Director if enabled
    director_active = args.director != "none"
    actuator = EngineActuator(game) if director_active else None
    diff_state = DifficultyState(doom_skill=args.skill)
    dqn_agent = None
    rule_controller = None

    if args.director == "dqn":
        dqn_agent = DQNAgent(state_dim=15, action_dim=7)
        model_path = Path(args.model_path)
        if not model_path.exists():
            fallback_path = PROJECT_ROOT / "models" / "checkpoints" / "best_director_dqn.pt"
            if fallback_path.exists():
                model_path = fallback_path
        if model_path.exists():
            dqn_agent.load_checkpoint(model_path)
            logger.info(f"Loaded trained DQN AI Game Director from: {model_path}")
        else:
            logger.warning(f"DQN weights not found at {model_path}. Falling back to Rule-Based DDA Director.")
            rule_controller = RuleBasedDDAController()
    elif args.director == "rule":
        rule_controller = RuleBasedDDAController()
        logger.info("Using Rule-Based DDA Game Director.")
    else:
        logger.info("Passive telemetry recording (AI Game Director disabled).")

    logger.info(f"Recording telemetry to: {output_path.resolve()}")
    logger.info("Controls: WASD/Arrows to move, Mouse to look/aim, Left Click or CTRL to shoot.")
    logger.info("Press ESC or close window to end early.")
    print("=" * 80)

    # Initialize live HUD Overlay if requested
    overlay = (
        DirectorHUDOverlay(standalone=True, game=game, title="ViZDoom AI Director - Human Play HUD")
        if (args.overlay and director_active)
        else None
    )
    if overlay:
        logger.info("Live AI Director HUD overlay banner initialized (Always-on-Top).")

    total_episodes = args.episodes
    last_print_time = 0.0
    print_interval_sec = 0.5  # Throttle terminal updates to 2 Hz for responsiveness
    decision_interval_tics = int(args.interval_sec * vzd.DEFAULT_TICRATE)

    try:
        for ep in range(1, total_episodes + 1):
            logger.info(f"Starting Episode {ep}/{total_episodes} (Doom Skill: {game.get_doom_skill()})")
            game.new_episode()
            try:
                game.send_game_command("sv_cheats 1")
            except Exception:
                pass
            feature_extractor.reset()
            skill_estimator.reset()
            diff_state = DifficultyState(doom_skill=args.skill)

            ep_start_time = time.time()
            tic_counter = 0
            last_director_decision_tic = 0
            latest_action_name = "NO_CHANGE"
            latest_actuation = "None"

            while not game.is_episode_finished():
                state = game.get_state()
                if state is None:
                    break

                # Advance 1 tic of human action
                game.advance_action()
                tic_counter += 1
                reward = game.get_last_reward()

                # Sample telemetry at configured interval
                if tic_counter % args.interval_tics == 0:
                    current_tic = state.number
                    now_timestamp = time.time()
                    g_vars = state.game_variables

                    # Extract Direct ViZDoom Variables safely
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

                    # Count active entities in the level
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
                        current_doom_skill=game.get_doom_skill(),
                        last_reward=reward,
                        enemy_count=enemy_count,
                        item_count=item_count,
                    )

                    # Compute derived metrics
                    derived = feature_extractor.compute_derived_metrics(
                        raw=raw_frame,
                        skill_estimate=skill_estimator.skill_score,
                    )

                    # Update online skill estimator
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
                    if director_active:
                        is_emergency = (health < 35.0 or ammo <= 5.0) and (tic_counter - last_director_decision_tic >= 35)
                        is_scheduled = (tic_counter - last_director_decision_tic >= decision_interval_tics)

                        if is_emergency or is_scheduled:
                            last_director_decision_tic = tic_counter
                            if dqn_agent is not None:
                                state_vec = feature_extractor.extract_normalized_state_vector(raw_frame, derived)
                                action_idx = dqn_agent.select_action(state_vec, epsilon=0.0)
                                latest_action = DDAAction(action_idx)
                            else:
                                latest_action = rule_controller.select_action_from_telemetry(raw_frame, derived, diff_state)

                            latest_actuation = actuator.execute(
                                latest_action,
                                diff_state,
                                player_health=health,
                                player_ammo=ammo,
                            )
                            latest_action_name = latest_action.name

                            prefix = ">>> [EMERGENCY AID TRIGGERED] " if is_emergency else "[AI DIRECTOR] "
                            print(
                                f"\n{prefix}Action: {latest_action_name:20s} | "
                                f"{latest_actuation} | HP: {health:.0f} | Ammo: {ammo:.0f}"
                            )

                    # Update history
                    feature_extractor.update_history(raw_frame, derived)

                    # Update live HUD overlay
                    if overlay is not None:
                        is_emerg = is_emergency if (director_active and "is_emergency" in locals()) else False
                        overlay.update(
                            health=health,
                            ammo=ammo,
                            killcount=int(killcount),
                            doom_skill=game.get_doom_skill(),
                            skill_score=skill_estimator.skill_score,
                            skill_tier=skill_estimator.tier.value,
                            latest_action=latest_action_name,
                            latest_actuation=latest_actuation,
                            flow_reward=reward,
                            is_emergency=is_emerg,
                            game=game,
                        )

                    # Assemble complete CSV row
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
                        "director_action": latest_action_name,
                        "director_actuation": latest_actuation,
                    }
                    recorder.record_row(row_data)

                    # Live terminal dashboard (throttled)
                    if now_timestamp - last_print_time >= print_interval_sec:
                        last_print_time = now_timestamp
                        elapsed = derived.elapsed_time_sec
                        print(
                            f"\r[Ep {ep}] Time: {elapsed:5.1f}s | "
                            f"Health: {health:5.1f} | Ammo: {ammo:3.0f} | "
                            f"Kills: {int(killcount):2d} | Deaths: {int(deathcount):2d} | "
                            f"Enemies: {enemy_count:2d} | Reward: {reward:+.2f} | "
                            f"Skill: {skill_estimator.skill_score:0.2f} ({skill_estimator.tier.value})",
                            end="",
                            flush=True,
                        )

            print()  # newline after episode
            ep_time = time.time() - ep_start_time
            logger.info(
                f"Episode {ep} finished | Duration: {ep_time:.1f}s | "
                f"Total Reward: {game.get_total_reward():.2f} | "
                f"Total Rows Saved: {recorder.row_count}"
            )

    except KeyboardInterrupt:
        print()
        logger.info("Session interrupted by user (KeyboardInterrupt). Finalizing data...")
    finally:
        if overlay is not None:
            overlay.close()
        recorder.close()
        game.close()
        logger.info(f"Session closed successfully. Telemetry saved ({recorder.row_count} rows) to:")
        logger.info(f"  --> {output_path.resolve()}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_session(cli_args)
