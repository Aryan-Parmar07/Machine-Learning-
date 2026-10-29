#!/usr/bin/env python3
"""Evaluation and Comparative Benchmark Pipeline for the AI Game Director.

Empirically benchmarks and compares three experimental regimes:
  - Condition A: Fixed Difficulty (Static baseline across Doom Skill 1 [Easy], Skill 3 [Normal], Skill 5 [Hard])
  - Condition B: Rule-Based DDA (Heuristic baseline using Flow State hierarchy)
  - Condition C: DQN AI Game Director (Learned neural network policy with safety guardrails)

Key Metrics Computed:
  - Flow Channel Retention Rate (% of time player is maintained in [35, 75] HP)
  - Distress Time Ratio (% of time player is < 35 HP)
  - Boredom / Comfort Time Ratio (% of time player is > 75 HP without threat)
  - Survival Rate & Average Survival Duration (sec)
  - Ammo Starvation Count (occurrences of ammo <= 5)
  - Kill Rate & Combat Engagement
  - Flow State Reward (mean & cumulative)

Outputs:
  - CSV: results/evaluation_benchmark.csv
  - JSON: results/benchmark_summary.json
  - Visual: results/plots/dda_comparative_benchmark.png

Usage:
    # Run full comparative benchmark (5 episodes per condition across novice & intermediate personas)
    python scripts/evaluate.py --episodes 5 --scenario defend_the_center.cfg

    # Fast evaluation run
    python scripts/evaluate.py --episodes 3 --conditions fixed_normal rule_based dqn
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import vizdoom as vzd

from src.agent.dqn import DQNAgent
from src.dda.difficulty import DifficultyState, DDAAction, EngineActuator
from src.dda.rule_based import RuleBasedDDAController
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

logger = get_logger("EvaluateDDA")

SUPPORTED_CONDITIONS = ["fixed_easy", "fixed_normal", "fixed_hard", "rule_based", "dqn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate & Benchmark DDA Controllers")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["fixed_easy", "fixed_normal", "fixed_hard", "rule_based", "dqn"],
        choices=SUPPORTED_CONDITIONS,
        help="Evaluation conditions to benchmark",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="defend_the_center.cfg",
        help="ViZDoom scenario config (e.g. defend_the_center.cfg, deadly_corridor.cfg, deathmatch.cfg)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per condition. Default: 5",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=3.0,
        help="Director evaluation interval in seconds. Default: 3.0",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(PROJECT_ROOT / "models" / "final" / "director_dqn.pt"),
        help="Path to trained DQN model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save benchmark metrics and figures",
    )
    return parser.parse_args()


def configure_headless_eval_game(scenario_path: str, initial_skill: int) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(scenario_path)
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_render_hud(False)
    game.set_doom_skill(initial_skill)
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
    ]
    for gv in game_vars:
        game.add_available_game_variable(gv)

    game.init()
    try:
        game.send_game_command("sv_cheats 1")
    except Exception:
        pass
    return game


def run_single_episode(
    condition: str,
    ep_idx: int,
    persona: str,
    game: vzd.DoomGame,
    scenario_path: str,
    interval_sec: float,
    dqn_agent: Optional[DQNAgent],
    rule_controller: Optional[RuleBasedDDAController],
    initial_skill: int,
) -> Dict[str, Any]:
    available_buttons = game.get_available_buttons()
    bot_agent = SimulatedPlayerAgent(persona, available_buttons)
    feature_extractor = FeatureExtractor(ticrate=vzd.DEFAULT_TICRATE)
    skill_estimator = SkillEstimator()
    reward_fn = FlowStateReward()
    difficulty_state = DifficultyState(doom_skill=initial_skill)
    actuator = EngineActuator(game)

    decision_interval_tics = int(interval_sec * vzd.DEFAULT_TICRATE)

    game.set_doom_skill(initial_skill)
    game.new_episode()
    try:
        game.send_game_command("sv_cheats 1")
    except Exception:
        pass
    feature_extractor.reset()
    skill_estimator.reset()

    tic_counter = 0
    last_director_decision_tic = 0
    latest_action = DDAAction.NO_CHANGE
    latest_actuation = "Initialized"

    health_history = []
    ammo_history = []
    rewards = []
    action_counts = {a.name: 0 for a in DDAAction}

    flow_channel_tics = 0
    distress_tics = 0
    comfort_tics = 0
    ammo_starvation_events = 0
    was_starving = False

    while not game.is_episode_finished():
        state = game.get_state()
        if state is None:
            break

        g_vars = state.game_variables
        action_vec = bot_agent.compute_action(state, g_vars, Path(scenario_path).name)
        game.make_action(action_vec)
        tic_counter += 1

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

        enemy_count, item_count = count_level_entities(state.objects)

        raw_frame = RawTelemetryFrame(
            episode=ep_idx,
            tic=state.number,
            timestamp=time.time(),
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
            selected_weapon=float(g_vars[10]) if len(g_vars) > 10 else 0.0,
            pos_x=float(g_vars[11]) if len(g_vars) > 11 else 0.0,
            pos_y=float(g_vars[12]) if len(g_vars) > 12 else 0.0,
            pos_z=float(g_vars[13]) if len(g_vars) > 13 else 0.0,
            vel_x=float(g_vars[14]) if len(g_vars) > 14 else 0.0,
            vel_y=float(g_vars[15]) if len(g_vars) > 15 else 0.0,
            vel_z=float(g_vars[16]) if len(g_vars) > 16 else 0.0,
            attack_ready=bool(g_vars[17] > 0.0) if len(g_vars) > 17 else False,
            current_doom_skill=difficulty_state.doom_skill,
            last_reward=game.get_last_reward(),
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

        # Track vitals
        health_history.append(health)
        ammo_history.append(ammo)

        if 35.0 <= health <= 75.0:
            flow_channel_tics += 1
        elif health < 35.0:
            distress_tics += 1
        else:
            comfort_tics += 1

        if ammo <= 5.0 and not was_starving:
            ammo_starvation_events += 1
            was_starving = True
        elif ammo > 5.0:
            was_starving = False

        # -------------------------------------------------------------
        # DIRECTOR DECISION POINT (Every interval_sec)
        # -------------------------------------------------------------
        if tic_counter - last_director_decision_tic >= decision_interval_tics:
            last_director_decision_tic = tic_counter
            step_reward = reward_fn.calculate_reward(
                raw=raw_frame,
                derived=derived,
                step_duration_sec=interval_sec,
            )
            rewards.append(step_reward)

            if condition == "rule_based":
                latest_action = rule_controller.select_action_from_telemetry(
                    raw=raw_frame,
                    derived=derived,
                    current_state=difficulty_state,
                )
                latest_actuation = actuator.execute(
                    latest_action,
                    difficulty_state,
                    player_health=health,
                    player_ammo=ammo,
                )
            elif condition == "dqn":
                state_vec = feature_extractor.extract_normalized_state_vector(raw_frame, derived)
                action_idx = dqn_agent.select_action(state_vec, epsilon=0.0)
                latest_action = DDAAction(action_idx)
                latest_actuation = actuator.execute(
                    latest_action,
                    difficulty_state,
                    player_health=health,
                    player_ammo=ammo,
                )
            else:
                # Fixed baseline - no active dynamic adjustments
                latest_action = DDAAction.NO_CHANGE

            action_counts[latest_action.name] += 1

    total_tics = max(1, tic_counter)
    duration_sec = total_tics / vzd.DEFAULT_TICRATE
    survived = not is_dead and (health > 0)

    # Compute episode summary metrics
    flow_ratio = (flow_channel_tics / total_tics) * 100.0
    distress_ratio = (distress_tics / total_tics) * 100.0
    comfort_ratio = (comfort_tics / total_tics) * 100.0

    return {
        "condition": condition,
        "episode": ep_idx,
        "persona": persona,
        "scenario": Path(scenario_path).name,
        "survived": int(survived),
        "duration_sec": round(duration_sec, 2),
        "total_tics": total_tics,
        "final_health": round(health, 1),
        "mean_health": round(float(np.mean(health_history)), 1) if health_history else 0.0,
        "health_std": round(float(np.std(health_history)), 2) if health_history else 0.0,
        "flow_channel_ratio": round(flow_ratio, 2),
        "distress_ratio": round(distress_ratio, 2),
        "comfort_ratio": round(comfort_ratio, 2),
        "ammo_starvation_events": ammo_starvation_events,
        "total_kills": int(killcount),
        "kill_rate": round(float(killcount / max(1.0, duration_sec)), 3),
        "damage_dealt": round(damagecount, 1),
        "damage_taken": round(damage_taken, 1),
        "mean_reward": round(float(np.mean(rewards)), 3) if rewards else 0.0,
        "total_reward": round(float(np.sum(rewards)), 2) if rewards else 0.0,
        "action_counts": action_counts,
    }


def generate_benchmark_plots(df: pd.DataFrame, output_dir: Path) -> Path:
    """Generates a 4-panel publication-ready comparison plot."""
    plot_path = output_dir / "plots" / "dda_comparative_benchmark.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Adaptive Game Director: Comparative Benchmark Analysis", fontsize=16, fontweight="bold")

    # Order of conditions for clean aesthetic comparison
    cond_order = [c for c in ["fixed_easy", "fixed_normal", "fixed_hard", "rule_based", "dqn"] if c in df["condition"].unique()]
    palette = {
        "fixed_easy": "#4a90e2",
        "fixed_normal": "#50e3c2",
        "fixed_hard": "#e94e77",
        "rule_based": "#f5a623",
        "dqn": "#9013fe",
    }
    colors = [palette.get(c, "#333333") for c in cond_order]

    # Panel 1: Flow Channel Retention Ratio (%)
    flow_means = [df[df["condition"] == c]["flow_channel_ratio"].mean() for c in cond_order]
    flow_stds = [df[df["condition"] == c]["flow_channel_ratio"].std() for c in cond_order]
    axes[0, 0].bar(cond_order, flow_means, yerr=flow_stds, capsize=5, color=colors, alpha=0.85, edgecolor="black")
    axes[0, 0].set_title("Flow Channel Retention Ratio [35, 75 HP] (%) - Higher is Better", fontweight="bold")
    axes[0, 0].set_ylabel("Time in Flow Channel (%)")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Distress vs Boredom Breakdown
    distress_means = [df[df["condition"] == c]["distress_ratio"].mean() for c in cond_order]
    comfort_means = [df[df["condition"] == c]["comfort_ratio"].mean() for c in cond_order]
    x = np.arange(len(cond_order))
    width = 0.35
    axes[0, 1].bar(x - width/2, distress_means, width, label="Distress (<35 HP)", color="#d0021b", alpha=0.85)
    axes[0, 1].bar(x + width/2, comfort_means, width, label="Comfort (>75 HP)", color="#4a90e2", alpha=0.85)
    axes[0, 1].set_title("Negative Affect Time Ratios (%) - Lower is Better", fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(cond_order)
    axes[0, 1].set_ylabel("Time Ratio (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 3: Survival Duration & Rate
    dur_means = [df[df["condition"] == c]["duration_sec"].mean() for c in cond_order]
    axes[1, 0].bar(cond_order, dur_means, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 0].set_title("Mean Survival Duration (Seconds)", fontweight="bold")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 4: Mean Flow State Reward
    rew_means = [df[df["condition"] == c]["mean_reward"].mean() for c in cond_order]
    axes[1, 1].bar(cond_order, rew_means, color=colors, alpha=0.85, edgecolor="black")
    axes[1, 1].set_title("Mean Flow State Reward per Step", fontweight="bold")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = resolve_scenario_path(args.scenario)

    logger.info("=" * 80)
    logger.info("Starting Full Comparative DDA Benchmark Suite")
    logger.info(f"Scenario: {scenario_path} | Episodes Per Condition: {args.episodes}")
    logger.info(f"Target Conditions: {args.conditions}")

    # Load DQN Agent
    dqn_agent = None
    if "dqn" in args.conditions:
        dqn_agent = DQNAgent(state_dim=15, action_dim=7)
        model_path = Path(args.model_path)
        if not model_path.exists():
            fallback = PROJECT_ROOT / "models" / "checkpoints" / "best_director_dqn.pt"
            model_path = fallback if fallback.exists() else model_path
        dqn_agent.load_checkpoint(model_path)
        logger.info(f"Loaded DQN Director weights from: {model_path}")

    rule_controller = RuleBasedDDAController() if "rule_based" in args.conditions else None

    # Condition map -> Doom native skill level
    skill_map = {
        "fixed_easy": 1,
        "fixed_normal": 3,
        "fixed_hard": 5,
        "rule_based": 3,
        "dqn": 3,
    }

    # Evaluate across alternating personas (Novice, Intermediate, Aggressive)
    personas = ["novice", "intermediate", "aggressive"]
    all_results: List[Dict[str, Any]] = []

    for cond in args.conditions:
        initial_skill = skill_map[cond]
        logger.info(f"\n>>> Running Benchmark: Condition '{cond.upper()}' (Skill {initial_skill}) <<<")
        game = configure_headless_eval_game(scenario_path, initial_skill)

        try:
            for ep in range(1, args.episodes + 1):
                persona = personas[(ep - 1) % len(personas)]
                res = run_single_episode(
                    condition=cond,
                    ep_idx=ep,
                    persona=persona,
                    game=game,
                    scenario_path=scenario_path,
                    interval_sec=args.interval_sec,
                    dqn_agent=dqn_agent,
                    rule_controller=rule_controller,
                    initial_skill=initial_skill,
                )
                all_results.append(res)
                logger.info(
                    f"  [{cond:12s}] Ep {ep}/{args.episodes} ({persona:12s}) | "
                    f"Flow: {res['flow_channel_ratio']:5.1f}% | "
                    f"Dur: {res['duration_sec']:5.1f}s | "
                    f"Kills: {res['total_kills']:2d} | "
                    f"Starvations: {res['ammo_starvation_events']} | "
                    f"Mean Reward: {res['mean_reward']:+0.2f}"
                )
        finally:
            game.close()

    # Save detailed CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "evaluation_benchmark.csv"
    df.drop(columns=["action_counts"]).to_csv(csv_path, index=False)
    logger.info(f"\nSaved detailed evaluation metrics to: {csv_path}")

    # Generate summary JSON
    summary = {}
    for cond in args.conditions:
        sub = df[df["condition"] == cond]
        summary[cond] = {
            "episodes": len(sub),
            "flow_channel_mean_pct": round(float(sub["flow_channel_ratio"].mean()), 2),
            "flow_channel_std_pct": round(float(sub["flow_channel_ratio"].std()), 2),
            "distress_mean_pct": round(float(sub["distress_ratio"].mean()), 2),
            "comfort_mean_pct": round(float(sub["comfort_ratio"].mean()), 2),
            "survival_rate_pct": round(float(sub["survived"].mean()) * 100.0, 1),
            "mean_survival_sec": round(float(sub["duration_sec"].mean()), 2),
            "mean_kills": round(float(sub["total_kills"].mean()), 1),
            "ammo_starvation_events": int(sub["ammo_starvation_events"].sum()),
            "mean_reward": round(float(sub["mean_reward"].mean()), 3),
        }

    json_path = output_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved benchmark summary statistics to: {json_path}")

    # Generate visual multi-panel plot
    plot_path = generate_benchmark_plots(df, output_dir)
    logger.info(f"Generated comparative benchmark chart: {plot_path}")

    # Print summary table to console
    print("\n" + "=" * 95)
    print(f"{'Condition':16s} | {'Flow Ratio (%)':15s} | {'Distress (%)':13s} | {'Survival (s)':13s} | {'Kills':8s} | {'Reward':8s}")
    print("-" * 95)
    for cond, s in summary.items():
        std_str = f"+/- {s['flow_channel_std_pct']:4.1f}%" if not np.isnan(s['flow_channel_std_pct']) else ""
        print(
            f"{cond:16s} | "
            f"{s['flow_channel_mean_pct']:6.1f}% {std_str:10s} | "
            f"{s['distress_mean_pct']:6.1f}%       | "
            f"{s['mean_survival_sec']:6.1f}s       | "
            f"{s['mean_kills']:5.1f}  | "
            f"{s['mean_reward']:+0.2f}"
        )
    print("=" * 95)


if __name__ == "__main__":
    main()
