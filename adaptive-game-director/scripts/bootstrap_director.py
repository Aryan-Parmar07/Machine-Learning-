#!/usr/bin/env python3
"""Rapid Headless Persona Pre-Training Script for the AI Game Director.

Simulates 50 fast headless episodes across Novice, Intermediate, and Aggressive personas
to bootstrap the Director's neural network with thousands of diverse DDA transitions.

Features:
  - Dueling Double DQN architecture
  - Prioritized Experience Replay (SumTree)
  - 4x gradient updates per decision step
  - Auto-saves weights to models/final/director_dqn.pt
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.agent.dqn import DQNAgent
from src.environment.vizdoom_env import ViZDoomGameDirectorEnv
from src.dda.difficulty import DDAAction
from src.utils.logger import get_logger

logger = get_logger("BootstrapDirector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rapid Headless Persona Pre-Training for AI Game Director")
    parser.add_argument("--episodes", type=int, default=50, help="Total pre-training episodes. Default: 50")
    parser.add_argument("--scenario", type=str, default="defend_the_center.cfg", help="ViZDoom scenario config")
    parser.add_argument("--decision-interval", type=float, default=2.5, help="Decision interval in seconds. Default: 2.5")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate. Default: 0.0003")
    parser.add_argument(
        "--output-model",
        type=str,
        default=str(PROJECT_ROOT / "models" / "final" / "director_dqn.pt"),
        help="Path to save trained Director weights",
    )
    return parser.parse_args()


def run_bootstrap(args: argparse.Namespace) -> None:
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"STARTING RAPID HEADLESS DIRECTOR BOOTSTRAP ({args.episodes} EPISODES)")
    logger.info(f"Scenario: {args.scenario} | Decision Interval: {args.decision_interval}s | LR: {args.lr}")
    logger.info("Cycling Personas: NOVICE -> INTERMEDIATE -> AGGRESSIVE")
    logger.info("=" * 80)

    env = ViZDoomGameDirectorEnv(
        scenario=args.scenario,
        decision_interval_sec=args.decision_interval,
        player_mode="bot",
        persona="intermediate",
        render_mode=None,
        sound_enabled=False,
    )

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=args.lr,
        batch_size=64,
        tau=0.005,
    )

    # Load existing checkpoint if available to continue refining
    if output_path.exists():
        try:
            agent.load_checkpoint(output_path)
            logger.info(f"Loaded existing Director weights from: {output_path}")
        except Exception as e:
            logger.warning(f"Could not load existing checkpoint (starting fresh): {e}")

    personas = ["novice", "intermediate", "aggressive"]
    epsilon = 0.40  # Balanced exploration for pre-training
    epsilon_min = 0.05
    epsilon_decay = 0.96

    start_time = time.time()
    total_steps = 0
    recent_rewards = []
    action_counts = {a.name: 0 for a in DDAAction}

    for ep in range(1, args.episodes + 1):
        persona = personas[(ep - 1) % len(personas)]
        obs, _ = env.reset(options={"persona": persona})

        ep_reward = 0.0
        ep_steps = 0
        losses = []

        while True:
            action = agent.select_action(obs, epsilon=epsilon)
            action_counts[DDAAction(action).name] += 1

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            total_steps += 1
            ep_steps += 1
            ep_reward += reward

            # 4x gradient updates per transition for accelerated training
            for _ in range(4):
                l = agent.train_step()
                if l is not None:
                    losses.append(l)

            obs = next_obs
            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        recent_rewards.append(ep_reward)

        if ep % 5 == 0 or ep == args.episodes:
            avg_r = np.mean(recent_rewards[-5:])
            avg_loss = np.mean(losses) if losses else 0.0
            elapsed = time.time() - start_time
            logger.info(
                f"Episode {ep:2d}/{args.episodes} [{persona:12s}] | "
                f"Steps: {ep_steps:2d} (Total: {total_steps:4d}) | "
                f"Ep Reward: {ep_reward:+.2f} (Avg5: {avg_r:+.2f}) | "
                f"Loss: {avg_loss:.4f} | Eps: {epsilon:.3f} | {elapsed:.1f}s"
            )

    env.close()

    # Save final bootstrap weights
    agent.save_checkpoint(output_path)
    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"BOOTSTRAP COMPLETE in {total_time:.1f}s! Total Decision Steps: {total_steps}")
    logger.info("Action Distribution:")
    for a_name, count in action_counts.items():
        pct = (count / max(1, total_steps)) * 100.0
        logger.info(f"  {a_name:24s}: {count:4d} ({pct:5.1f}%)")
    logger.info(f"Saved Dueling Double DQN Director model to: {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    run_bootstrap(args)
