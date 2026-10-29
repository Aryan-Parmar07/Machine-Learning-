#!/usr/bin/env python3
"""DQN Training Pipeline for the AI Game Director.

Trains a Deep Q-Network to dynamically adjust game difficulty based on real-time
player telemetry, learning an optimal policy that sustains the player inside the
Flow State channel (neither bored nor frustrated).

During training, player personas ('novice', 'intermediate', 'aggressive') are cycled
across episodes so the RL Director learns how to dynamically handle any human playstyle.

Usage:
    # Train DQN Game Director for 50 episodes in defend_the_center
    python scripts/train_dqn.py --scenario defend_the_center.cfg --episodes 50

    # Fast decision cycle (every 3 seconds) with custom learning rate
    python scripts/train_dqn.py --episodes 100 --decision-interval-sec 3.0 --lr 0.0003
"""

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.agent.dqn import DQNAgent
from src.environment.vizdoom_env import ViZDoomGameDirectorEnv
from src.dda.difficulty import DDAAction
from src.utils.logger import get_logger

logger = get_logger("TrainDQN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN AI Game Director")
    parser.add_argument(
        "--scenario",
        type=str,
        default="defend_the_center.cfg",
        help="ViZDoom scenario (e.g. defend_the_center.cfg, deathmatch.cfg). Default: defend_the_center.cfg",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of training episodes. Default: 50",
    )
    parser.add_argument(
        "--decision-interval-sec",
        type=float,
        default=3.0,
        help="Director decision interval in seconds. Default: 3.0",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for Adam optimizer. Default: 0.0003",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for replay buffer sampling. Default: 64",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for Bellman target. Default: 0.99",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="Initial epsilon for exploration. Default: 1.0",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="Minimum exploration rate. Default: 0.05",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.96,
        help="Multiplicative decay factor per episode. Default: 0.96",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "checkpoints"),
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--final-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "final"),
        help="Directory to save final model weights",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Open live visual game window so you can watch the AI play while the Director trains",
    )
    parser.add_argument(
        "--sound",
        action="store_true",
        default=False,
        help="Enable sound during visual training. Default: False (Muted / No Sound)",
    )
    parser.add_argument(
        "--no-sound",
        dest="sound",
        action="store_false",
        help="Ensure sound is disabled during visual training",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=35,
        help="Target framerate when rendered visually (e.g. 35 for realtime, 0 for max speed). Default: 35",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    checkpoint_dir = Path(args.checkpoint_dir)
    final_dir = Path(args.final_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Starting Reinforcement Learning (DQN) Game Director Training")
    logger.info(f"Scenario: {args.scenario} | Episodes: {args.episodes}")
    logger.info(f"Decision Interval: {args.decision_interval_sec}s | Batch Size: {args.batch_size} | LR: {args.lr}")

    # Initialize Gymnasium environment with automated combat bot
    env = ViZDoomGameDirectorEnv(
        scenario=args.scenario,
        decision_interval_sec=args.decision_interval_sec,
        player_mode="bot",
        persona="intermediate",
        render_mode="human" if args.render else None,
        sound_enabled=args.sound,
        render_fps=args.fps,
    )

    # Initialize DQN Agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
    )

    epsilon = args.epsilon_start
    best_episode_reward = -float("inf")
    personas = ["novice", "intermediate", "aggressive"]

    start_time = time.time()
    total_director_steps = 0

    print("=" * 80)
    logger.info("Training Loop Initiated (Cycling personas to learn dynamic adaptation)...")

    for ep in range(1, args.episodes + 1):
        # Rotate persona across episodes so the Director learns to adapt to any playstyle
        current_persona = personas[(ep - 1) % len(personas)]
        obs, info = env.reset(options={"persona": current_persona})

        episode_reward = 0.0
        episode_steps = 0
        losses = []
        flow_zone_steps = 0
        actions_taken = {a: 0 for a in range(7)}

        while True:
            # Director selects difficulty adjustment
            action = agent.select_action(obs, epsilon=epsilon)
            actions_taken[action] += 1

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            total_director_steps += 1

            # Log live action if rendered visually
            if args.render:
                logger.info(
                    f"  [DIRECTOR INTERVENTION] {step_info['applied_action']:22s} | "
                    f"{step_info['actuation_msg']:35s} | HP: {step_info['health']:4.0f} | "
                    f"Ammo: {step_info['ammo']:2.0f} | Reward: {reward:+.2f}"
                )

            # Check if player was maintained in optimal Flow Channel [35, 75 HP]
            hp = step_info["health"]
            if 35.0 <= hp <= 75.0:
                flow_zone_steps += 1

            # Store transition in experience replay buffer
            done = terminated or truncated
            agent.store_transition(obs, action, reward, next_obs, done)

            # Gradient descent training steps (4x updates per transition for accelerated learning)
            for _ in range(4):
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

            obs = next_obs
            if done:
                break

        # Decay exploration epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)

        avg_loss = np.mean(losses) if losses else 0.0
        flow_percent = (flow_zone_steps / max(1, episode_steps)) * 100.0

        # Most frequent action name
        top_action = max(actions_taken, key=actions_taken.get)
        top_action_name = DDAAction(top_action).name

        logger.info(
            f"Ep {ep:3d}/{args.episodes:3d} [{current_persona[:6].upper()}] | "
            f"Reward: {episode_reward:+6.2f} | Flow: {flow_percent:4.1f}% | "
            f"Steps: {episode_steps:2d} | Loss: {avg_loss:.4f} | Eps: {epsilon:.2f} | "
            f"Top Act: {top_action_name}"
        )

        # Checkpoint saving
        if episode_reward > best_episode_reward and ep >= 5:
            best_episode_reward = episode_reward
            best_ckpt_path = checkpoint_dir / "best_director_dqn.pt"
            agent.save_checkpoint(best_ckpt_path)

        if ep % 25 == 0:
            periodic_ckpt = checkpoint_dir / f"director_dqn_ep{ep}.pt"
            agent.save_checkpoint(periodic_ckpt)

    # Save final model
    final_model_path = final_dir / "director_dqn.pt"
    agent.save_checkpoint(final_model_path)
    env.close()

    total_duration = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Training completed successfully in {total_duration:.1f}s ({total_director_steps} Director decisions)!")
    logger.info(f"Best Episode Reward: {best_episode_reward:+.2f}")
    logger.info(f"Final model saved to: {final_model_path.resolve()}")


if __name__ == "__main__":
    cli_args = parse_args()
    train(cli_args)
