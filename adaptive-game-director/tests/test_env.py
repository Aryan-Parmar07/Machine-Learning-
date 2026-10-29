import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.vizdoom_env import ViZDoomGameDirectorEnv
from src.dda.difficulty import DDAAction

print("Initializing ViZDoomGameDirectorEnv...")
env = ViZDoomGameDirectorEnv(
    scenario="defend_the_center.cfg",
    decision_interval_sec=2.0,
    player_mode="bot",
    persona="intermediate",
)
obs, info = env.reset()
print("Initial obs shape:", obs.shape, "Initial HP:", info.get("health"))

actions = [
    DDAAction.NO_CHANGE,
    DDAAction.INCREASE_RESOURCES,
    DDAAction.INCREASE_ENEMY_SPAWN,
]

for idx, action in enumerate(actions):
    obs, reward, terminated, truncated, step_info = env.step(action.value)
    print(f"Step {idx + 1} | Action: {action.name:22s} | Reward: {reward:+.3f} | {step_info['actuation_msg']}")
    print(f"       HP: {step_info['health']:4.0f} | Ammo: {step_info['ammo']:2.0f} | Skill: {step_info['skill_estimate']:.2f} ({step_info['skill_tier']}) | Done: {terminated}")
    if terminated:
        print("Episode finished.")
        break

env.close()
print("ViZDoomGameDirectorEnv Gymnasium test PASSED successfully!")
