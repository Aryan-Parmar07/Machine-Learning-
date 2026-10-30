# Adaptive Game Director
> **Reinforcement Learning-Based Dynamic Difficulty Adjustment for First-Person Shooters**

**Submitted by:**
- **Lekkala Phanidhar Reddy** [Reg No: `RA2411026030032`]
- **Aryan Parmar** [Reg No: `RA2411026030034`]
- **Mohammad Anaan Ansari** [Reg No: `RA2411026030055`]

[![ViZDoom](https://img.shields.io/badge/ViZDoom-1.3.1-red.svg)](https://vizdoom.farama.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-brightgreen.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://python.org/)

---

## 1. Project Goal

The **Adaptive Game Director** is a college/research reinforcement learning project that investigates **Dynamic Difficulty Adjustment (DDA)** in fast-paced 3D combat environments.

Traditional game difficulty systems rely on static difficulty presets ("Easy", "Medium", "Hard") chosen upfront by players, which often lead to either **boredom** (when the challenge is insufficient) or **frustration** (when the challenge creates unfair punishment). This project designs an autonomous **AI Game Director** that:
- Observes real-time gameplay telemetry from a human player.
- Estimates the player's cognitive load, combat efficiency, and skill level.
- Dynamically modulates the game environment (enemy aggressiveness, spawn density, ammo and health resources) to sustain the player inside the optimal **Flow State** (Csikszentmihalyi's flow channel).

> [!IMPORTANT]
> **RL Agent Role Distinction:**
> The Reinforcement Learning agent **does NOT** control the player character.
> - The **HUMAN** controls the player character via keyboard and mouse in real time.
> - The **RL AGENT** acts as the high-level **AI Director**, observing player telemetry and issuing environment modulation decisions.

---

## 2. Architecture

```
                       ┌──────────────────────┐
                       │     Human Player     │
                       │   (Keyboard & Mouse) │
                       └──────────┬───────────┘
                                  │ (User Actions)
                                  ▼
                       ┌──────────────────────┐
                       │    ViZDoom Engine    │
                       │   (SPECTATOR Mode)   │
                       └──────────┬───────────┘
                                  │ (State & Entity Inspection)
                                  ▼
                       ┌──────────────────────┐
                       │  Gameplay Telemetry  │
                       │ (Direct & Derived)   │
                       └──────────┬───────────┘
                                  │ (15-Dim Normalized Vector)
                                  ▼
                       ┌──────────────────────┐
                       │  Player Model &      │
                       │  Skill Estimator     │
                       └──────────┬───────────┘
                                  │ (Normalized Observation)
                                  ▼
                       ┌──────────────────────┐
                       │    DDA Controller    │
                       │ (Rule-Based or DQN)  │
                       └──────────┬───────────┘
                                  │ (Difficulty Modulation Action)
                                  ▼
                       ┌──────────────────────┐
                       │    ViZDoom Engine    │
                       │ (Difficulty Modifiers│
                       └──────────────────────┘
```

### Dual-Loop Hierarchy:
1. **Inner Loop (Game Tics @ 35 FPS):** The human player plays Doom, shooting, dodging, and collecting pickups. ViZDoom simulates physics and bot interactions.
2. **Outer Loop (Director Step @ ~0.2 Hz / every 5 seconds):** The AI Game Director receives aggregated player performance vectors, assesses flow adherence, and adjusts difficulty parameters.

---

## 3. Directory Structure

```
adaptive-game-director/
│
├── README.md                      # Project overview, architecture, & instructions
├── requirements.txt               # Pinned project dependencies
├── .gitignore                     # Git ignore rules for data, models, logs, & caches
│
├── config/
│   └── config.yaml                # Scenarios, telemetry rates, DDA rules, & RL hyperparameters
│
├── src/
│   ├── __init__.py                # Package declaration
│   ├── environment/
│   │   ├── __init__.py
│   │   └── vizdoom_env.py         # Gymnasium Game Director environment interface
│   ├── player_model/
│   │   ├── __init__.py
│   │   ├── features.py            # Feature extractor: direct vs derived metrics & normalization
│   │   └── skill_estimator.py     # Exponential moving average player skill model
│   ├── dda/
│   │   ├── __init__.py
│   │   ├── difficulty.py          # DoomSkill enums, 7-action space, & DifficultyState
│   │   └── rule_based.py          # Heuristic baseline DDA controller
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── network.py             # PyTorch MLP Q-network (DDA_QNetwork)
│   │   ├── replay_buffer.py       # Experience replay buffer for transition storage
│   │   └── dqn.py                 # DQN agent skeleton (action selection & checkpointing)
│   ├── rewards/
│   │   ├── __init__.py
│   │   └── reward_functions.py    # Flow State reward formulation
│   └── utils/
│       ├── __init__.py
│       └── logger.py              # Telemetry CSV recorder & console logger
│
├── scripts/
│   ├── play_human.py              # [Milestone 1] Live human telemetry collection pipeline
│   ├── collect_telemetry.py       # Multi-scenario batch telemetry collector
│   ├── run_rule_based_dda.py      # Rule-based DDA live controller execution
│   ├── train_dqn.py               # Deep Q-Network training pipeline
│   └── evaluate.py                # Comparative benchmark across experimental conditions
│
├── data/
│   ├── raw/                       # Unprocessed session captures
│   ├── processed/                 # Cleaned and partitioned datasets
│   └── telemetry/                 # Recorded human session CSV files
│
├── models/
│   ├── checkpoints/               # Intermediate training checkpoints
│   └── final/                     # Trained final policy weights
│
├── logs/
│   └── tensorboard/               # TensorBoard experiment runs
│
├── results/
│   ├── plots/                     # Generated evaluation plots and figures
│   ├── metrics/                   # Tabulated benchmark metrics
│   └── experiments/               # Experiment logs and summaries
│
└── notebooks/
    ├── 01_telemetry_analysis.ipynb # Telemetry integrity & vital signs analysis
    ├── 02_player_model.ipynb       # Skill estimation and aggression clustering
    └── 03_dda_analysis.ipynb       # Flow channel validation & comparative analysis
```

---

## 4. ViZDoom Dependency

ViZDoom is treated as an **external game engine dependency**.
- Existing installation: `C:\PROJECTS\ML\ViZDoom`
- Scenarios and WAD files are accessed either directly through `vizdoom.scenarios_path` or explicit scenario paths.
- **No ViZDoom source files are modified.** All project logic resides entirely within `C:\PROJECTS\ML\adaptive-game-director`.

---

## 5. Current Milestone: Milestone 1

### Objective: Minimal Working Human-Play Telemetry Pipeline
1. Launch ViZDoom in human-playable `vzd.Mode.SPECTATOR` mode.
2. Enable mouse freelook (`+freelook 1`) and standard FPS controls (WASD + mouse look + left click shoot).
3. Sample direct engine state variables and calculate derived metrics without faking data.
4. Record sessions to `data/telemetry/human_session.csv`.
5. Stream a live terminal dashboard showing vitals, kill count, and estimated skill tier.

### Direct vs. Derived Variables

| Metric | Source | Description |
| :--- | :--- | :--- |
| `health`, `armor`, `ammo` | **Direct** (`vzd.GameVariable`) | Player status read directly from engine state |
| `killcount`, `deathcount` | **Direct** (`vzd.GameVariable`) | Cumulative combat counts |
| `hitcount`, `hits_taken` | **Direct** (`vzd.GameVariable`) | Shots landed on targets and hits received |
| `damagecount`, `damage_taken`| **Direct** (`vzd.GameVariable`) | Raw damage dealt by player and taken by player |
| `position_x/y/z`, `vel_x/y/z`| **Direct** (`vzd.GameVariable`) | Spatial coordinates and instantaneous velocities |
| `enemy_count`, `item_count` | **Direct** (`state.objects`) | Active entities dynamically filtered from level actors |
| `elapsed_time_sec` | **Derived** | `tic / vzd.DEFAULT_TICRATE` (35 tics/sec) |
| `kill_rate`, `death_rate` | **Derived** | Kills or deaths per elapsed second |
| `damage_dealt/taken_rate` | **Derived** | Damage volume per elapsed second |
| `accuracy_estimate` | **Derived** | `hitcount / shots_fired_estimate` |
| `aggression_score` | **Derived** | Composite heuristic of player speed and damage ratio |
| `player_skill_estimate` | **Derived** | EMA combat efficiency, K/D ratio, and accuracy `[0, 1]` |

---

## 6. How to Run the Human Telemetry Collector

### Step 1: Navigate to Project Directory
```powershell
cd C:\PROJECTS\ML\adaptive-game-director
```

### Step 2: Run Human Gameplay Collector
```powershell
python scripts/play_human.py
```

### Step 3: Gameplay Options & Flags
You can select different scenarios, durations, or logging intervals:

```powershell
# Play Deathmatch with 2 episodes and 5-tic telemetry interval (~7 Hz)
python scripts/play_human.py --scenario deathmatch.cfg --episodes 2 --interval-tics 5

# Play Defend The Center scenario starting at Hard difficulty (Skill 4)
python scripts/play_human.py --scenario defend_the_center.cfg --skill 4

# Save telemetry to a custom destination
python scripts/play_human.py --output data/telemetry/my_test_run.csv

# Play with DQN Director and Live HUD Overlay (Default):
python scripts/play_human.py --scenario defend_the_center.cfg --director dqn

# Run Learned DQN DDA in Human Mode with Live HUD:
python scripts/run_dqn_dda.py --mode human --scenario defend_the_center.cfg

# Run Automated Bot Evaluation with Live HUD:
python scripts/run_dqn_dda.py --mode bot --scenario defend_the_center.cfg
```

### Live AI Director HUD Overlay:
Both Human and AI playthroughs feature an always-on-top broadcast HUD banner displaying:
- **Difficulty State & Treatment:** `EASY (Skill 1-2)`, `MEDIUM (Skill 3)`, `HARD (Skill 4)`, or `NIGHTMARE (Skill 5)`
- **Player Skill Assessment:** `NOVICE (<0.35)`, `INTERMEDIATE (0.35-0.75)`, or `EXPERT (>0.75)`
- **Flow Channel Status:** `FLOW CHANNEL (Optimal [35-75 HP])`, `DISTRESS (<35 HP)`, or `COMFORT (>75 HP)`
- **Live Pacing Actions:** Immediate emergency aid notifications (`+25 HP & 50 Ammo`), threat spawns, or monster thinning.
- **In-Game Text:** Synchronized on-screen console banners directly inside the ViZDoom screen.

### Controls:
- **Move:** `W`, `A`, `S`, `D` or Arrow Keys
- **Aim / Look:** Mouse (Freelook enabled)
- **Shoot / Attack:** Left Mouse Click or `CTRL`
- **Weapon Selection:** Number keys `1` through `7`
- **Exit / Finish:** Press `ESC` or close the game window

---

## 7. Planned DDA / RL Architecture

### State Representation (15-Dimensional Normalized Vector)
The state vector $\mathbf{s}_t \in \mathbb{R}^{15}$ fed to the Director agent encodes:
1. `norm_health`: Current health normalized to $[0, 1]$.
2. `norm_armor`: Current armor normalized to $[0, 1]$.
3. `norm_ammo`: Current ammo normalized to $[0, 1]$.
4. `norm_kills`: Cumulative kill count.
5. `norm_deaths`: Cumulative death count.
6. `kill_rate`: Kills per second.
7. `death_rate`: Deaths per second.
8. `damage_dealt_rate`: Damage inflicted per second.
9. `damage_received_rate`: Damage sustained per second.
10. `accuracy_estimate`: Percentage of hits landed.
11. `player_speed`: Normalized 2D velocity magnitude.
12. `enemy_count`: Number of active hostile actors.
13. `recent_trend`: Health delta over sliding 10-second window $[-1, 1]$.
14. `aggression_score`: Forward momentum and combat engagement.
15. `player_skill_estimate`: Ongoing EMA skill score $[0, 1]$.

### Action Space (Discrete 7 Actions)
At each decision step (every 5 seconds), the Game Director executes one discrete action:
- `0`: **No change** (maintain current parameters)
- `1`: **Increase difficulty** (escalate bot aggression / Doom skill)
- `2`: **Decrease difficulty** (de-escalate bot aggression / Doom skill)
- `3`: **Increase enemy spawn rate** (spawn additional adversaries)
- `4`: **Decrease enemy spawn rate** (reduce adversary pressure)
- `5`: **Increase resources** (spawn health packs and ammunition)
- `6`: **Decrease resources** (restrict health and ammunition drops)

### Flow-State Reward Formulation
$$R_{director} = R_{health\_balance} + R_{combat\_engagement} - P_{death} - P_{boredom}$$
- **$R_{health\_balance}$**: Maximized when player health hovers around the tension median ($\approx 55$ HP).
- **$R_{combat\_engagement}$**: Rewarded when player is engaged in active combat.
- **$P_{death}$**: Severe penalty if the player dies (indicates excessive frustration or unfair spikes).
- **$P_{boredom}$**: Penalty if player sits continuously at 100% health without taking damage.

---

## 8. Planned Experiments

In Milestone 3, we will conduct comparative empirical evaluations across three conditions:

| Condition | Description |
| :--- | :--- |
| **Condition A: Fixed Difficulty** | Fixed engine skill levels (Easy = Skill 1, Medium = Skill 3, Hard = Skill 5) without dynamic adaptation. |
| **Condition B: Rule-Based DDA** | Deterministic heuristic rules adjusting difficulty based on static health and ammo thresholds. |
| **Condition C: DQN DDA Agent** | Reinforcement learning Game Director trained to maximize the continuous Flow-State reward. |

### Evaluation Metrics:
1. **Survival Time:** Total seconds the player survives per encounter.
2. **Death Rate:** Frequency of player deaths per minute of play.
3. **Kill Rate:** Rate of enemy eliminations.
4. **Difficulty Stability:** Standard deviation of difficulty adjustments (penalizing disruptive oscillations).
5. **Adaptation Latency:** Time in seconds taken by the controller to intervene when the player enters critical distress.
6. **Target-Difficulty Error:** Mean Squared Error (MSE) between player state and the target flow band $[35, 75]$ HP.
7. **Player Performance Score:** Composite metric of combat efficiency and accuracy.
8. **Cumulative Flow Reward:** Total discounted reward accumulated by the director policy.

---

## 9. Parallel Multi-Window AI Surveillance Grid & Advanced RL Architecture

The project features a real-time, side-by-side **Surveillance Grid** powered by Pygame to visually observe AI training live across multiple concurrent ViZDoom game instances with a unified dashboard and zero audio pollution.

### Dual-Agent Co-Evolution Architecture:
1. **The Combat Marine (Dueling Double DQN Player AI):** Learns combat locomotion, weapon acquisition, aiming lock-on, circle-strafing, and dodging hostile projectiles.
2. **The Game Director (Pacing AI):** Learns dynamic difficulty modulation to keep the marine alive inside the Flow Channel ($[35, 75]$ HP).

### Advanced Reinforcement Learning Upgrades:
- **Dueling Network Architecture:** State-value $V(s)$ and action-advantage $A(s, a)$ are split into separate streams and aggregated as $Q(s,a) = V(s) + (A(s,a) - \frac{1}{|A|}\sum A)$. This lets the agent learn state valuation independently from action quality.
- **Double DQN:** Action selection is performed by the online network ($\arg\max_a Q(s', a; \theta)$) while evaluation is performed by the target network ($Q(s', a^*; \theta^-)$), preventing positive maximization bias.
- **Prioritized Experience Replay (PER):** Binary SumTree structure enables $O(\log n)$ sampling proportional to temporal-difference error magnitude $|\delta_i|^\alpha$, with importance-sampling weights $w_i = (N \cdot P(i))^{-\beta}$ for unbiased gradient updates.
- **Polyak Soft Target Updates:** Target network parameters smoothly track online parameters via $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ with $\tau=0.005$.
- **Autonomous Curriculum Learning:** Parallel game instances start at Tier 1 and automatically graduate to harder scenarios as combat mastery improves:
  1. **Tier 1 (Center Arena):** `defend_the_center.cfg` — Stationary 360° aiming, angle lock-on, and shooting. (Target: Avg $\ge 3.0$ Kills)
  2. **Tier 2 (Line Corridor):** `defend_the_line.cfg` — Corridor defense, dodging incoming fireballs. (Target: Avg $\ge 3.5$ Kills)
  3. **Tier 3 (Deadly Corridor):** `deadly_corridor.cfg` — Advance navigation under heavy hostile fire. (Target: Reach level end or Avg $\ge 2.5$ Kills)
  4. **Tier 4 (Open Arena):** `deathmatch.cfg` — Full open roaming, health/weapon pickups, and dynamic multi-enemy navigation.

### Quickstart Commands:
```bash
# 4-Window Grid with Autonomous Curriculum Progression (Default):
python scripts/visual_training_grid.py --grid-size 4 --mode co_evolution

# 6-Window Parallel Surveillance Matrix (3x2):
python scripts/visual_training_grid.py --grid-size 6 --mode co_evolution

# Focus strictly on Combat Marine training:
python scripts/visual_training_grid.py --grid-size 6 --mode combat_only

# Static mixed scenario grid (without auto-graduation):
python scripts/visual_training_grid.py --grid-size 4 --no-curriculum --scenario mixed
```
- **Exit & Checkpoint Save:** Press `Q` or `ESC` in the Pygame surveillance grid window at any point to save neural network weights to `models/final/`.

