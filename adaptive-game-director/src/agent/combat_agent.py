#!/usr/bin/env python3
"""Deep Q-Network Combat Agent for the Doom Marine (Player AI).

Learns real-time combat tactics:
- Target tracking & directional turning toward threats
- Firing weapon when on-target
- Advancing / closing distance during long-range engagements
- Strafing and evasive dodging under hostile fire

Works in tandem with the AI Game Director (Co-Evolution Architecture):
- The Combat Marine learns to maximize frags, damage dealt, and survival.
- The Game Director learns to modulate difficulty, keeping the marine in the Flow Channel [35, 75 HP].
"""

import math
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import vizdoom as vzd

from src.agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

# 7 Discrete Combat Tactics for the Doom Marine
COMBAT_ACTIONS = [
    "TURN_LEFT",            # 0: Snap aim left towards threat
    "TURN_RIGHT",           # 1: Snap aim right towards threat
    "ATTACK",               # 2: Fire weapon
    "MOVE_FORWARD",         # 3: Advance / close distance
    "MOVE_FORWARD_ATTACK",  # 4: Advance and fire simultaneously
    "STRAFE_LEFT_ATTACK",   # 5: Circle-strafe left while firing
    "STRAFE_RIGHT_ATTACK",  # 6: Circle-strafe right while firing
]


class CombatQNetwork(nn.Module):
    """Dueling Deep Q-Network for FPS combat tactical decision making.

    Splits Q-value into state-value V(s) and action-advantage A(s,a):
        Q(s,a) = V(s) + (A(s,a) - mean(A))

    This lets the network learn 'how good is this situation' independently from
    'which action is best here', dramatically improving learning efficiency.
    """

    def __init__(self, state_dim: int = 14, action_dim: int = 7) -> None:
        super().__init__()
        # Shared feature extraction backbone (larger capacity)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        # Value stream: V(s) — "how good is this state regardless of action?"
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        # Advantage stream: A(s,a) — "how much better is this action vs average?"
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)           # (batch, 1)
        advantage = self.advantage_stream(features)   # (batch, action_dim)
        # Dueling aggregation: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class CombatStateExtractor:
    """Extracts a normalized 14-dimensional spatial combat vector from ViZDoom state."""

    @staticmethod
    def extract_state(state: vzd.GameState) -> np.ndarray:
        """Constructs a normalized 14-dim combat observation:

        [0] Normalized Health: [0, 1]
        [1] Normalized Ammo: [0, 1]
        [2] Normalized Armor: [0, 1]
        [3] Nearest Threat / Item Angle Diff: [-1, 1] (-180 deg to +180 deg mapped to [-1, 1])
        [4] Nearest Threat / Item Distance: [0, 1] (clamped to 1000 units)
        [5] Second Threat Angle Diff: [-1, 1]
        [6] Second Threat Distance: [0, 1]
        [7] Threat Count: [0, 1] (count / 10 clamped)
        [8] Velocity X: [-1, 1] (vx / 20 clamped)
        [9] Velocity Y: [-1, 1] (vy / 20 clamped)
        [10] Attack Ready: 0 or 1
        [11] Damage Intake Rate (short-term hit flag)
        [12] On-Target Flag (1 if nearest threat visibly within crosshairs)
        [13] Item Nearby Flag (1 if weapon/ammo/medikit within 350 units)
        """
        g_vars = state.game_variables
        health = float(g_vars[0]) if len(g_vars) > 0 else 0.0
        armor = float(g_vars[1]) if len(g_vars) > 1 else 0.0
        ammo = float(g_vars[2]) if len(g_vars) > 2 else 0.0
        attack_ready = 1.0 if (len(g_vars) > 17 and g_vars[17] > 0) else 0.0
        vx = float(g_vars[14]) if len(g_vars) > 14 else 0.0
        vy = float(g_vars[15]) if len(g_vars) > 15 else 0.0

        # Determine player position and angle from engine game variables
        px, py, p_angle = 0.0, 0.0, 0.0
        if len(g_vars) > 12:
            px = float(g_vars[11])
            py = float(g_vars[12])
        if len(g_vars) > 18:
            p_angle = float(g_vars[18])

        # Locate player object to get exact local player ID and orientation
        local_player_id = None
        enemies: List[Tuple[float, float, float]] = []
        items: List[Tuple[float, float]] = []

        if hasattr(state, "objects") and state.objects:
            # First pass: find the local player object
            for obj in state.objects:
                name_l = obj.name.lower()
                if "player" in name_l:
                    d_pos = math.hypot(obj.position_x - px, obj.position_y - py)
                    if d_pos < 50.0 or local_player_id is None:
                        local_player_id = obj.id
                        px, py = obj.position_x, obj.position_y
                        p_angle = obj.angle

            # Second pass: catalog hostile enemies and items/weapons (strictly excluding local player)
            for obj in state.objects:
                if local_player_id is not None and obj.id == local_player_id:
                    continue
                name_l = obj.name.lower()
                if (
                    "demon" in name_l or "imp" in name_l or "guy" in name_l or
                    "marine" in name_l or "caco" in name_l or "knight" in name_l or
                    "baron" in name_l or "revenant" in name_l or "mancubus" in name_l or
                    "arachno" in name_l or "vile" in name_l or "soul" in name_l or
                    "cyber" in name_l or "spider" in name_l or "zombie" in name_l or
                    "vzd" in name_l or ("player" in name_l and obj.id != local_player_id)
                ):
                    enemies.append((obj.position_x, obj.position_y, 0.0))
                elif (
                    "medikit" in name_l or "clip" in name_l or "stimpack" in name_l or
                    "shell" in name_l or "rocket" in name_l or "bonus" in name_l or
                    "box" in name_l or "armor" in name_l or "pack" in name_l or
                    "shotgun" in name_l or "chaingun" in name_l or "plasmarifle" in name_l or
                    "rocketlauncher" in name_l or "bfg" in name_l or "chainsaw" in name_l or
                    "weapon" in name_l
                ):
                    items.append((obj.position_x, obj.position_y))

        # Sort enemies by Euclidean distance from player
        enemy_dists: List[Tuple[float, float]] = []  # (dist, angle_diff)
        for ex, ey, _ in enemies:
            dx = ex - px
            dy = ey - py
            dist = math.sqrt(dx * dx + dy * dy)
            target_deg = math.degrees(math.atan2(dy, dx))
            angle_diff = (target_deg - p_angle + 180.0) % 360.0 - 180.0
            enemy_dists.append((dist, angle_diff))

        enemy_dists.sort(key=lambda t: t[0])

        e1_dist, e1_angle = (1000.0, 0.0)
        e2_dist, e2_angle = (1000.0, 0.0)
        if len(enemy_dists) > 0:
            e1_dist, e1_angle = enemy_dists[0]
        if len(enemy_dists) > 1:
            e2_dist, e2_angle = enemy_dists[1]

        # Nearest item / gun distance and heading
        min_item_dist = 1000.0
        item_angle = 0.0
        for ix, iy in items:
            dist = math.hypot(ix - px, iy - py)
            if dist < min_item_dist:
                min_item_dist = dist
                i_dx = ix - px
                i_dy = iy - py
                i_deg = math.degrees(math.atan2(i_dy, i_dx))
                item_angle = (i_deg - p_angle + 180.0) % 360.0 - 180.0

        # TRUE LINE-OF-SIGHT OCCLUSION TARGETING VIA state.labels
        # If an enemy is behind a stone wall, it is NOT rendered into state.labels
        vis_enemies: List[Tuple[float, float]] = []  # (abs_offset, offset)
        if hasattr(state, "labels") and state.labels:
            for l in state.labels:
                name_l = l.object_name.lower()
                if l.value != 255 and (
                    "demon" in name_l or "imp" in name_l or "guy" in name_l or
                    "marine" in name_l or "caco" in name_l or "knight" in name_l or
                    "baron" in name_l or "revenant" in name_l or "mancubus" in name_l or
                    "arachno" in name_l or "vile" in name_l or "soul" in name_l or
                    "cyber" in name_l or "spider" in name_l or "zombie" in name_l or
                    "vzd" in name_l or ("player" in name_l and "doomplayer" not in name_l)
                ):
                    cx = l.x + (l.width / 2.0)
                    offset = cx - 160.0  # Screen width 320, center 160
                    vis_enemies.append((abs(offset), offset))

        vis_enemies.sort(key=lambda t: t[0])
        has_visible_enemy = len(vis_enemies) > 0

        # On-target is ONLY True when a hostile enemy is visibly inside the crosshairs (|offset| <= 25 px)
        on_target = 1.0 if (has_visible_enemy and vis_enemies[0][0] <= 25.0) else 0.0
        item_nearby = 1.0 if min_item_dist < 350.0 else 0.0

        # Exact normalized angle to target:
        # 1. If visible enemy on screen: use direct crosshair pixel offset [-1, 1]
        # 2. If enemies in map but not in line-of-sight: turn towards nearest threat
        # 3. If all enemies dead: turn towards nearest weapon/item pickup!
        if has_visible_enemy:
            target_angle_norm = np.clip(vis_enemies[0][1] / 160.0, -1.0, 1.0)
            target_dist_norm = np.clip(e1_dist / 1000.0, 0.0, 1.0)
        elif len(enemy_dists) > 0:
            target_angle_norm = np.clip(e1_angle / 180.0, -1.0, 1.0)
            target_dist_norm = np.clip(e1_dist / 1000.0, 0.0, 1.0)
        elif len(items) > 0:
            target_angle_norm = np.clip(item_angle / 180.0, -1.0, 1.0)
            target_dist_norm = np.clip(min_item_dist / 1000.0, 0.0, 1.0)
        else:
            target_angle_norm = 0.0
            target_dist_norm = 1.0

        vec = np.zeros(14, dtype=np.float32)
        vec[0] = np.clip(health / 100.0, 0.0, 1.0)
        vec[1] = np.clip(ammo / 50.0, 0.0, 1.0)
        vec[2] = np.clip(armor / 100.0, 0.0, 1.0)
        vec[3] = target_angle_norm
        vec[4] = target_dist_norm
        vec[5] = np.clip(e2_angle / 180.0, -1.0, 1.0)
        vec[6] = np.clip(e2_dist / 1000.0, 0.0, 1.0)
        vec[7] = np.clip(len(enemies) / 10.0, 0.0, 1.0)
        vec[8] = np.clip(vx / 20.0, -1.0, 1.0)
        vec[9] = np.clip(vy / 20.0, -1.0, 1.0)
        vec[10] = attack_ready
        vec[11] = 1.0 if (len(g_vars) > 6 and g_vars[6] > 0) else 0.0  # hits taken
        vec[12] = on_target
        vec[13] = item_nearby
        return vec


class CombatAgent:
    """Reinforcement learning agent that controls the Doom Marine in combat."""

    def __init__(
        self,
        state_dim: int = 14,
        action_dim: int = 7,
        lr: float = 3e-4,
        gamma: float = 0.99,
        batch_size: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = CombatQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = CombatQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=80000, alpha=0.6, beta=0.4)
        self.train_steps = 0
        self.tau = 0.005  # Polyak soft target update coefficient

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Spatially-guided exploration and Q-network combat action selection."""
        if random.random() < epsilon:
            # Guided exploration: NEVER fire unless locked on target!
            angle_diff = state[3]  # [-1, 1], where > 0 is right, < 0 is left
            on_target = state[12] > 0.5

            if on_target:
                # Locked on enemy: circle-strafe and fire to avoid taking hits!
                return random.choices([2, 4, 5, 6], weights=[0.40, 0.20, 0.20, 0.20])[0]
            elif angle_diff < -0.05:
                # Threat/Item is to the left: turn left or strafe left
                return random.choices([0, 5, 3], weights=[0.50, 0.30, 0.20])[0]
            elif angle_diff > 0.05:
                # Threat/Item is to the right: turn right or strafe right
                return random.choices([1, 6, 3], weights=[0.50, 0.30, 0.20])[0]
            else:
                # Sprint forward towards threats or weapon pickups
                return random.choices([3, 5, 6, 0, 1], weights=[0.60, 0.15, 0.15, 0.05, 0.05])[0]

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.q_network(state_t)
            return int(torch.argmax(q_vals, dim=1).item())

    def convert_action_to_buttons(
        self,
        action_idx: int,
        available_buttons: List[vzd.Button],
        on_target: bool = False,
    ) -> List[bool]:
        """Translates discrete combat action into ViZDoom button vector.

        Supports true 3D circle-strafing and strict fire discipline trigger gating.
        """
        turn_left = action_idx == 0
        turn_right = action_idx == 1
        strafe_left = action_idx == 5
        strafe_right = action_idx == 6
        move_forward = action_idx in [3, 4]

        # Trigger is pulled ONLY when on_target (visible enemy in direct line of sight)
        attack = (action_idx in [2, 4, 5, 6]) and on_target

        # If attack was chosen but not on target, advance forward instead of standing still
        if action_idx == 2 and not on_target:
            move_forward = True

        # In scenarios without strafe buttons (e.g. defend_the_center), map strafe to turning
        has_strafe = vzd.Button.MOVE_LEFT in available_buttons
        if not has_strafe:
            if strafe_left:
                turn_left = True
                strafe_left = False
            if strafe_right:
                turn_right = True
                strafe_right = False

        btn_vector = []
        for btn in available_buttons:
            if btn == vzd.Button.TURN_LEFT:
                btn_vector.append(turn_left)
            elif btn == vzd.Button.TURN_RIGHT:
                btn_vector.append(turn_right)
            elif btn == vzd.Button.ATTACK:
                btn_vector.append(attack)
            elif btn == vzd.Button.MOVE_FORWARD:
                btn_vector.append(move_forward)
            elif btn == vzd.Button.MOVE_LEFT:
                btn_vector.append(strafe_left)
            elif btn == vzd.Button.MOVE_RIGHT:
                btn_vector.append(strafe_right)
            else:
                btn_vector.append(False)
        return btn_vector

    def calculate_combat_reward(
        self,
        prev_vars: List[float],
        curr_vars: List[float],
        action_idx: int,
        state_vec: np.ndarray,
        prev_state_vec: Optional[np.ndarray] = None,
    ) -> float:
        """Dense combat reward: rewards aim tracking, on-target shooting, kills, dodging, and gun pickups;
        penalizes dry spraying at walls and damage intake."""
        if not prev_vars or not curr_vars:
            return 0.0

        prev_health = prev_vars[0] if len(prev_vars) > 0 else 100.0
        curr_health = curr_vars[0] if len(curr_vars) > 0 else 100.0
        prev_kills = prev_vars[3] if len(prev_vars) > 3 else 0.0
        curr_kills = curr_vars[3] if len(curr_vars) > 3 else 0.0
        prev_damage = prev_vars[7] if len(prev_vars) > 7 else 0.0
        curr_damage = curr_vars[7] if len(curr_vars) > 7 else 0.0
        prev_dmg_taken = prev_vars[8] if len(prev_vars) > 8 else 0.0
        curr_dmg_taken = curr_vars[8] if len(curr_vars) > 8 else 0.0
        is_dead = (curr_vars[9] > 0 if len(curr_vars) > 9 else False) or curr_health <= 0

        # Kills and damage dealt
        delta_kills = max(0.0, curr_kills - prev_kills)
        delta_damage_dealt = max(0.0, curr_damage - prev_damage)
        delta_damage_taken = max(0.0, curr_dmg_taken - prev_dmg_taken)

        r_kill = delta_kills * 5.0
        r_damage = (delta_damage_dealt / 50.0) * 2.0
        p_damage_taken = (delta_damage_taken / 50.0) * 1.5
        p_death = 5.0 if is_dead else 0.0

        # Aiming alignment progress reward (turning toward enemy/weapon)
        r_aim_progress = 0.0
        if prev_state_vec is not None:
            prev_angle_err = abs(prev_state_vec[3]) * 180.0
            curr_angle_err = abs(state_vec[3]) * 180.0
            angle_delta = prev_angle_err - curr_angle_err
            r_aim_progress = float(np.clip(angle_delta / 25.0, -0.2, 0.2))

        # Smart shooting reward vs wall-shooting penalty
        attack_used = action_idx in [2, 4, 5, 6]
        on_target = state_vec[12] > 0.5
        if attack_used:
            if on_target or delta_damage_dealt > 0:
                r_fire = 0.2  # Good shot lined up
            else:
                r_fire = -0.15  # Wasting ammo shooting at a wall!
        else:
            r_fire = 0.0

        # Forward movement reward when targets are in the distance
        r_advance = 0.0
        if action_idx in [3, 4] and state_vec[4] > 0.15:
            r_advance = 0.04

        # Evasive strafe-dodging reward under fire
        r_dodge = 0.0
        if delta_damage_taken > 0 and action_idx in [5, 6]:
            r_dodge = 0.35  # Circle-strafing to evade enemy projectiles!

        # Gun / Ammo / Armor / Health pickup rewards
        prev_ammo = prev_vars[2] if len(prev_vars) > 2 else 0.0
        curr_ammo = curr_vars[2] if len(curr_vars) > 2 else 0.0
        prev_armor = prev_vars[1] if len(prev_vars) > 1 else 0.0
        curr_armor = curr_vars[1] if len(curr_vars) > 1 else 0.0
        r_pickup = 0.0
        if curr_ammo > prev_ammo:
            r_pickup += 0.5  # Picked up weapon or ammo!
        if curr_armor > prev_armor:
            r_pickup += 0.5  # Picked up armor vest!
        if curr_health > prev_health:
            r_pickup += 0.5  # Picked up medikit!

        reward = r_kill + r_damage - p_damage_taken - p_death + r_aim_progress + r_fire + r_advance + r_dodge + r_pickup
        return float(np.clip(reward, -10.0, 10.0))

    def train_step(self) -> Optional[float]:
        """Double DQN with Prioritized Replay and soft target updates.

        Double DQN: Online network selects best action, target network evaluates it.
        This eliminates Q-value overestimation that makes vanilla DQN choose bad actions.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Prioritized sampling — critical transitions (kills, deaths) replayed more often
        states, actions, rewards, next_states, dones, is_weights, tree_indices = \
            self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.BoolTensor(dones).unsqueeze(1).to(self.device)
        is_weights_t = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        current_q = self.q_network(states_t).gather(1, actions_t)

        with torch.no_grad():
            # DOUBLE DQN: Use online network to SELECT the best action...
            best_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            # ...but use target network to EVALUATE that action's Q-value
            next_q = self.target_network(next_states_t).gather(1, best_actions)
            target_q = rewards_t + (self.gamma * next_q * (~dones_t).float())

        # Per-element TD-errors for priority updates
        td_errors = (current_q - target_q).detach().cpu().numpy().flatten()

        # Importance-sampling weighted Huber loss (unbiased gradient under non-uniform sampling)
        element_losses = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (element_losses * is_weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities in the replay buffer based on TD-errors
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        # Soft target network update (Polyak averaging) — smoother than hard copy
        self.train_steps += 1
        for target_param, online_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

        return float(loss.item())

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "train_steps": self.train_steps,
        }, path)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.train_steps = checkpoint.get("train_steps", 0)
