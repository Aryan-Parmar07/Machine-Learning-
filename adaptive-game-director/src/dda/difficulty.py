"""Difficulty states, enums, and action space definitions for the DDA Director."""

from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, Any, Optional


class DoomSkill(IntEnum):
    """ViZDoom native skill levels (1 to 5)."""
    VERY_EASY = 1  # "I'm Too Young to Die"
    EASY = 2       # "Hey, Not Too Rough"
    NORMAL = 3     # "Hurt Me Plenty"
    HARD = 4       # "Ultra-Violence"
    VERY_HARD = 5  # "Nightmare!"


class DDAAction(IntEnum):
    """Action space for the RL AI Game Director (Discrete(7)).

    The Game Director agent outputs one of these discrete decisions:
    0 = no change
    1 = increase difficulty
    2 = decrease difficulty
    3 = increase enemy spawn rate
    4 = decrease enemy spawn rate
    5 = increase resources
    6 = decrease resources
    """
    NO_CHANGE = 0
    INCREASE_DIFFICULTY = 1
    DECREASE_DIFFICULTY = 2
    INCREASE_ENEMY_SPAWN = 3
    DECREASE_ENEMY_SPAWN = 4
    INCREASE_RESOURCES = 5
    DECREASE_RESOURCES = 6


@dataclass
class DifficultyState:
    """Represents the multi-faceted game difficulty configuration.

    Attributes:
        doom_skill: ViZDoom native engine skill level (1 to 5).
        enemy_spawn_multiplier: Relative factor scaling monster spawn frequency.
        resource_multiplier: Relative factor scaling ammo/health drop rates.
    """
    doom_skill: int = DoomSkill.NORMAL
    enemy_spawn_multiplier: float = 1.0
    resource_multiplier: float = 1.0

    def apply_action(self, action: DDAAction) -> None:
        """Updates difficulty state parameters in response to an action."""
        if action == DDAAction.NO_CHANGE:
            return

        elif action == DDAAction.INCREASE_DIFFICULTY:
            self.doom_skill = min(5, self.doom_skill + 1)

        elif action == DDAAction.DECREASE_DIFFICULTY:
            self.doom_skill = max(1, self.doom_skill - 1)

        elif action == DDAAction.INCREASE_ENEMY_SPAWN:
            self.enemy_spawn_multiplier = min(2.5, self.enemy_spawn_multiplier + 0.25)

        elif action == DDAAction.DECREASE_ENEMY_SPAWN:
            self.enemy_spawn_multiplier = max(0.25, self.enemy_spawn_multiplier - 0.25)

        elif action == DDAAction.INCREASE_RESOURCES:
            self.resource_multiplier = min(2.5, self.resource_multiplier + 0.25)

        elif action == DDAAction.DECREASE_RESOURCES:
            self.resource_multiplier = max(0.25, self.resource_multiplier - 0.25)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doom_skill": self.doom_skill,
            "enemy_spawn_multiplier": round(self.enemy_spawn_multiplier, 2),
            "resource_multiplier": round(self.resource_multiplier, 2),
        }


class EngineActuator:
    """Dispatches live difficulty commands to the active ViZDoom game engine.

    Translates discrete DDA Director decisions into real-time in-game events
    such as spawning health/ammo pickups, introducing monsters, or pacing threats.
    """

    def __init__(self, game: Any) -> None:
        self.game = game
        self._ensure_cheats()

    def _ensure_cheats(self) -> None:
        """Enables cheats in ViZDoom engine to allow live DDA actor/pickup summoning."""
        try:
            self.game.send_game_command("sv_cheats 1")
        except Exception:
            pass

    def execute(
        self,
        action: DDAAction,
        difficulty_state: DifficultyState,
        player_health: Optional[float] = None,
        player_ammo: Optional[float] = None,
    ) -> str:
        """Applies DDA decision to difficulty_state and triggers live engine modulation.

        Args:
            action: Selected DDAAction.
            difficulty_state: Current DifficultyState to be updated.
            player_health: Optional current player health to enforce safety guardrails.
            player_ammo: Optional current weapon ammo to prevent weapon starvation.

        Returns:
            Human-readable description of the in-game actuation performed.
        """
        self._ensure_cheats()

        # Safety Envelope 1: Ammo Starvation (<= 5 bullets remaining)
        # A player with no ammo cannot fight back; always guarantee a resource drop.
        if player_ammo is not None and player_ammo <= 5.0:
            action = DDAAction.INCREASE_RESOURCES

        # Safety Envelope 2: Critical Distress (<35 HP)
        # Strictly protect player from dying; convert all actions into emergency aid.
        elif player_health is not None and player_health < 35.0:
            action = DDAAction.INCREASE_RESOURCES

        difficulty_state.apply_action(action)

        if action == DDAAction.NO_CHANGE:
            return "No change (maintained current parameters)"

        elif action == DDAAction.INCREASE_DIFFICULTY:
            # Spawn high-tier projectile monsters
            threats = ["Cacodemon", "Revenant", "HellKnight"]
            import random
            chosen_threat = random.choice(threats)
            try:
                self.game.send_game_command(f"summon {chosen_threat}")
                return f"Escalated challenge: Spawned {chosen_threat}!"
            except Exception as e:
                return f"Challenge escalation failed: {e}"

        elif action == DDAAction.DECREASE_DIFFICULTY:
            try:
                # Provide instant health stabilization + physical pickup
                self.game.send_game_command("give health 15")
                self.game.send_game_command("give clip 20")
                self.game.send_game_command("summon Stimpack")
                return "Assisted player: Dropped emergency Stimpack (+15 HP & 20 Ammo)"
            except Exception as e:
                return f"Difficulty reduction failed: {e}"

        elif action == DDAAction.INCREASE_ENEMY_SPAWN:
            # Intelligently scale spawned monster based on current player health
            try:
                if player_health is not None and player_health < 55.0:
                    self.game.send_game_command("summon DoomImp")
                    return "Pacing increased: Spawned 1 Imp"
                else:
                    self.game.send_game_command("summon Demon")
                    return "Pacing increased: Spawned 1 Demon"
            except Exception as e:
                return f"Enemy spawn failed: {e}"

        elif action == DDAAction.DECREASE_ENEMY_SPAWN:
            try:
                self.game.send_game_command("kill monsters")
                return "Pacing eased: Thinned hostile monsters"
            except Exception as e:
                return f"Enemy reduction failed: {e}"

        elif action == DDAAction.INCREASE_RESOURCES:
            try:
                # 1. Give direct replenishment so stationary players (e.g. defend_the_center)
                # and active runners instantly receive vital relief on their HUD
                self.game.send_game_command("give health 25")
                self.game.send_game_command("give clip 50")
                # 2. Also summon physical pickups in the 3D level
                self.game.send_game_command("summon Medikit")
                self.game.send_game_command("summon ClipBox")
                return "Emergency Aid: Delivered +25 HP Medikit & 50 Ammo (Inventory & Spawns)"
            except Exception as e:
                return f"Resource drop failed: {e}"

        elif action == DDAAction.DECREASE_RESOURCES:
            return "Restricted resources: Withholding pickup spawns"

        return "Unknown action"

