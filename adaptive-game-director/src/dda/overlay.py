"""Real-Time In-Game HUD Overlay for the Dynamic Difficulty Adjustment (DDA) Game Director.

Provides visual feedback in both human playthroughs and AI agent playthroughs,
illustrating in real-time how the AI Director assesses the player (Novice / Intermediate / Expert),
what difficulty treatment is currently enforced (Easy / Medium / Hard / Nightmare), the active
Flow State tension, and the real-time pacing actions being issued.
"""

import sys
import time
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

import pygame

# Mapping of native Doom skill level (1 to 5) to human-readable names and treatments
DOOM_SKILL_META = {
    1: {
        "name": "VERY EASY",
        "sub": "I'm Too Young to Die",
        "treatment": "Generous Aid & Minimal Pressure",
        "color": (52, 211, 153),      # Emerald
        "bg_accent": (6, 78, 59),
    },
    2: {
        "name": "EASY",
        "sub": "Hey, Not Too Rough",
        "treatment": "Assisted Flow & Forgiving Pacing",
        "color": (74, 222, 128),      # Light Green
        "bg_accent": (20, 83, 45),
    },
    3: {
        "name": "MEDIUM",
        "sub": "Hurt Me Plenty",
        "treatment": "Balanced Pacing & Flow Retention",
        "color": (251, 191, 36),      # Amber
        "bg_accent": (120, 53, 15),
    },
    4: {
        "name": "HARD",
        "sub": "Ultra-Violence",
        "treatment": "High Threat & Aggressive Spawns",
        "color": (248, 113, 113),     # Coral Red
        "bg_accent": (127, 29, 29),
    },
    5: {
        "name": "NIGHTMARE",
        "sub": "Nightmare!",
        "treatment": "Maximum Pressure & Lethal Spawns",
        "color": (192, 132, 252),     # Purple
        "bg_accent": (88, 28, 135),
    },
}

# Player skill tier presentation
SKILL_TIER_META = {
    "NOVICE": {
        "label": "NOVICE",
        "desc": "Protective Pacing",
        "color": (56, 189, 248),      # Sky Blue
    },
    "INTERMEDIATE": {
        "label": "INTERMEDIATE",
        "desc": "Standard Flow Tuning",
        "color": (52, 211, 153),      # Emerald Green
    },
    "EXPERT": {
        "label": "EXPERT / MASTER",
        "desc": "Elevated Pacing & Spawns",
        "color": (168, 85, 247),      # Purple
    },
}


class DirectorHUDOverlay:
    """Live Visual HUD Overlay for the AI Game Director.

    Can be rendered as:
    1. A standalone, borderless 'Always-on-Top' HUD banner pinned to the screen
       during human play (spectator mode) without stealing mouse/keyboard focus.
    2. An in-place surface drawn directly into existing Pygame windows (e.g.
       visual surveillance grid or integrated bot viewer).
    """

    def __init__(
        self,
        width: int = 860,
        height: int = 90,
        title: str = "ViZDoom AI Game Director - Live HUD",
        standalone: bool = True,
        game: Optional[Any] = None,
        screen_x: int = 80,
        screen_y: int = 12,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.standalone = standalone
        self.game = game
        self.screen_x = screen_x
        self.screen_y = screen_y

        self.last_in_game_msg_time = 0.0
        self.last_broadcast_state = ""

        # Ensure Pygame and Pygame font are initialized
        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        # Initialize fonts
        try:
            self.font_title = pygame.font.SysFont("Segoe UI, Arial", 14, bold=True)
            self.font_main = pygame.font.SysFont("Segoe UI, Arial", 12, bold=True)
            self.font_small = pygame.font.SysFont("Segoe UI, Arial", 10, bold=False)
            self.font_badge = pygame.font.SysFont("Consolas, Courier New", 11, bold=True)
        except Exception:
            self.font_title = pygame.font.Font(None, 18)
            self.font_main = pygame.font.Font(None, 15)
            self.font_small = pygame.font.Font(None, 13)
            self.font_badge = pygame.font.Font(None, 14)

        self.screen = None
        if self.standalone:
            self._init_standalone_window()

    def _init_standalone_window(self) -> None:
        """Creates a borderless topmost HUD banner window without stealing input focus."""
        try:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.NOFRAME)
            pygame.display.set_caption(self.title)

            # Pin window Always-On-Top in Windows without stealing focus
            if sys.platform == "win32":
                import ctypes
                hwnd = pygame.display.get_wm_info().get("window")
                if hwnd:
                    # HWND_TOPMOST = -1, SWP_NOACTIVATE = 0x0010, SWP_SHOWWINDOW = 0x0040
                    ctypes.windll.user32.SetWindowPos(
                        hwnd, -1, self.screen_x, self.screen_y, self.width, self.height, 0x0010 | 0x0040
                    )
        except Exception as e:
            # Fallback to standard window if topmost/noframe fails
            try:
                self.screen = pygame.display.set_mode((self.width, self.height))
            except Exception:
                self.standalone = False

    def draw(
        self,
        surface: pygame.Surface,
        rect: Optional[Tuple[int, int, int, int]] = None,
        health: float = 100.0,
        ammo: float = 50.0,
        killcount: int = 0,
        doom_skill: int = 3,
        skill_score: float = 0.5,
        skill_tier: str = "INTERMEDIATE",
        latest_action: str = "NO_CHANGE",
        latest_actuation: str = "Maintaining Flow",
        flow_reward: float = 0.0,
        is_emergency: bool = False,
    ) -> None:
        """Renders the comprehensive DDA Director status card onto the target surface."""
        if rect is None:
            rx, ry, rw, rh = 0, 0, surface.get_width(), surface.get_height()
        else:
            rx, ry, rw, rh = rect

        # Determine Flow State
        in_flow = 35.0 <= health <= 75.0
        is_distress = health < 35.0 or ammo <= 5.0
        is_comfort = health > 75.0

        if is_distress or is_emergency:
            flow_status = "DISTRESS (NEEDS AID)"
            border_color = (239, 68, 68)      # Alert Red
            flow_badge_color = (248, 113, 113)
        elif in_flow:
            flow_status = "FLOW CHANNEL (OPTIMAL)"
            border_color = (16, 185, 129)     # Flow Emerald
            flow_badge_color = (52, 211, 153)
        else:
            flow_status = "COMFORT (CHALLENGE UP)"
            border_color = (59, 130, 246)     # Cool Blue
            flow_badge_color = (96, 165, 250)

        skill_meta = DOOM_SKILL_META.get(doom_skill, DOOM_SKILL_META[3])
        tier_meta = SKILL_TIER_META.get(skill_tier.upper(), SKILL_TIER_META["INTERMEDIATE"])

        # 1. Base Dark Card
        card = pygame.Surface((rw, rh), pygame.SRCALPHA)
        card.fill((15, 23, 42, 245))  # Deep Slate Navy
        surface.blit(card, (rx, ry))

        # 2. Glowing Accent Border
        pygame.draw.rect(surface, border_color, (rx, ry, rw, rh), width=2, border_radius=4)

        # 3. Top Banner: Title & Flow State Pulse
        top_bar = pygame.Surface((rw - 4, 22), pygame.SRCALPHA)
        top_bar.fill((30, 41, 59, 230))
        surface.blit(top_bar, (rx + 2, ry + 2))

        # Title Tag
        title_surf = self.font_title.render("AI GAME DIRECTOR  |  ADAPTIVE DDA", True, (241, 245, 249))
        surface.blit(title_surf, (rx + 8, ry + 4))

        # Flow Channel Badge
        flow_surf = self.font_badge.render(f"[{flow_status}]", True, flow_badge_color)
        surface.blit(flow_surf, (rx + rw - flow_surf.get_width() - 10, ry + 4))

        # 4. Three Main Columns (Middle Row: Y = ry + 26)
        col_y = ry + 26
        col_w = (rw - 20) // 3

        # -------------------------------------------------------------
        # COLUMN 1: DIFFICULTY & TREATMENT (How the Director treats you)
        # -------------------------------------------------------------
        c1_x = rx + 8
        diff_badge = f"DIFFICULTY: {skill_meta['name']} (Skill {doom_skill}/5)"
        diff_surf = self.font_main.render(diff_badge, True, skill_meta["color"])
        surface.blit(diff_surf, (c1_x, col_y))

        treat_text = f"Treatment: {skill_meta['treatment'][:32]}"
        treat_surf = self.font_small.render(treat_text, True, (203, 213, 225))
        surface.blit(treat_surf, (c1_x, col_y + 16))

        # -------------------------------------------------------------
        # COLUMN 2: PLAYER ASSESSMENT & TENSION GAUGE
        # -------------------------------------------------------------
        c2_x = c1_x + col_w
        tier_text = f"DIRECTOR SEES: {tier_meta['label']} ({skill_score:.2f})"
        tier_surf = self.font_main.render(tier_text, True, tier_meta["color"])
        surface.blit(tier_surf, (c2_x, col_y))

        # Mini Health / Flow Channel Bar
        bar_x = c2_x
        bar_y = col_y + 18
        bar_w = min(170, col_w - 10)
        bar_h = 10
        pygame.draw.rect(surface, (51, 65, 85), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        # Highlight target flow zone [35, 75] in green
        zone_start = int(bar_w * 0.35)
        zone_w = int(bar_w * 0.40)
        pygame.draw.rect(surface, (16, 185, 129, 140), (bar_x + zone_start, bar_y, zone_w, bar_h))
        # Draw current player health marker
        health_ratio = max(0.0, min(1.0, health / 100.0))
        marker_x = bar_x + int(bar_w * health_ratio)
        pygame.draw.circle(surface, border_color, (marker_x, bar_y + bar_h // 2), 4)

        bar_label = self.font_small.render(f"Flow Target [35-75 HP] (HP: {health:.0f})", True, (148, 163, 184))
        surface.blit(bar_label, (bar_x + bar_w + 6, bar_y - 2))

        # -------------------------------------------------------------
        # COLUMN 3: LIVE VITALS & FLOW REWARD
        # -------------------------------------------------------------
        c3_x = c2_x + col_w + 30
        rwd_color = (52, 211, 153) if flow_reward >= 0 else (248, 113, 113)
        vitals_text = f"HP: {health:3.0f}  |  Ammo: {ammo:2.0f}  |  Kills: {killcount:2d}"
        vitals_surf = self.font_main.render(vitals_text, True, (241, 245, 249))
        surface.blit(vitals_surf, (c3_x, col_y))

        rwd_text = f"Flow Retention Reward: {flow_reward:+.2f}"
        rwd_surf = self.font_small.render(rwd_text, True, rwd_color)
        surface.blit(rwd_surf, (c3_x, col_y + 16))

        # 5. Bottom Row: Pacing Action & Live Actuation (Y = ry + 60)
        bot_y = ry + 60
        bot_bar = pygame.Surface((rw - 16, 22), pygame.SRCALPHA)
        bot_bar.fill((30, 41, 59, 180))
        surface.blit(bot_bar, (rx + 8, bot_y))

        # Action tag
        act_prefix = ">>> [EMERGENCY AID] " if is_emergency else "[DIRECTOR ACTION] "
        act_color = (248, 113, 113) if is_emergency else ((52, 211, 153) if "Aid" in latest_actuation or "Medikit" in latest_actuation else (251, 191, 36))
        action_str = f"{act_prefix}{latest_action}: {latest_actuation[:52]}"
        action_surf = self.font_badge.render(action_str, True, act_color)
        surface.blit(action_surf, (rx + 14, bot_y + 3))

    def update(
        self,
        health: float = 100.0,
        ammo: float = 50.0,
        killcount: int = 0,
        doom_skill: int = 3,
        skill_score: float = 0.5,
        skill_tier: str = "INTERMEDIATE",
        latest_action: str = "NO_CHANGE",
        latest_actuation: str = "Maintaining Flow",
        flow_reward: float = 0.0,
        is_emergency: bool = False,
        game: Optional[Any] = None,
    ) -> None:
        """Updates and repaints the HUD overlay, and synchronizes in-game ViZDoom announcements."""
        active_game = game or self.game

        # 1. Update standalone Pygame window
        if self.standalone and self.screen is not None:
            pygame.event.pump()
            self.draw(
                surface=self.screen,
                rect=(0, 0, self.width, self.height),
                health=health,
                ammo=ammo,
                killcount=killcount,
                doom_skill=doom_skill,
                skill_score=skill_score,
                skill_tier=skill_tier,
                latest_action=latest_action,
                latest_actuation=latest_actuation,
                flow_reward=flow_reward,
                is_emergency=is_emergency,
            )
            pygame.display.flip()

        # 2. In-Game Engine Text Synchronization
        # Broadcasts changes to ViZDoom engine screen via 'print' command
        now = time.time()
        skill_name = DOOM_SKILL_META.get(doom_skill, {}).get("name", "NORMAL")
        state_signature = f"{skill_name}_{skill_tier}_{latest_action}"

        should_broadcast = is_emergency or (
            state_signature != self.last_broadcast_state and (now - self.last_in_game_msg_time >= 2.5)
        )

        if active_game is not None and should_broadcast:
            self.last_broadcast_state = state_signature
            self.last_in_game_msg_time = now
            try:
                q = '"'
                if is_emergency:
                    msg = f"*** [AI DIRECTOR: AID DISPATCHED] HP:{health:.0f} Ammo:{ammo:.0f} ***"
                else:
                    msg = f"[AI DIRECTOR] DIFFICULTY: {skill_name} | SEES: {skill_tier} | {latest_action}"
                active_game.send_game_command(f"print {q}{msg}{q}")
            except Exception:
                pass

    def close(self) -> None:
        """Safely cleans up Pygame window."""
        if self.standalone and self.screen is not None:
            try:
                pygame.display.quit()
            except Exception:
                pass
            self.screen = None

