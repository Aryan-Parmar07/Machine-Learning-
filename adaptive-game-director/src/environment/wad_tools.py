#!/usr/bin/env python3
"""Doom WAD and Scenario Map Manipulation Tools for the AI Game Director.

Powered by `omgifol` (`omg`), this module provides programmatic capabilities to:
1. Inspect standard and UDMF Doom WAD files (maps, things, monsters, pickups, player spawns).
2. Dynamically extract entity coordinates and sector bounds.
3. Inject custom Director supply drops, health stations, and monster ambush zones into maps.
4. Export customized DDA test maps for benchmark experiments.

Usage:
    # Inspect entities in a scenario WAD
    python src/environment/wad_tools.py inspect --wad deadly_corridor.wad

    # Create an augmented DDA variant with custom supply depots
    python src/environment/wad_tools.py augment --wad deadly_corridor.wad --output data/custom_wads/augmented_corridor.wad
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import omg
import vizdoom as vzd

from src.utils.logger import get_logger
from scripts.play_human import resolve_scenario_path

logger = get_logger("WadTools")

# Standard Doom EdNums (Thing types)
DOOM_THING_TYPES = {
    1: "Player 1 Start",
    2: "Player 2 Start",
    3: "Player 3 Start",
    4: "Player 4 Start",
    11: "Deathmatch Start",
    # Monsters
    9: "Shotgun Guy",
    3004: "Zombieman",
    65: "Chaingun Guy",
    3001: "Doom Imp",
    3002: "Pinky Demon",
    3003: "Baron of Hell",
    3005: "Cacodemon",
    3006: "Lost Soul",
    66: "Revenant",
    67: "Mancubus",
    68: "Arachnotron",
    69: "Hell Knight",
    71: "Pain Elemental",
    64: "Archvile",
    16: "Cyberdemon",
    7: "Spider Mastermind",
    # Pickups & Weapons
    2001: "Shotgun",
    2002: "Chaingun",
    2003: "Rocket Launcher",
    2004: "Plasma Rifle",
    2005: "Chainsaw",
    2006: "BFG 9000",
    82: "Super Shotgun",
    2007: "Ammo Clip",
    2008: "Shotgun Shells",
    2010: "Rocket Box",
    2046: "Clip Box (Bullet Ammo)",
    2047: "Cell Pack (Plasma Ammo)",
    2048: "Box of Shells",
    2011: "Stimpack (+10 HP)",
    2012: "Medikit (+25 HP)",
    2014: "Health Bonus (+1 HP)",
    2015: "Armor Bonus (+1 Armor)",
    2018: "Green Armor",
    2019: "Blue MegaArmor",
    2022: "Invulnerability Sphere",
    2023: "Berserk Pack (+100 HP)",
    2024: "Invisibility Blur Sphere",
    2025: "Radiation Suit",
    2026: "Computer Map",
}


class WadInspector:
    """Inspects and manipulates Doom WAD files for DDA scenarios."""

    def __init__(self, wad_path: str) -> None:
        self.wad_path = Path(wad_path)
        if not self.wad_path.is_absolute():
            try:
                self.wad_path = Path(resolve_scenario_path(str(self.wad_path)))
            except Exception:
                # Fallback to local data dir
                self.wad_path = PROJECT_ROOT / "data" / wad_path

        if not self.wad_path.exists():
            raise FileNotFoundError(f"WAD file not found at: {self.wad_path}")

        self.wad = omg.WAD(str(self.wad_path))
        self.is_udmf = bool(self.wad.udmfmaps)
        self.map_names = list(self.wad.udmfmaps.keys()) if self.is_udmf else list(self.wad.maps.keys())

    def get_map_editor(self, map_name: Optional[str] = None) -> Tuple[str, Any]:
        """Returns the appropriate MapEditor (standard or UDMF) for a given map name."""
        if not self.map_names:
            raise ValueError(f"WAD at {self.wad_path} contains no maps (might be a resource/DECORATE patch WAD).")

        target_map = map_name if map_name else self.map_names[0]
        if self.is_udmf:
            return target_map, omg.UMapEditor(self.wad.udmfmaps[target_map])
        else:
            return target_map, omg.MapEditor(self.wad.maps[target_map])

    def summarize_entities(self, map_name: Optional[str] = None) -> Dict[str, Any]:
        """Counts and catalogs all things/monsters/items present in the map."""
        target_map, editor = self.get_map_editor(map_name)

        things_catalog: List[Dict[str, Any]] = []
        monster_count = 0
        pickup_count = 0
        player_starts = 0

        for t in editor.things:
            t_type = int(t.type)
            name = DOOM_THING_TYPES.get(t_type, f"Custom/Unknown Thing ({t_type})")
            is_player = t_type in [1, 2, 3, 4, 11]
            is_monster = t_type in [9, 3004, 65, 3001, 3002, 3003, 3005, 3006, 66, 67, 68, 69, 71, 64, 16, 7]
            is_pickup = t_type >= 2001 and t_type <= 2048

            if is_player:
                player_starts += 1
            elif is_monster:
                monster_count += 1
            elif is_pickup:
                pickup_count += 1

            things_catalog.append({
                "type_id": t_type,
                "name": name,
                "x": float(t.x),
                "y": float(t.y),
                "angle": getattr(t, "angle", 0),
            })

        return {
            "wad_path": str(self.wad_path),
            "map_name": target_map,
            "format": "UDMF" if self.is_udmf else "DOOM Binary",
            "total_things": len(editor.things),
            "total_sectors": len(editor.sectors),
            "player_starts": player_starts,
            "monsters": monster_count,
            "pickups": pickup_count,
            "things": things_catalog,
        }

    def inject_supply_drop(
        self,
        x: float,
        y: float,
        item_type: str = "medikit",
        map_name: Optional[str] = None,
    ) -> None:
        """Injects a new pickup thing at specified (x, y) coordinates."""
        target_map, editor = self.get_map_editor(map_name)

        item_type_map = {
            "medikit": 2012,
            "stimpack": 2011,
            "clipbox": 2046,
            "shells": 2048,
            "greenarmor": 2018,
            "imp": 3001,
            "demon": 3002,
        }
        thing_type = item_type_map.get(item_type.lower(), 2012)

        if self.is_udmf:
            new_thing = omg.UThing()
            new_thing.type = thing_type
            new_thing.x = float(x)
            new_thing.y = float(y)
            new_thing.angle = 0
            editor.things.append(new_thing)
            self.wad.udmfmaps[target_map] = editor.to_lumps()
        else:
            new_thing = omg.Thing()
            new_thing.type = thing_type
            new_thing.x = int(x)
            new_thing.y = int(y)
            new_thing.angle = 0
            editor.things.append(new_thing)
            self.wad.maps[target_map] = editor.to_lumps()

        logger.info(f"Injected {DOOM_THING_TYPES.get(thing_type)} at ({x:.1f}, {y:.1f}) in {target_map}")

    def save_to_file(self, output_path: str) -> Path:
        """Saves the modified WAD to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.wad.to_file(str(out))
        logger.info(f"Saved WAD with dynamic DDA edits to: {out}")
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Doom WAD Inspection and DDA Map Tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect entities in a WAD file")
    inspect_parser.add_argument("--wad", type=str, required=True, help="WAD filename or path")
    inspect_parser.add_argument("--map", type=str, default=None, help="Map lump name (e.g. MAP01)")

    # Augment command
    aug_parser = subparsers.add_parser("augment", help="Inject DDA Director supply caches into a map")
    aug_parser.add_argument("--wad", type=str, required=True, help="Input WAD filename")
    aug_parser.add_argument("--output", type=str, required=True, help="Output WAD path")

    args = parser.parse_args()

    if args.command == "inspect":
        inspector = WadInspector(args.wad)
        summary = inspector.summarize_entities(args.map)
        print("=" * 80)
        print(f"WAD Entity Analysis: {summary['wad_path']} [{summary['format']}]")
        print(f"Map: {summary['map_name']} | Sectors: {summary['total_sectors']} | Things: {summary['total_things']}")
        print(f"Player Starts: {summary['player_starts']} | Monsters: {summary['monsters']} | Pickups: {summary['pickups']}")
        print("-" * 80)
        print(f"{'Thing Type ID':15s} | {'Entity Name':30s} | {'Coordinates (X, Y)':25s}")
        print("-" * 80)
        for t in summary["things"][:25]:
            print(f"{t['type_id']:<15d} | {t['name']:30s} | ({t['x']:8.1f}, {t['y']:8.1f})")
        if len(summary["things"]) > 25:
            print(f"... and {len(summary['things']) - 25} more entities.")
        print("=" * 80)

    elif args.command == "augment":
        inspector = WadInspector(args.wad)
        # Add emergency supply stations along the corridor / arena
        inspector.inject_supply_drop(x=400.0, y=0.0, item_type="medikit")
        inspector.inject_supply_drop(x=800.0, y=0.0, item_type="clipbox")
        out_path = inspector.save_to_file(args.output)
        print(f"Augmented DDA scenario created at: {out_path}")


if __name__ == "__main__":
    main()
