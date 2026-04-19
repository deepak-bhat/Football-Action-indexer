import json
from pathlib import Path
from typing import Optional

from rapidfuzz import process, fuzz


_PLAYERS: list = []
_TEAMS: list = []
_ALIAS_TO_PLAYER: dict = {}
_ALIAS_TO_TEAM: dict = {}

_MAPS_DIR = Path(__file__).parent.parent / "config" / "entity_maps"


def _load_maps():
    global _PLAYERS, _TEAMS, _ALIAS_TO_PLAYER, _ALIAS_TO_TEAM
    if _PLAYERS and _TEAMS:
        return

    players_file = _MAPS_DIR / "players.json"
    teams_file = _MAPS_DIR / "teams.json"

    if players_file.exists():
        data = json.loads(players_file.read_text())
        _PLAYERS = data.get("players", [])
        for p in _PLAYERS:
            for alias in p["aliases"] + [p["canonical"]]:
                _ALIAS_TO_PLAYER[alias.lower()] = p

    if teams_file.exists():
        data = json.loads(teams_file.read_text())
        _TEAMS = data.get("teams", [])
        for t in _TEAMS:
            for alias in t["aliases"] + [t["canonical"]]:
                _ALIAS_TO_TEAM[alias.lower()] = t


def normalize_player(name: str, threshold: int = 80) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (canonical_name, normalized_key) or (None, None) if no match found.
    Uses exact lookup first, then fuzzy matching.
    """
    _load_maps()
    if not name:
        return None, None

    key = name.lower().strip()
    if key in _ALIAS_TO_PLAYER:
        p = _ALIAS_TO_PLAYER[key]
        return p["canonical"], p["normalized_key"]

    if not _ALIAS_TO_PLAYER:
        return name, name.lower().replace(" ", "_")

    match, score, _ = process.extractOne(
        key,
        list(_ALIAS_TO_PLAYER.keys()),
        scorer=fuzz.token_sort_ratio,
    )
    if score >= threshold:
        p = _ALIAS_TO_PLAYER[match]
        return p["canonical"], p["normalized_key"]

    return name, name.lower().replace(" ", "_")


def normalize_team(name: str, threshold: int = 75) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (canonical_name, normalized_key) or (None, None) if no match.
    """
    _load_maps()
    if not name:
        return None, None

    key = name.lower().strip()
    if key in _ALIAS_TO_TEAM:
        t = _ALIAS_TO_TEAM[key]
        return t["canonical"], t["normalized_key"]

    if not _ALIAS_TO_TEAM:
        return name, name.lower().replace(" ", "_")

    match, score, _ = process.extractOne(
        key,
        list(_ALIAS_TO_TEAM.keys()),
        scorer=fuzz.token_sort_ratio,
    )
    if score >= threshold:
        t = _ALIAS_TO_TEAM[match]
        return t["canonical"], t["normalized_key"]

    return name, name.lower().replace(" ", "_")


def normalize_competition(name: str) -> tuple[str, str]:
    """Minimal normalization: lowercase slug for common competitions."""
    mapping = {
        "champions league": ("UEFA Champions League", "champions_league"),
        "ucl": ("UEFA Champions League", "champions_league"),
        "europa league": ("UEFA Europa League", "europa_league"),
        "premier league": ("Premier League", "premier_league"),
        "pl": ("Premier League", "premier_league"),
        "la liga": ("La Liga", "la_liga"),
        "bundesliga": ("Bundesliga", "bundesliga"),
        "serie a": ("Serie A", "serie_a"),
        "ligue 1": ("Ligue 1", "ligue_1"),
        "fa cup": ("FA Cup", "fa_cup"),
        "world cup": ("FIFA World Cup", "world_cup"),
        "euro": ("UEFA Euro", "euro"),
    }
    key = name.lower().strip()
    for k, v in mapping.items():
        if k in key:
            return v
    return name, name.lower().replace(" ", "_")
