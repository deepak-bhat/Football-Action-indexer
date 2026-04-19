import json
import re
from typing import List

from openai import AzureOpenAI

from config.settings import settings
from indexing.event_schema import EventType, FootballEvent

_client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)

_SYSTEM_PROMPT = """You are an expert football analyst extracting structured events from commentary transcripts.
Return ONLY valid JSON — no markdown, no explanation."""

_USER_TEMPLATE = """Analyze this football commentary window and identify ALL events.

Match context: {match_context}
Window start (seconds): {window_start}
Commentary:
{text}

Return a JSON array (possibly empty). Each element:
{{
  "event_type": one of [goal, own_goal, penalty, missed_penalty, foul, yellow_card, red_card,
                        corner, free_kick, substitution, save, shot, offside, var_review,
                        injury, kickoff, half_time, full_time],
  "player": "player name or null",
  "team": "team name or null",
  "timestamp_offset": seconds_from_window_start (float),
  "confidence": 0.0-1.0,
  "commentary_quote": "the exact phrase that indicates this event",
  "score_at_event": "e.g. 1-0 or null",
  "match_minute": integer or null
}}

Rules:
- timestamp_offset must be between 0 and the window duration
- Only include events you are confident about (confidence >= 0.6)
- Replays use phrases like "let's look at that again", "watch the replay" — if detected, set event_type to match the original event but add "is_replay": true
- For substitutions include both players as "player_off: X, player_on: Y" in the quote
"""


def _parse_llm_response(raw: str) -> list:
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract first JSON array
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []


def detect_events_in_window(
    window: dict,
    match_context: str,
    match_id: str,
    video_file: str,
    match_metadata: dict,
) -> List[FootballEvent]:
    """Call Claude on one transcript window; return parsed FootballEvent list."""

    prompt = _USER_TEMPLATE.format(
        match_context=match_context,
        window_start=window["window_start"],
        text=window["text"],
    )

    response = _client.chat.completions.create(
        model=settings.azure_chat_deployment,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    raw_items = _parse_llm_response(response.choices[0].message.content)
    events = []

    for item in raw_items:
        if item.get("confidence", 0) < settings.min_event_confidence:
            continue

        try:
            event_type = EventType(item["event_type"])
        except ValueError:
            continue

        timestamp = window["window_start"] + float(item.get("timestamp_offset", 0))
        player = item.get("player")
        team = item.get("team")

        description = _build_description(
            event_type=event_type,
            player=player,
            team=team,
            opposing_team=_opposing_team(team, match_metadata),
            match_metadata=match_metadata,
            match_minute=item.get("match_minute"),
            score=item.get("score_at_event"),
        )

        event = FootballEvent(
            video_file=video_file,
            timestamp_seconds=timestamp,
            timestamp_end_seconds=timestamp + 30.0,
            event_type=event_type,
            commentary_quote=item.get("commentary_quote", ""),
            event_description=description,
            is_replay=bool(item.get("is_replay", False)),
            primary_player=player,
            team=team,
            opposing_team=_opposing_team(team, match_metadata),
            match_id=match_id,
            competition=match_metadata.get("competition", ""),
            season=match_metadata.get("season", ""),
            match_date=match_metadata.get("match_date", ""),
            home_team=match_metadata.get("home_team", ""),
            away_team=match_metadata.get("away_team", ""),
            score_at_event=item.get("score_at_event"),
            match_minute=item.get("match_minute"),
            confidence=float(item.get("confidence", 0.0)),
        )
        events.append(event)

    return events


def _opposing_team(team: str | None, meta: dict) -> str | None:
    if not team:
        return None
    home = meta.get("home_team", "")
    away = meta.get("away_team", "")
    if team.lower() in home.lower():
        return away or None
    if team.lower() in away.lower():
        return home or None
    return None


def _build_description(
    event_type: EventType,
    player: str | None,
    team: str | None,
    opposing_team: str | None,
    match_metadata: dict,
    match_minute: int | None,
    score: str | None,
) -> str:
    parts = []
    verb = {
        EventType.GOAL: "Goal scored",
        EventType.OWN_GOAL: "Own goal",
        EventType.PENALTY: "Penalty kick",
        EventType.MISSED_PENALTY: "Penalty missed",
        EventType.FOUL: "Foul committed",
        EventType.YELLOW_CARD: "Yellow card shown",
        EventType.RED_CARD: "Red card shown",
        EventType.CORNER: "Corner kick",
        EventType.FREE_KICK: "Free kick",
        EventType.SUBSTITUTION: "Substitution",
        EventType.SAVE: "Save made",
        EventType.SHOT: "Shot on target",
        EventType.OFFSIDE: "Offside",
        EventType.VAR_REVIEW: "VAR review",
        EventType.INJURY: "Injury stoppage",
        EventType.KICKOFF: "Kick-off",
        EventType.HALF_TIME: "Half-time",
        EventType.FULL_TIME: "Full-time",
    }.get(event_type, event_type.value)

    parts.append(verb)
    if player:
        parts.append(f"by {player}")
    if team:
        parts.append(f"for {team}")
    if opposing_team:
        parts.append(f"against {opposing_team}")

    competition = match_metadata.get("competition", "")
    season = match_metadata.get("season", "")
    if competition or season:
        parts.append(f"in {' '.join(filter(None, [competition, season]))}")

    if match_minute:
        parts.append(f"{match_minute}' minute")
    if score:
        parts.append(f"score {score}")

    return " ".join(parts) + "."
