"""
Basic unit tests — no video files or API keys required.
Run: pytest tests/
"""
import pytest

from indexing.event_schema import FootballEvent, EventType
from ingestion.entity_normalizer import normalize_player, normalize_team
from ingestion.event_detector import _build_description, _opposing_team


def _make_event(**kwargs) -> FootballEvent:
    defaults = dict(
        video_file="test.mp4",
        timestamp_seconds=100.0,
        event_type=EventType.GOAL,
        commentary_quote="He scores!",
        event_description="Goal scored.",
    )
    defaults.update(kwargs)
    return FootballEvent(**defaults)


class TestEventSchema:
    def test_timestamp_human_minutes_only(self):
        e = _make_event(timestamp_seconds=185.0)
        assert e.timestamp_human == "3:05"

    def test_timestamp_human_with_hours(self):
        e = _make_event(timestamp_seconds=3665.0)
        assert e.timestamp_human == "1:01:05"

    def test_clip_start_never_negative(self):
        e = _make_event(timestamp_seconds=5.0)
        assert e.clip_start == 0.0

    def test_clip_end(self):
        e = _make_event(timestamp_seconds=100.0)
        assert e.clip_end == 145.0


class TestEntityNormalizer:
    def test_exact_alias_player(self):
        canonical, key = normalize_player("Messi")
        assert canonical == "Lionel Messi"
        assert key == "messi_lionel"

    def test_case_insensitive_player(self):
        canonical, _ = normalize_player("messi")
        assert canonical == "Lionel Messi"

    def test_unknown_player_passthrough(self):
        canonical, key = normalize_player("John Doe Unknown Player XYZ")
        assert canonical is not None

    def test_exact_alias_team(self):
        canonical, key = normalize_team("Barca")
        assert canonical == "FC Barcelona"
        assert key == "fc_barcelona"

    def test_unknown_team_passthrough(self):
        canonical, key = normalize_team("Some Obscure FC")
        assert canonical is not None


class TestDescriptionBuilder:
    def test_goal_description(self):
        desc = _build_description(
            event_type=EventType.GOAL,
            player="Messi",
            team="Barcelona",
            opposing_team="Real Madrid",
            match_metadata={"competition": "Champions League", "season": "2010-11"},
            match_minute=76,
            score="2-0",
        )
        assert "Goal scored" in desc
        assert "Messi" in desc
        assert "Barcelona" in desc
        assert "Real Madrid" in desc
        assert "76" in desc

    def test_opposing_team_detection(self):
        meta = {"home_team": "FC Barcelona", "away_team": "Real Madrid"}
        assert _opposing_team("FC Barcelona", meta) == "Real Madrid"
        assert _opposing_team("Real Madrid", meta) == "FC Barcelona"
        assert _opposing_team(None, meta) is None
