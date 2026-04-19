from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import uuid


class EventType(str, Enum):
    GOAL = "goal"
    OWN_GOAL = "own_goal"
    PENALTY = "penalty"
    MISSED_PENALTY = "missed_penalty"
    FOUL = "foul"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    CORNER = "corner"
    FREE_KICK = "free_kick"
    SUBSTITUTION = "substitution"
    SAVE = "save"
    SHOT = "shot"
    OFFSIDE = "offside"
    VAR_REVIEW = "var_review"
    INJURY = "injury"
    KICKOFF = "kickoff"
    HALF_TIME = "half_time"
    FULL_TIME = "full_time"


@dataclass
class FootballEvent:
    video_file: str
    timestamp_seconds: float
    event_type: EventType
    commentary_quote: str
    event_description: str

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_end_seconds: float = 0.0
    duration_seconds: float = 30.0

    is_replay: bool = False
    replay_of_event_id: Optional[str] = None

    primary_player: Optional[str] = None
    primary_player_key: Optional[str] = None
    secondary_player: Optional[str] = None
    team: Optional[str] = None
    team_key: Optional[str] = None
    opposing_team: Optional[str] = None
    opposing_team_key: Optional[str] = None

    match_id: Optional[str] = None
    competition: Optional[str] = None
    competition_key: Optional[str] = None
    season: Optional[str] = None
    match_date: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    score_at_event: Optional[str] = None
    match_minute: Optional[int] = None

    detection_method: str = "transcript"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    @property
    def timestamp_human(self) -> str:
        total = int(self.timestamp_seconds)
        h, remainder = divmod(total, 3600)
        m, s = divmod(remainder, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    @property
    def clip_start(self) -> float:
        return max(0.0, self.timestamp_seconds - 15.0)

    @property
    def clip_end(self) -> float:
        return self.timestamp_seconds + 45.0


@dataclass
class MatchMetadata:
    video_file: str
    home_team: str
    away_team: str

    match_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    competition: str = ""
    competition_key: str = ""
    season: str = ""
    match_date: str = ""
    final_score: str = ""
    language: str = "en"
    video_duration_seconds: float = 0.0
    processing_status: str = "pending"
    processing_errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
