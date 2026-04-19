import sqlite3
import json
from pathlib import Path
from typing import List, Optional

from indexing.event_schema import FootballEvent, MatchMetadata


class MetadataStore:
    def __init__(self, db_path: str = "data/football.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id TEXT PRIMARY KEY,
                video_file TEXT NOT NULL,
                home_team TEXT,
                away_team TEXT,
                competition TEXT,
                competition_key TEXT,
                season TEXT,
                match_date TEXT,
                final_score TEXT,
                language TEXT,
                video_duration_seconds REAL,
                processing_status TEXT,
                processing_errors TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                video_file TEXT NOT NULL,
                match_id TEXT,
                timestamp_seconds REAL,
                timestamp_end_seconds REAL,
                event_type TEXT,
                is_replay INTEGER,
                primary_player TEXT,
                primary_player_key TEXT,
                secondary_player TEXT,
                team TEXT,
                team_key TEXT,
                opposing_team TEXT,
                opposing_team_key TEXT,
                competition TEXT,
                competition_key TEXT,
                season TEXT,
                match_date TEXT,
                home_team TEXT,
                away_team TEXT,
                score_at_event TEXT,
                match_minute INTEGER,
                commentary_quote TEXT,
                event_description TEXT,
                detection_method TEXT,
                confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches(match_id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_match ON events(match_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_player ON events(primary_player_key);
        """)
        self._conn.commit()

    def save_match(self, meta: MatchMetadata):
        self._conn.execute(
            """
            INSERT OR REPLACE INTO matches
            (match_id, video_file, home_team, away_team, competition, competition_key,
             season, match_date, final_score, language, video_duration_seconds,
             processing_status, processing_errors)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                meta.match_id, meta.video_file, meta.home_team, meta.away_team,
                meta.competition, meta.competition_key, meta.season, meta.match_date,
                meta.final_score, meta.language, meta.video_duration_seconds,
                meta.processing_status, json.dumps(meta.processing_errors),
            ),
        )
        self._conn.commit()

    def update_match_status(self, match_id: str, status: str):
        self._conn.execute(
            "UPDATE matches SET processing_status = ? WHERE match_id = ?",
            (status, match_id),
        )
        self._conn.commit()

    def save_events(self, events: List[FootballEvent]):
        rows = [
            (
                e.event_id, e.video_file, e.match_id, e.timestamp_seconds,
                e.timestamp_end_seconds, e.event_type.value, int(e.is_replay),
                e.primary_player, e.primary_player_key, e.secondary_player,
                e.team, e.team_key, e.opposing_team, e.opposing_team_key,
                e.competition, e.competition_key, e.season, e.match_date,
                e.home_team, e.away_team, e.score_at_event, e.match_minute,
                e.commentary_quote, e.event_description,
                e.detection_method, e.confidence,
            )
            for e in events
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO events
            (event_id, video_file, match_id, timestamp_seconds, timestamp_end_seconds,
             event_type, is_replay, primary_player, primary_player_key, secondary_player,
             team, team_key, opposing_team, opposing_team_key, competition, competition_key,
             season, match_date, home_team, away_team, score_at_event, match_minute,
             commentary_quote, event_description, detection_method, confidence)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        self._conn.commit()

    def get_match(self, match_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM matches WHERE match_id = ?", (match_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_matches(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM matches ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_match_events(self, match_id: str) -> List[dict]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE match_id = ? ORDER BY timestamp_seconds",
            (match_id,),
        ).fetchall()
        return [dict(r) for r in rows]
