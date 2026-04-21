from typing import List, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    ScoredPoint,
)

from config.settings import settings
from indexing.event_schema import FootballEvent


@dataclass
class SearchResult:
    event_id: str
    video_file: str
    timestamp_seconds: float
    timestamp_human: str
    event_type: str
    primary_player: Optional[str]
    team: Optional[str]
    opposing_team: Optional[str]
    match_id: Optional[str]
    competition: Optional[str]
    season: Optional[str]
    match_date: Optional[str]
    home_team: Optional[str]
    away_team: Optional[str]
    score_at_event: Optional[str]
    match_minute: Optional[int]
    event_description: str
    commentary_quote: str
    confidence: float
    score: float
    clip_start: float
    clip_end: float


class VectorStore:
    def __init__(self):
        if settings.qdrant_host:
            # Remote server mode (Docker or standalone Qdrant binary)
            self._client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
            )
        else:
            # Embedded mode: Qdrant runs in-process, persists to local directory.
            # No Docker, no server binary required.
            import os
            os.makedirs(settings.qdrant_path, exist_ok=True)
            self._client = QdrantClient(path=settings.qdrant_path)

        self._collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=settings.embedding_dims,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_events(self, events: List[FootballEvent]) -> None:
        points = []
        for event in events:
            if not hasattr(event, "embedding_vector") or event.embedding_vector is None:
                continue
            payload = {
                "event_id": event.event_id,
                "video_file": event.video_file,
                "timestamp_seconds": event.timestamp_seconds,
                "timestamp_human": _fmt_timestamp(event.timestamp_seconds),
                "event_type": event.event_type.value,
                "is_replay": event.is_replay,
                "primary_player": event.primary_player,
                "primary_player_key": event.primary_player_key,
                "team": event.team,
                "team_key": event.team_key,
                "opposing_team": event.opposing_team,
                "opposing_team_key": event.opposing_team_key,
                "match_id": event.match_id,
                "competition": event.competition,
                "competition_key": event.competition_key,
                "season": event.season,
                "match_date": event.match_date,
                "home_team": event.home_team,
                "away_team": event.away_team,
                "score_at_event": event.score_at_event,
                "match_minute": event.match_minute,
                "event_description": event.event_description,
                "commentary_quote": event.commentary_quote,
                "confidence": event.confidence,
                "clip_start": max(0.0, event.timestamp_seconds - 15.0),
                "clip_end": event.timestamp_seconds + 45.0,
            }
            points.append(
                PointStruct(
                    id=_event_id_to_int(event.event_id),
                    vector=event.embedding_vector,
                    payload=payload,
                )
            )
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        query_vector: List[float],
        event_types: Optional[List[str]] = None,
        players: Optional[List[str]] = None,
        teams: Optional[List[str]] = None,
        competition_key: Optional[str] = None,
        exclude_replays: bool = True,
        top_k: int = 20,
    ) -> List[SearchResult]:
        must = []

        if exclude_replays:
            must.append(FieldCondition(key="is_replay", match=MatchValue(value=False)))

        if event_types:
            must.append(FieldCondition(key="event_type", match=MatchAny(any=event_types)))

        if players:
            # Match on normalized key (lowercase slug) for robustness
            player_keys = [p.lower().replace(" ", "_") for p in players]
            must.append(
                FieldCondition(key="primary_player_key", match=MatchAny(any=player_keys))
            )

        if teams:
            team_keys = [t.lower().replace(" ", "_") for t in teams]
            must.append(FieldCondition(key="team_key", match=MatchAny(any=team_keys)))

        if competition_key:
            must.append(
                FieldCondition(key="competition_key", match=MatchValue(value=competition_key))
            )

        query_filter = Filter(must=must) if must else None

        hits: List[ScoredPoint] = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=settings.min_score_threshold,
        )

        return [_point_to_result(h) for h in hits]

    def get_match_events(self, match_id: str) -> List[dict]:
        results, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="match_id", match=MatchValue(value=match_id))]
            ),
            limit=500,
            with_payload=True,
        )
        return [r.payload for r in results]

    def delete_match(self, match_id: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="match_id", match=MatchValue(value=match_id))]
            ),
        )


def _fmt_timestamp(seconds: float) -> str:
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _event_id_to_int(event_id: str) -> int:
    """Convert UUID string to stable int for Qdrant point ID."""
    import hashlib
    return int(hashlib.md5(event_id.encode()).hexdigest(), 16) % (2**63)


def _point_to_result(point: ScoredPoint) -> SearchResult:
    p = point.payload
    return SearchResult(
        event_id=p.get("event_id", ""),
        video_file=p.get("video_file", ""),
        timestamp_seconds=p.get("timestamp_seconds", 0.0),
        timestamp_human=p.get("timestamp_human", "0:00"),
        event_type=p.get("event_type", ""),
        primary_player=p.get("primary_player"),
        team=p.get("team"),
        opposing_team=p.get("opposing_team"),
        match_id=p.get("match_id"),
        competition=p.get("competition"),
        season=p.get("season"),
        match_date=p.get("match_date"),
        home_team=p.get("home_team"),
        away_team=p.get("away_team"),
        score_at_event=p.get("score_at_event"),
        match_minute=p.get("match_minute"),
        event_description=p.get("event_description", ""),
        commentary_quote=p.get("commentary_quote", ""),
        confidence=p.get("confidence", 0.0),
        score=point.score,
        clip_start=p.get("clip_start", 0.0),
        clip_end=p.get("clip_end", 0.0),
    )
