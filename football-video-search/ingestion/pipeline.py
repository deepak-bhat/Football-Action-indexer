"""
Main ingestion pipeline: video → events → indexed in Qdrant + SQLite.
"""
import logging
from pathlib import Path
from typing import List

from indexing.event_schema import FootballEvent, MatchMetadata
from indexing.metadata_store import MetadataStore
from indexing.vector_store import VectorStore
from indexing.embedder import embed_events
from ingestion.audio_extractor import extract_audio, get_video_duration
from ingestion.transcriber import transcribe, build_windows
from ingestion.event_detector import detect_events_in_window
from ingestion.entity_normalizer import normalize_player, normalize_team, normalize_competition
from config.settings import settings

log = logging.getLogger(__name__)


def _deduplicate_events(events: List[FootballEvent]) -> List[FootballEvent]:
    """
    Collapse duplicate detections of the same event within replay_window_seconds.
    Keeps the earliest occurrence (live event); marks later ones as replays.
    """
    window = settings.replay_window_seconds
    events_sorted = sorted(events, key=lambda e: e.timestamp_seconds)
    kept = []
    suppressed_ids = set()

    for i, event in enumerate(events_sorted):
        if event.event_id in suppressed_ids:
            continue
        if not event.is_replay:
            kept.append(event)
        for j in range(i + 1, len(events_sorted)):
            other = events_sorted[j]
            if other.timestamp_seconds - event.timestamp_seconds > window:
                break
            if (
                other.event_type == event.event_type
                and other.primary_player_key == event.primary_player_key
                and other.event_id not in suppressed_ids
            ):
                suppressed_ids.add(other.event_id)

    return kept


def _normalize_event(event: FootballEvent) -> FootballEvent:
    if event.primary_player:
        canonical, key = normalize_player(event.primary_player)
        event.primary_player = canonical
        event.primary_player_key = key

    if event.team:
        canonical, key = normalize_team(event.team)
        event.team = canonical
        event.team_key = key

    if event.opposing_team:
        canonical, key = normalize_team(event.opposing_team)
        event.opposing_team = canonical
        event.opposing_team_key = key

    if event.competition:
        canonical, key = normalize_competition(event.competition)
        event.competition = canonical
        event.competition_key = key

    return event


def ingest_video(
    video_path: str,
    match_metadata: dict,
    wav_cache_dir: str = "/tmp",
) -> MatchMetadata:
    """
    Full ingestion pipeline for one video file.

    match_metadata keys (all optional except home_team / away_team):
      home_team, away_team, competition, season, match_date, final_score, language
    """
    video_path = str(Path(video_path).resolve())
    log.info("Starting ingestion: %s", video_path)

    meta = MatchMetadata(
        video_file=video_path,
        home_team=match_metadata.get("home_team", "Unknown"),
        away_team=match_metadata.get("away_team", "Unknown"),
        competition=match_metadata.get("competition", ""),
        season=match_metadata.get("season", ""),
        match_date=match_metadata.get("match_date", ""),
        final_score=match_metadata.get("final_score", ""),
        language=match_metadata.get("language", "en"),
    )

    db = MetadataStore(settings.sqlite_db_path)
    vs = VectorStore()

    try:
        # Stage 1: audio
        log.info("Extracting audio…")
        wav_path = extract_audio(video_path, wav_cache_dir)
        meta.video_duration_seconds = get_video_duration(video_path)

        # Stage 2: transcribe
        log.info("Transcribing audio…")
        segments = transcribe(wav_path, wav_cache_dir)

        # Stage 3: sliding-window event detection
        log.info("Detecting events…")
        windows = build_windows(
            segments,
            settings.transcript_window_seconds,
            settings.transcript_window_step_seconds,
            meta.video_duration_seconds,
        )

        match_context = (
            f"{meta.home_team} vs {meta.away_team}"
            f"{', ' + meta.competition if meta.competition else ''}"
            f"{', ' + meta.season if meta.season else ''}"
            f"{', ' + meta.match_date if meta.match_date else ''}"
        )

        raw_events: List[FootballEvent] = []
        for window in windows:
            try:
                found = detect_events_in_window(
                    window=window,
                    match_context=match_context,
                    match_id=meta.match_id,
                    video_file=video_path,
                    match_metadata=match_metadata,
                )
                raw_events.extend(found)
            except Exception as exc:
                log.warning("Event detection failed for window %.0f: %s", window["window_start"], exc)

        # Stage 4: normalize entities
        log.info("Normalizing entities…")
        normalized = [_normalize_event(e) for e in raw_events]

        # Stage 5: deduplicate / replay filtering
        log.info("Deduplicating %d raw events…", len(normalized))
        final_events = _deduplicate_events(normalized)
        log.info("%d events after deduplication", len(final_events))

        # Stage 6: embed + index
        log.info("Embedding events…")
        embed_events(final_events)

        log.info("Upserting to Qdrant…")
        vs.upsert_events(final_events)

        log.info("Saving to SQLite…")
        db.save_match(meta)
        db.save_events(final_events)

        meta.processing_status = "indexed"

    except Exception as exc:
        log.error("Ingestion failed: %s", exc, exc_info=True)
        meta.processing_status = "failed"
        meta.processing_errors.append(str(exc))
        db.save_match(meta)
        raise

    db.update_match_status(meta.match_id, meta.processing_status)
    log.info("Ingestion complete: %s", video_path)
    return meta
