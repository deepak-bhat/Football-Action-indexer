from typing import List

from indexing.embedder import embed_query
from indexing.vector_store import VectorStore, SearchResult
from retrieval.query_parser import ParsedQuery
from config.settings import settings


def search(parsed: ParsedQuery, top_k: int = None) -> List[SearchResult]:
    """
    Execute hybrid retrieval: embed the semantic query, apply metadata filters,
    return ranked SearchResult list.
    """
    top_k = top_k or settings.default_top_k
    query_vector = embed_query(parsed.semantic_query)

    vs = VectorStore()
    results = vs.search(
        query_vector=query_vector,
        event_types=parsed.event_types,
        players=parsed.player_keys,
        teams=parsed.team_keys,
        competition_key=parsed.competition_key,
        exclude_replays=True,
        top_k=top_k,
    )

    # Post-filter by match minute range if specified
    if parsed.match_minute_min is not None or parsed.match_minute_max is not None:
        results = _filter_by_minute(results, parsed.match_minute_min, parsed.match_minute_max)

    return results


def _filter_by_minute(
    results: List[SearchResult],
    min_minute: int | None,
    max_minute: int | None,
) -> List[SearchResult]:
    filtered = []
    for r in results:
        if r.match_minute is None:
            filtered.append(r)
            continue
        if min_minute is not None and r.match_minute < min_minute:
            continue
        if max_minute is not None and r.match_minute > max_minute:
            continue
        filtered.append(r)
    return filtered


def format_results(results: List[SearchResult]) -> List[dict]:
    """Convert SearchResult objects to JSON-serialisable dicts."""
    return [
        {
            "rank": i + 1,
            "video_file": r.video_file,
            "timestamp_seconds": r.timestamp_seconds,
            "timestamp_human": r.timestamp_human,
            "clip_start": r.clip_start,
            "clip_end": r.clip_end,
            "event_type": r.event_type,
            "player": r.primary_player,
            "team": r.team,
            "opposing_team": r.opposing_team,
            "match": _match_label(r),
            "match_minute": r.match_minute,
            "score_at_event": r.score_at_event,
            "description": r.event_description,
            "commentary_quote": r.commentary_quote,
            "confidence": round(r.confidence, 2),
            "relevance_score": round(r.score, 4),
        }
        for i, r in enumerate(results)
    ]


def _match_label(r: SearchResult) -> str:
    parts = []
    if r.home_team and r.away_team:
        parts.append(f"{r.home_team} vs {r.away_team}")
    if r.competition:
        parts.append(r.competition)
    if r.season:
        parts.append(r.season)
    if r.match_date:
        parts.append(r.match_date)
    return ", ".join(parts) if parts else "Unknown match"
