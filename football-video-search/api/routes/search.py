from typing import Optional, List
from fastapi import APIRouter, Query

from retrieval.query_parser import parse_query
from retrieval.searcher import search, format_results
from config.settings import settings

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
async def search_events(
    q: str = Query(..., description="Natural language query, e.g. 'goals scored by Messi against Real Madrid'"),
    top_k: Optional[int] = Query(None, ge=1, le=100),
):
    """
    Search for football events by natural language query.
    Returns ranked list of video file + timestamp matches.
    """
    parsed = parse_query(q)
    results = search(parsed, top_k=top_k or settings.default_top_k)
    return {
        "query": q,
        "parsed_filters": {
            "event_types": parsed.event_types,
            "players": parsed.player_names,
            "teams": parsed.team_keys,
            "competition": parsed.competition_key,
        },
        "total_results": len(results),
        "results": format_results(results),
    }
