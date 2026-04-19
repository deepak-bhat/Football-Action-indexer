import json
import re
from typing import Optional
from dataclasses import dataclass, field

from openai import AzureOpenAI

from config.settings import settings

_client = AzureOpenAI(
    azure_endpoint=settings.azure_openai_endpoint,
    api_key=settings.azure_openai_api_key,
    api_version=settings.azure_openai_api_version,
)

_SYSTEM = "You are a football search assistant. Extract structured filters from user queries. Return only valid JSON."

_TEMPLATE = """Parse this football video search query into structured filters.

Query: "{query}"

Return JSON:
{{
  "event_types": ["goal", "foul", ...] or null for all types,
  "player_keys": ["messi_lionel", ...] normalized underscore slugs or null,
  "player_names": ["Lionel Messi", ...] canonical names or null,
  "team_keys": ["fc_barcelona", ...] or null,
  "opposing_team_keys": ["real_madrid", ...] or null,
  "competition_key": "champions_league" or null,
  "season": "2010-11" or null,
  "match_minute_min": integer or null,
  "match_minute_max": integer or null,
  "semantic_query": "expanded natural language string to embed for vector search"
}}

Valid event_types: goal, own_goal, penalty, missed_penalty, foul, yellow_card, red_card,
  corner, free_kick, substitution, save, shot, offside, var_review, injury

For semantic_query: expand with synonyms, related terms, and football-specific vocabulary.
Example: "Messi goals" → "Lionel Messi Leo goal scored net strike finish celebration Champions League La Liga"
"""


@dataclass
class ParsedQuery:
    original: str
    event_types: Optional[list] = None
    player_keys: Optional[list] = None
    player_names: Optional[list] = None
    team_keys: Optional[list] = None
    opposing_team_keys: Optional[list] = None
    competition_key: Optional[str] = None
    season: Optional[str] = None
    match_minute_min: Optional[int] = None
    match_minute_max: Optional[int] = None
    semantic_query: str = ""


def parse_query(user_query: str) -> ParsedQuery:
    prompt = _TEMPLATE.format(query=user_query)
    response = _client.chat.completions.create(
        model=settings.azure_chat_deployment,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to pure semantic search with original query
        return ParsedQuery(original=user_query, semantic_query=user_query)

    return ParsedQuery(
        original=user_query,
        event_types=data.get("event_types"),
        player_keys=data.get("player_keys"),
        player_names=data.get("player_names"),
        team_keys=data.get("team_keys"),
        opposing_team_keys=data.get("opposing_team_keys"),
        competition_key=data.get("competition_key"),
        season=data.get("season"),
        match_minute_min=data.get("match_minute_min"),
        match_minute_max=data.get("match_minute_max"),
        semantic_query=data.get("semantic_query") or user_query,
    )
