from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from ingestion.pipeline import ingest_video
from indexing.metadata_store import MetadataStore
from config.settings import settings

router = APIRouter(prefix="/ingest", tags=["ingest"])
db = MetadataStore(settings.sqlite_db_path)


class IngestRequest(BaseModel):
    video_path: str
    home_team: str
    away_team: str
    competition: Optional[str] = ""
    season: Optional[str] = ""
    match_date: Optional[str] = ""
    final_score: Optional[str] = ""
    language: Optional[str] = "en"


class IngestResponse(BaseModel):
    match_id: str
    status: str
    message: str


def _run_ingest(request: IngestRequest):
    match_metadata = {
        "home_team": request.home_team,
        "away_team": request.away_team,
        "competition": request.competition,
        "season": request.season,
        "match_date": request.match_date,
        "final_score": request.final_score,
        "language": request.language,
    }
    ingest_video(request.video_path, match_metadata)


@router.post("", response_model=IngestResponse)
async def start_ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    video_path = Path(request.video_path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    background_tasks.add_task(_run_ingest, request)

    return IngestResponse(
        match_id="pending",
        status="queued",
        message=f"Ingestion started for {video_path.name}. Check /matches for status.",
    )


@router.get("/matches")
async def list_matches():
    return db.list_matches()


@router.get("/matches/{match_id}")
async def get_match(match_id: str):
    match = db.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return match


@router.get("/matches/{match_id}/events")
async def get_match_events(match_id: str):
    match = db.get_match(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    return db.get_match_events(match_id)
