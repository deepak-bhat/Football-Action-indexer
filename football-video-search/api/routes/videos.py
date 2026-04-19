import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

from config.settings import settings

router = APIRouter(prefix="/videos", tags=["videos"])


@router.get("/{video_name:path}/stream")
async def stream_video(video_name: str):
    """
    Stream a video file with byte-range support (required for Video.js seeking).
    """
    video_path = Path(settings.video_base_dir) / video_name
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=str(video_path),
        media_type=_media_type(video_path.suffix),
        headers={"Accept-Ranges": "bytes"},
    )


def _media_type(ext: str) -> str:
    return {
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }.get(ext.lower(), "video/mp4")
