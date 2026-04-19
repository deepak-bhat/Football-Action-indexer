import json
from pathlib import Path
from typing import List
from dataclasses import dataclass

from config.settings import settings


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def transcribe(wav_path: str, cache_dir: str = "/tmp") -> List[TranscriptSegment]:
    """
    Transcribe audio with faster-whisper. Returns segment list with timestamps.
    Caches transcript JSON next to the WAV to skip re-transcription on re-runs.
    """
    cache_path = Path(cache_dir) / f"{Path(wav_path).stem}_transcript.json"

    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        return [TranscriptSegment(**s) for s in raw]

    from faster_whisper import WhisperModel

    model = WhisperModel(
        settings.whisper_model_size,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )

    segments_iter, _ = model.transcribe(
        wav_path,
        beam_size=5,
        vad_filter=True,
    )

    segments = [
        TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
        for s in segments_iter
    ]

    cache_path.write_text(json.dumps([vars(s) for s in segments]))
    return segments


def build_windows(
    segments: List[TranscriptSegment],
    window_seconds: int,
    step_seconds: int,
    total_duration: float,
) -> List[dict]:
    """
    Slice transcript segments into overlapping time windows for event detection.
    Returns list of {window_start, window_end, text}.
    """
    windows = []
    t = 0.0
    while t < total_duration:
        window_end = t + window_seconds
        window_text = " ".join(
            s.text
            for s in segments
            if s.end >= t and s.start < window_end
        )
        if window_text.strip():
            windows.append({
                "window_start": t,
                "window_end": min(window_end, total_duration),
                "text": window_text,
            })
        t += step_seconds
    return windows
