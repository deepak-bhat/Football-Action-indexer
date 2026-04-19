import subprocess
from pathlib import Path


def extract_audio(video_path: str, output_dir: str = "/tmp") -> str:
    """Extract 16kHz mono WAV from video. Returns path to WAV file."""
    video = Path(video_path)
    wav_path = Path(output_dir) / f"{video.stem}.wav"

    if wav_path.exists():
        return str(wav_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    return str(wav_path)


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path)
    ]
    import json
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    data = json.loads(result.stdout)
    return float(data.get("format", {}).get("duration", 0))
