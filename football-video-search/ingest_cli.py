#!/usr/bin/env python3
"""
CLI tool to ingest a single video file without running the API server.

Usage:
  python ingest_cli.py path/to/match.mp4 \
    --home "FC Barcelona" --away "Real Madrid" \
    --competition "Champions League" --season "2010-11" \
    --date "2011-05-03"
"""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Ingest a football video into the RAG system")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--home", required=True, help="Home team name")
    parser.add_argument("--away", required=True, help="Away team name")
    parser.add_argument("--competition", default="", help="Competition name")
    parser.add_argument("--season", default="", help="Season (e.g. 2010-11)")
    parser.add_argument("--date", default="", help="Match date (e.g. 2011-05-03)")
    parser.add_argument("--score", default="", help="Final score (e.g. 2-0)")
    parser.add_argument("--language", default="en", help="Commentary language code")
    parser.add_argument("--wav-cache", default="/tmp", help="Directory for WAV + transcript cache")
    args = parser.parse_args()

    from ingestion.pipeline import ingest_video

    match_metadata = {
        "home_team": args.home,
        "away_team": args.away,
        "competition": args.competition,
        "season": args.season,
        "match_date": args.date,
        "final_score": args.score,
        "language": args.language,
    }

    try:
        meta = ingest_video(args.video, match_metadata, wav_cache_dir=args.wav_cache)
        print(f"\nIngestion complete!")
        print(f"  Match ID : {meta.match_id}")
        print(f"  Status   : {meta.processing_status}")
        print(f"  Duration : {meta.video_duration_seconds:.0f}s")
    except Exception as exc:
        print(f"\nIngestion failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
