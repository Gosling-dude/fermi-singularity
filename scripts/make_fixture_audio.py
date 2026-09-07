#!/usr/bin/env python3
"""Generate development fixture audio from the scripts in scripts/fixtures/.

These are NOT Fermi podcast episodes. They are synthetic, locally generated
recordings whose only purpose is to let the pipeline be exercised end to end —
real audio in, our own ASR, real retrieval, real evaluation — on a machine
that does not yet have the supplied episodes.

Replace the files in audio/ with the real episodes and re-run `make ingest`;
nothing else in the system changes.

Usage:  python scripts/make_fixture_audio.py [--voice Samantha] [--rate 170]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "scripts" / "fixtures"
OUTPUT_DIR = ROOT / "audio"
VOICE_PREFERENCE = ["Samantha", "Daniel", "Karen", "Alex", "Fred"]


def pick_voice(requested: str | None) -> str:
    listing = subprocess.run(
        ["say", "-v", "?"], capture_output=True, text=True, check=True
    ).stdout
    available = {line.split()[0] for line in listing.splitlines() if line.strip()}
    for candidate in ([requested] if requested else []) + VOICE_PREFERENCE:
        if candidate and candidate in available:
            return candidate
    raise SystemExit(
        "No usable macOS voice found. Pass --voice with a name from `say -v '?'`."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", default=None, help="macOS voice name")
    parser.add_argument("--rate", type=int, default=170, help="words per minute")
    args = parser.parse_args()

    if sys.platform != "darwin" or not shutil.which("say"):
        print(
            "Fixture generation uses the macOS `say` command and only runs on "
            "macOS.\nOn other platforms, place any audio files in audio/ and "
            "run `make ingest`.",
            file=sys.stderr,
        )
        return 1
    if not shutil.which("ffmpeg"):
        print("ffmpeg is required. Install it with `brew install ffmpeg`.",
              file=sys.stderr)
        return 1

    scripts = sorted(FIXTURE_DIR.glob("*.txt"))
    if not scripts:
        print(f"No fixture scripts found in {FIXTURE_DIR}", file=sys.stderr)
        return 1

    voice = pick_voice(args.voice)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(scripts)} fixture episodes with voice '{voice}' "
          f"at {args.rate} wpm\n")

    for script in scripts:
        title = script.stem.replace("_", " ").title()
        aiff = OUTPUT_DIR / f"{script.stem}.aiff"
        mp3 = OUTPUT_DIR / f"{script.stem}.mp3"
        if mp3.exists():
            print(f"  = {mp3.name} already exists, skipping")
            continue
        print(f"  → speaking {title} ...", flush=True)
        subprocess.run(
            ["say", "-v", voice, "-r", str(args.rate), "-o", str(aiff),
             "-f", str(script)],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(aiff),
             "-codec:a", "libmp3lame", "-b:a", "64k", "-ac", "1", str(mp3)],
            check=True,
        )
        aiff.unlink(missing_ok=True)
        size_mb = mp3.stat().st_size / 1e6
        print(f"    wrote {mp3.name} ({size_mb:.1f} MB)")

    print("\nFixture audio ready in audio/. Run `make ingest` next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
