# audio/

Put the podcast episodes here as audio files (`.mp3`, `.m4a`, `.wav`, `.flac`,
`.ogg`, `.opus`). Then run:

```bash
make ingest
```

## What goes here

The three Fermi-supplied episode files, totalling no more than three hours.

Nothing in the system hard-codes a filename or an episode count — whatever
valid audio is present is discovered, validated, transcribed and indexed.
Episode titles are derived from filenames, so name the files the way you want
them to appear in the product and in citations:

```
audio/The Birth of the Quantum.mp3
audio/Shannon and the Birth of Information.mp3
audio/Bell's Theorem.mp3
```

A leading track number (`01 - Title.mp3`) is stripped automatically.

## Transcripts are always generated here, from this audio

The system runs its own ASR (faster-whisper) over these files. It never reads
YouTube or platform captions, transcript APIs, or any externally supplied
transcript. Delete `data/` and re-run `make ingest` to rebuild everything from
the audio alone.

## Development fixtures

If you do not have the episodes yet, generate synthetic audio to exercise the
pipeline end to end:

```bash
make fixtures   # macOS only; writes three short synthetic episodes here
make ingest
```

These are **not** Fermi podcast content — they are locally synthesised
recordings of scripts in `scripts/fixtures/`, used only to validate the
pipeline. Replace them with the real episodes and re-run `make ingest`;
nothing else changes.

## Note on re-ingestion

Ingestion is keyed on each file's SHA-256. Re-running `make ingest` re-uses
existing transcripts and only re-transcribes files whose content changed.
Removing a file from this folder discards its transcript, chunks and index
entries on the next run.
