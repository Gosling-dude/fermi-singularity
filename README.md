# Fermi Podcast Companion

A conversational companion for a small collection of long-form physics
podcasts, built **from raw audio**. Ask a question, get a clear answer, and
click a timestamp to hear the exact moment that supports it — or be told
plainly that the episodes don't cover it.

Product reasoning: [`PRODUCT.md`](PRODUCT.md) · Evaluation, failures and the
measured improvement: [`EVAL.md`](EVAL.md)

---

## Why it exists

After listening to three hour-long episodes, a learner cannot find the moment
where an idea was explained, compare what two episodes said about it, or check
a half-remembered claim without scrubbing through an hour of audio. A plain
chatbot doesn't help either: it already knows a lot of physics and will answer
from that knowledge, leaving the learner unable to separate what the podcast
said from what the model filled in.

So the guiding principle is:

> **If the supplied audio does not support an answer, the system says so
> instead of guessing.**

---

## Architecture

```
audio/*.mp3                    the supplied episodes — read-only, never modified
      │
      ▼
  FFmpeg  ──────────────────►  16 kHz mono WAV        data/normalized/
      │                        (SHA-256 keyed, idempotent)
      ▼
  faster-whisper (medium)  ─►  timestamped transcript  data/transcripts/
      │                        our own ASR — no external captions, ever
      ▼
  timestamp-aware chunking ─►  ~75 s passages          data/chunks/
      │                        split on sentence + time, 15 s overlap
      ├──────────────┬───────────────────┐
      ▼              ▼                   ▼
  BGE embeddings   BM25            gate calibration
      │              │             (negative probes)
      ▼              ▼                   │
   ChromaDB      rank-bm25               │          data/index/
      │              │                   │
      └──────┬───────┘                   │
             ▼                           │
   Reciprocal Rank Fusion                │
             ▼                           │
   cross-encoder rerank  ◄───────────────┘
             │
             ├──── relevance below the calibrated gate? ──► "not covered"
             │                                              (no LLM call)
             ▼
        top-5 passages
             │
             ▼
   LLM + grounding prompt   (Anthropic or OpenAI)
             │
             ▼
   citation validation  ──► citations that don't overlap a
             │              retrieved passage are deleted
             ▼
   answer + episode/timestamp sources ──► click to hear the audio
```

**The pipeline starts at raw MP3 and does its own ASR.** No YouTube captions,
platform transcripts, caption APIs, or externally supplied transcripts are used
at any point. Delete `data/` and `make ingest` rebuilds everything from the
audio alone.

---

## Requirements

- **Python 3.11+**
- **FFmpeg** — `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Debian)
- **~2 GB disk** for the ASR, embedding and reranking models (downloaded once)
- **An API key** for chat and the evaluation judge — Anthropic or OpenAI.
  ASR, embeddings, reranking and the vector store all run locally and free.

---

## Quick start

```bash
make setup          # venv + dependencies + .env from the template
# put the podcast .mp3 files in audio/    (or: make fixtures — see below)
# add your API key to .env
make ingest         # transcribe + index   (~3× realtime on Apple silicon)
make chat           # start talking
```

`make doctor` reports exactly what is present and what is missing if anything
goes wrong.

### No audio yet?

```bash
make fixtures       # macOS: synthesises 3 short episodes into audio/
make ingest
```

These are locally generated recordings of the scripts in `scripts/fixtures/` —
**not** Fermi podcast content. They exist so the pipeline can be exercised end
to end. Replace them with the real episodes and re-run `make ingest`.

### No API key yet?

Everything except answer generation still works:

```bash
make search Q="ultraviolet catastrophe"    # hybrid retrieval, free
make eval-retrieval                        # full offline evaluation, free
make test
```

---

## Using it

```
make chat
```

```
You: Why did Planck introduce the quantum idea?
...answer with inline (Ep. "…" 12:30–13:10) citations...

Sources — verify these in the audio
  • The Birth of the Quantum  2:34–3:44

You: Explain that more simply.          ← follow-up, resolved from context
You: How do these episodes differ on uncertainty?   ← cross-episode
You: Take me to the part where they explain what a bit is.
You: What do these episodes say about black holes?
    → "The supplied episodes don't cover this."
```

Commands: `/episodes`, `/episode <n>` (scope to one episode), `/sources`,
`/clear`, `/help`, `/quit`.

### Web UI

```bash
make web     # http://127.0.0.1:8000
```

Same conversation, plus **clicking a source seeks the audio player to that
second** — the fastest way to verify an answer. The CLI remains the guaranteed
interface; the web UI exists for that one capability.

### API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status, episode count, whether a key is configured |
| `GET /api/episodes` | the catalogue |
| `GET /api/episodes/{id}` | one episode plus a timestamped outline |
| `GET /api/episodes/{id}/audio` | original audio, byte-range seekable |
| `POST /api/search` | hybrid retrieval — **no API key needed** |
| `POST /api/chat` | grounded answer with citations and sources |
| `POST /api/locate` | "take me to the part where…" |

```bash
curl -s localhost:8000/api/search -H 'Content-Type: application/json' \
  -d '{"query":"ultraviolet catastrophe","top_k":3}'
```

---

## Evaluation

```bash
make eval-retrieval   # offline: retrieval + refusal, no key, $0
make eval             # full: + generation and the LLM judge (needs a key)
make eval-compare     # baseline vs improved
```

20 cases across 8 categories, run through the **real** agent — no
evaluation-only code path, no hard-coded answers. Measured result of the one
headline improvement:

| Metric | Baseline | Improved | Δ |
|---|---|---|---|
| Pass rate | 75.0% | 85.0% | **+10.0%** |
| Refusal accuracy | 0.0% | 50.0% | **+50.0%** |
| False refusals | 0 | 0 | 0 |
| Retrieval: episode hit | 100% | 100% | 0 |
| Mean latency | 0.01 s | 0.57 s | +0.56 s |

Full methodology, three inspected failures, root causes, an attempt that made
things *worse*, and remaining weaknesses: [`EVAL.md`](EVAL.md).

---

## How it works — the decisions that matter

**ASR — faster-whisper `medium`, locally.** Free, private, and good enough that
timestamps land within a second. `medium` on Apple silicon runs ~3× realtime,
so three hours of audio takes about an hour, once. If the model can't load, the
loader steps down (`medium → small → base → tiny`) and records which model
actually ran in the transcript rather than failing.

**Chunking — timestamp-aware, not character-based.** Chunks accumulate whole
ASR segments and close when they hit ~75 s *and* land on sentence-final
punctuation, with a hard 110 s cap. Every chunk boundary is therefore a real
boundary in the audio, which is what makes a citation's timestamps meaningful.
15 s of overlap keeps an idea explained across a boundary retrievable.

**Retrieval — hybrid, fused with RRF.** Dense embeddings find paraphrases
("why energy comes in packets"); BM25 finds exact terminology ("ultraviolet
catastrophe"). They are combined with Reciprocal Rank Fusion rather than a
weighted score blend because cosine and BM25 scores live on different,
corpus-dependent scales — RRF consumes only ranks, so it needs no re-tuning
when the audio changes.

**Cross-episode questions get balanced retrieval.** In `compare` mode, results
are taken round-robin across episodes, so one strongly-matching episode can't
occupy the whole context window and leave the model "comparing" an episode with
itself.

**Refusal is two layers, not one.** A calibrated relevance gate refuses
clearly-unrelated questions *before* any model call — free, fast, and immune to
prompt manipulation. The threshold is computed at ingestion time by scoring
domain-free probe questions against the actual corpus, so it adapts to whatever
audio is supplied and is never fitted to the evaluation set. Topically adjacent
questions pass through to the grounding prompt, which can read the passages and
decide. See `EVAL.md` §6 for why one layer provably cannot work.

**Citations are validated, not trusted.** Every citation the model emits is
parsed and checked: the episode must be one that was retrieved, the interval
must be well-formed and inside the episode, and it must overlap a passage that
was actually in context. Citations that fail are **deleted from the answer** —
showing one we know is wrong is worse than showing none.

**Follow-ups are rewritten, not concatenated.** A context-dependent follow-up
is rewritten into a standalone query using the recent conversation. Messages
that already stand alone are left untouched, since rewriting a good query only
risks drifting off topic.

**Providers are swappable.** Chat, query rewriting and the judge all go through
one `LLMProvider` abstraction; Anthropic and OpenAI are both implemented and
selected by environment variable.

---

## Cost

| Component | Where it runs | Cost |
|---|---|---|
| ASR (faster-whisper) | local CPU | **free** |
| Embeddings (BGE-small) | local CPU | **free** |
| Reranking (MiniLM cross-encoder) | local CPU | **free** |
| Vector store (Chroma) + BM25 | local disk | **free** |
| Offline evaluation | local | **free** |
| Chat | API | ~$0.005–0.01 per turn (`claude-sonnet-5`) |
| Full evaluation (20 cases + judge) | API | ~$0.10–0.20 per run |

Ingestion, indexing, retrieval, refusal and the entire offline evaluation cost
nothing. The API budget is spent only on generation and judging. Actual token
counts and cost are recorded in every eval `summary.json`.

---

## Configuration

All settings live in `.env` (see `.env.example`). The ones worth knowing:

| Variable | Default | Notes |
|---|---|---|
| `CHAT_PROVIDER` / `CHAT_MODEL` | `anthropic` / `claude-sonnet-5` | `claude-opus-5` for higher quality |
| `ASR_MODEL` | `medium` | `small` if ingestion is too slow |
| `RETRIEVAL_MODE` | `hybrid` | `dense` reproduces the eval baseline |
| `ENABLE_RERANK` | `true` | `false` falls back to the cosine gate |
| `RERANK_GATE_OVERRIDE` | *(calibrated)* | raise to refuse more readily |
| `TOP_K` | `5` | passages given to the model |

---

## Project layout

```
audio/                     supplied episodes (gitignored)
data/                      generated artefacts — all reproducible
  transcripts/             our ASR output, .json + readable .txt
  chunks/                  timestamped passages
  index/                   Chroma + BM25 + the calibrated gate
src/companion/
  config.py                one place for every tunable
  models.py                Transcript / Chunk / Episode / RetrievedChunk
  errors.py                typed errors that carry a suggested fix
  ingest/                  normalize → transcribe → chunk → embed → index
  retrieve/                bm25, retriever (RRF), rerank, calibrate
  chat/                    agent, prompt, citations, session, providers
  interface/               cli.py, web.py + static UI
eval/
  cases.yaml               20 cases, 8 categories
  runner.py                drives the real pipeline
  checks.py                deterministic checks
  judge.py                 LLM-as-judge
  report.py                baseline vs improved
  results/                 committed raw evidence
tests/                     113 tests
scripts/fixtures/          synthetic dev audio scripts
```

---

## Limitations

- **Answer quality is not yet measured.** The committed evaluation runs cover
  retrieval and refusal only; the full generation + judge run needs an API key
  and has not been executed. `EVAL.md` §9 says exactly what that would add.
- **Two adjacent-topic refusals depend on the prompt**, not the deterministic
  gate — the largest open risk, and unmeasured for the same reason.
- **Follow-up retrieval is weaker without an API key**, since query rewriting
  needs the LLM.
- **The committed numbers come from a 20-minute synthetic fixture corpus**, not
  the Fermi episodes. Re-run `make ingest && make eval-baseline &&
  make eval-improved` on the real audio to regenerate them.
- **Sessions are in-process.** Restarting the server clears conversations —
  multi-user infrastructure is an explicit non-goal.
- **English-only** by default (`ASR_LANGUAGE=en`).
- **First query after startup is slow** (~10 s) while the reranker loads.
