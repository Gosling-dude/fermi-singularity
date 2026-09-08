# Fermi Podcast Companion

A conversational companion for a small collection of long-form physics
podcasts, built **from raw audio**. Ask a question, get a clear answer, and
click a timestamp to hear the exact moment that supports it — or be told
plainly that the episodes don't cover it.

Product reasoning: [`PRODUCT.md`](PRODUCT.md) ·
Evaluation, failures and the measured improvement: [`EVAL.md`](EVAL.md) ·
Dependencies and cost: [`SERVICES_REQUIRED.md`](SERVICES_REQUIRED.md) ·
3-minute demo: [`docs/DEMO.md`](docs/DEMO.md)

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

## The collection

The three supplied Fermi "Great Papers" episodes, transcribed by this system
from the raw MP3s. Nothing is hard-coded — these are simply what was in
`audio/` when `make ingest` last ran.

| # | Episode | Length | Segments | Passages |
|---|---|---|---|---|
| 01 | Einstein's Special Relativity | 52:14 | 594 | 41 |
| 09 | Bell's Theorem, 1964 | 34:00 | 399 | 26 |
| 12 | The Dirac Equation and Antimatter, 1928 | 33:00 | 357 | 25 |
| | **Total** | **1 h 59 m** | **1,350** | **92** |

Transcribed in 41 m on CPU (~2.9× realtime). The audio itself is gitignored —
it is not ours to redistribute — but every derived artefact is reproducible
with `make ingest`.

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
- **An API key** for chat and the evaluation judge — **OpenRouter** (default),
  Anthropic, or OpenAI. ASR, embeddings, reranking and the vector store all run
  locally and free.

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
make search Q="luminiferous aether"        # hybrid retrieval, free
make eval-retrieval                        # full offline evaluation, free
make test
```

### Providers

The default is **OpenRouter** — one key reaches every vendor, which is what
makes the cross-vendor judge below a config change rather than a second
account. Anthropic and OpenAI direct are also supported; set `CHAT_PROVIDER`
and the matching key.

```bash
make models              # list every model your key can reach, cheapest first
make models F=claude     # filter by substring
```

---

## Using it

```
make chat
```

```
You: What bothered Einstein about the magnet and conductor example?
...answer with inline (Ep. "…" 5:11–6:41) citations...

Sources — verify these in the audio
  • Great Papers 01 Einstein's Special Relativity  5:11–6:41
  • Great Papers 01 Einstein's Special Relativity  6:28–7:45

You: Explain that more simply.          ← follow-up, resolved from context
You: Which of these episodes mention CERN?          ← cross-episode
You: Take me to the part where they explain the light clock.
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
  -d '{"query":"luminiferous aether","top_k":3}'
```

---

## Evaluation

```bash
make eval-retrieval   # offline: retrieval + refusal, no key, $0
make eval             # full: + generation and the LLM judge (needs a key)
make eval-compare     # baseline vs improved
```

20 cases across 8 categories, run through the **real** agent — no
evaluation-only code path, no hard-coded answers.

All numbers below were measured on the **three supplied Fermi episodes**
(1 h 59 m of audio), transcribed by this system. Runs are committed under
`eval/results/fermi_*`. The `eval/results/synthetic_*` runs are from the
development fixture corpus used before the real audio arrived and are kept
only for provenance — see `eval/results/README.md`.

Offline (retrieval + refusal gate, no key, $0) — the measured effect of the one
headline improvement:

| Metric | Baseline | Improved | Δ |
|---|---|---|---|
| Pass rate | 70.0% | 90.0% | **+20.0%** |
| Refusal accuracy | 0.0% | 75.0% | **+75.0%** |
| False refusals | 0 | 1 | +1 |
| Retrieval: episode hit | 93.8% | 100% | +6.2% |
| Retrieval: term coverage | 84.4% | 84.4% | 0 |
| Mean latency | 0.01 s | 0.52 s | +0.51 s |

Full pipeline (generation + LLM judge, via OpenRouter):

| Metric | Result |
|---|---|
| Pass rate | **19/20 (95%)** |
| Refusal accuracy | **4/4 (100%)** |
| Citation validity | **100%** (0 invalid of 68) |
| Judge faithfulness | 4.85 / 5 |
| Judge completeness · citations · refusal · clarity | 5.00 / 5 each |
| Actual cost | $0.27 per run |

Full methodology, three inspected failures, root causes, an attempt that made
things *worse*, two defects the evaluation found in my own harness and product,
and the one real remaining hallucination: [`EVAL.md`](EVAL.md).

---

## How it works — the decisions that matter

**ASR — faster-whisper `medium`, locally.** Free, private, and good enough that
timestamps land within a second. Measured on the supplied episodes: 1 h 59 m of
audio transcribed in 41 m on Apple silicon CPU — **~2.9× realtime**, 1,350
segments. If the model can't load, the loader steps down
(`medium → small → base → tiny`) and records which model actually ran in the
transcript rather than failing.

**Chunking — timestamp-aware, not character-based.** Chunks accumulate whole
ASR segments and close when they hit ~75 s *and* land on sentence-final
punctuation, with a hard 110 s cap. Every chunk boundary is therefore a real
boundary in the audio, which is what makes a citation's timestamps meaningful.
15 s of overlap keeps an idea explained across a boundary retrievable.

**Retrieval — hybrid, fused with RRF.** Dense embeddings find paraphrases
("why a fast watch ticks slowly"); BM25 finds exact terminology
("luminiferous aether"). They are combined with Reciprocal Rank Fusion rather
than a weighted score blend because cosine and BM25 scores live on different,
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
one `LLMProvider` abstraction; OpenRouter, Anthropic and OpenAI are implemented
and selected by environment variable. OpenRouter is an OpenAI-compatible
endpoint, so it reuses the official SDK pointed at a different base URL rather
than a bespoke HTTP client, and it reports the **actual** cost of each call,
which is recorded instead of a locally estimated price.

**The judge runs on a different vendor from the chat model.** A judge grading
its own family's output is prone to self-preference bias. Defaults are
`anthropic/claude-sonnet-5` for chat and `openai/gpt-5-mini` for judging — one
key, two vendors, no extra setup.

---

## Cost

| Component | Where it runs | Cost |
|---|---|---|
| ASR (faster-whisper) | local CPU | **free** |
| Embeddings (BGE-small) | local CPU | **free** |
| Reranking (MiniLM cross-encoder) | local CPU | **free** |
| Vector store (Chroma) + BM25 | local disk | **free** |
| Offline evaluation | local | **free** |
| Chat | API | ~$0.015 per turn (`anthropic/claude-sonnet-5`) |
| Full evaluation (20 cases + judge) | API | **$0.30 per run** (measured) |

Ingestion, indexing, retrieval, refusal and the entire offline evaluation cost
nothing. The API budget is spent only on generation and judging. OpenRouter
reports the true cost of every call, and it is recorded per case in
`eval/results/*/raw.jsonl` — the $0.30 above is measured, not estimated.

---

## Configuration

All settings live in `.env` (see `.env.example`). The ones worth knowing:

| Variable | Default | Notes |
|---|---|---|
| `CHAT_PROVIDER` | `openrouter` | or `anthropic` / `openai` direct |
| `CHAT_MODEL` | `anthropic/claude-sonnet-5` | `make models` lists valid ids |
| `JUDGE_MODEL` | `openai/gpt-5-mini` | different vendor on purpose |
| `ASR_MODEL` | `medium` | `small` if ingestion is too slow |
| `RETRIEVAL_MODE` | `hybrid` | `dense` reproduces the eval baseline |
| `ENABLE_RERANK` | `true` | `false` falls back to the cosine gate |
| `RERANK_GATE_OVERRIDE` | *(calibrated)* | raise to refuse more readily |
| `TOP_K` | `5` | passages given to the model |

---

## Project layout

```
README.md                  this file — setup, usage, results
PRODUCT.md                 intended user, the problem, why this framing
EVAL.md                    evaluation system, failures, measured improvement
SERVICES_REQUIRED.md       every dependency, what needs a key, what it costs
docs/DEMO.md               the 3-minute demo script
Makefile                   every command in the project

audio/                     supplied episodes (gitignored — not ours to ship)
data/                      generated artefacts — all reproducible via make ingest
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
  cases.yaml               20 cases, 8 categories (written against the real audio)
  runner.py                drives the real pipeline
  checks.py                deterministic checks
  judge.py                 LLM-as-judge
  report.py                baseline vs improved
  results/fermi_*          final results, real Fermi episodes
  results/synthetic_*      historical dev-corpus runs, kept for provenance
tests/                     147 tests
scripts/fixtures/          synthetic dev audio scripts
```

---

## Limitations

- **ASR quality is a retrieval ceiling on proper nouns.** Whisper `medium`
  transcribes "Michelson-Morley" as "Mickelson-Morley". Asking with the correct
  spelling still returns the right passage first — the hyphen tokenises and
  "Morley" still matches — but at roughly half the BM25 score. Names are where
  ASR error quietly becomes retrieval error.
- **Follow-up retrieval is weaker without an API key**, since query rewriting
  needs the LLM. `make eval-retrieval` deliberately makes no LLM call at all,
  so its numbers are reproducible but pessimistic about follow-ups — this is
  the single offline failure (`followup_03`), and it passes in the full run.
- **The judge is unvalidated** against human labels, so treat its scores as a
  signal rather than a verdict.
- **Full-pipeline results vary run to run.** Generation is non-deterministic,
  and a borderline case can land either side of the line between runs. The
  offline numbers are deterministic; the full numbers are one measured sample.
- **Sessions are in-process.** Restarting the server clears conversations —
  multi-user infrastructure is an explicit non-goal.
- **English-only** by default (`ASR_LANGUAGE=en`).
- **First query after startup is slow** (~10 s) while the reranker loads.
