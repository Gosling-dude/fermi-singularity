# Manual Setup

## Already done — no action needed

The whole system is built, run and verified on this machine:

- ✅ Repository, dependencies, `pyproject.toml`, `Makefile`, `.env.example`, `.gitignore`
- ✅ FFmpeg installed
- ✅ Python 3.12 virtualenv at `.venv/` with all dependencies
- ✅ Full ingestion pipeline: discovery → SHA-256 → FFmpeg normalise → ASR →
  chunking → embeddings → Chroma → BM25 → gate calibration
- ✅ **Ingestion actually run** — 3 episodes, 20m22s of audio, 317 ASR
  segments, 19 chunks, ASR at ~3× realtime
- ✅ Hybrid retrieval (dense + BM25 + RRF), cross-encoder reranking,
  episode filtering, compare mode
- ✅ Grounded agent: prompt contract, refusal, citation parsing + validation,
  multi-turn sessions, query rewriting, locate mode
- ✅ CLI (`chat`, `ingest`, `search`, `episodes`, `doctor`) and web UI + JSON API
- ✅ Evaluation system: 20 cases, deterministic checks, LLM judge, runner,
  comparison report
- ✅ **Baseline and improved evaluation runs executed**, with raw evidence
  committed under `eval/results/`
- ✅ **113 tests passing**
- ✅ **OpenRouter provider** — verified with real API calls on both the chat and
  judge models; `make models` and `make doctor` validate model ids live
- ✅ **Full evaluation executed through OpenRouter** — 19/20 (95%), refusal
  accuracy 4/4, citation validity 100% of 68, measured cost $0.27
- ✅ `README.md`, `PRODUCT.md`, `EVAL.md`, `SERVICES_REQUIRED.md`, `docs/DEMO.md`

## What you need to do

### 1. Add your OpenRouter key  *(required for chat and the full evaluation)*

```bash
open -e .env          # .env already exists
```

Set exactly one line:

```env
OPENROUTER_API_KEY=sk-or-v1-...
```

That is the **only** credential the system needs — no Anthropic or OpenAI
account. The provider and both models are already configured for OpenRouter.

Verify:

```bash
make doctor          # both "key" rows turn green, and the configured
                     # CHAT_MODEL / JUDGE_MODEL ids are checked against
                     # OpenRouter's live catalogue
```

### 2. Swap in the official audio  *(required before submitting)*

The repository currently contains **synthetic development fixtures**, not Fermi
episodes. Replace them:

```bash
rm -f audio/*.mp3
cp /path/to/the/three/episodes/*.mp3 audio/
make ingest          # ~1 hour of ASR for 3 hours of audio; runs unattended
```

Name the files as you want the episode titles to read — titles are derived
from filenames, and they appear in every citation. A leading track number
(`01 - Title.mp3`) is stripped automatically.

### 3. Re-run the evaluation on the real audio

```bash
make eval-baseline    # dense-only baseline      (free)
make eval-improved    # hybrid + calibrated gate (free)
make eval             # full pipeline + LLM judge (~$0.27)
make eval-compare
```

Then update the numbers in `EVAL.md` §4, §7 and §9, and the two `README.md`
tables. `EVAL.md` §3 already flags that the committed figures come from the
fixture corpus, so the honest framing is in place — only the numbers change.

Note that the calibrated refusal gate is recomputed automatically during
`make ingest`, so it adapts to the real episodes with no manual tuning.

### 4. Record the demo

```bash
make demo            # prints the exact 3-minute script
```

Follow it top to bottom; it is sequenced so nothing needs improvising.

### 5. Submit

```bash
git add -A && git commit -m "Fermi Podcast Companion"
```

`.gitignore` already excludes `.env`, the audio, and the generated `data/`
directory, while keeping the evaluation evidence. Confirm before pushing:

```bash
git status --porcelain | grep -E '\.env$|\.mp3$' || echo "clean — no secrets or audio staged"
```

## Optional

- **Faster ingestion:** `ASR_MODEL=small` in `.env` (lower transcript quality).
- **Higher answer quality:** `CHAT_MODEL=anthropic/claude-opus-5` (~2.5× cost).
- **Cheaper:** `CHAT_MODEL=openai/gpt-5-mini` (~8× cheaper than Sonnet).
- **See what else your key can reach:** `make models` / `make models F=gemini`.
- **Try it without a key first:** `make search Q="your question"` and
  `make eval-retrieval` both work offline.
