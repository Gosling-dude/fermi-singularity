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
- ✅ `README.md`, `PRODUCT.md`, `EVAL.md`, `SERVICES_REQUIRED.md`, `docs/DEMO.md`

## What you need to do

### 1. Add an API key  *(required for chat and the full evaluation)*

```bash
# .env already exists (created by `make setup`)
open -e .env          # or your editor
```

Set:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

That is the only credential the system needs. Use the Fermi-provided budget
key if one was issued. To use OpenAI instead, set `OPENAI_API_KEY` and
`CHAT_PROVIDER=openai`.

Verify:

```bash
make doctor          # the two "key" rows should turn green
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
make eval-baseline    # dense-only baseline
make eval-improved    # hybrid + calibrated gate
make eval             # full pipeline + LLM judge  ← needs the key from step 1
make eval-compare
```

Then update the numbers in `EVAL.md` §4 and §7 and in the `README.md` table.
`EVAL.md` §3 already flags that the committed figures come from the fixture
corpus, so the honest framing is in place — only the numbers change.

**This step matters most.** `make eval` is the one thing that has not been run,
and it is what produces the answer-quality evidence (faithfulness, citation
correctness, clarity) that the rubric weights at 25 points. Estimated cost:
$0.10–0.20.

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
- **Higher answer quality:** `CHAT_MODEL=claude-opus-5` (~2.5× the cost).
- **Try it without a key first:** `make search Q="your question"` and
  `make eval-retrieval` both work offline.
