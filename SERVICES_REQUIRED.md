# Services Required

What this system actually depends on, as built.

| Service | Required? | Purpose | API key | Local alternative | Cost |
|---|---|---|---|---|---|
| **OpenRouter** | **Yes** *(default)* | Answer generation, follow-up query rewriting, evaluation judge — one key reaches every vendor | `OPENROUTER_API_KEY` | none — this is the only paid dependency | ~$0.015/turn; **$0.27** per full eval run (measured) |
| Anthropic *(direct)* | Optional | Alternative chat/judge provider | `ANTHROPIC_API_KEY` | use OpenRouter instead | ~$0.015/turn |
| OpenAI *(direct)* | Optional | Alternative chat/judge provider; optional Whisper ASR and embeddings | `OPENAI_API_KEY` | use OpenRouter; local for ASR/embeddings | ~$0.005/turn |
| **faster-whisper** (ASR) | Yes | Transcribes the raw audio — *this is our own ASR pass* | No | — *is* the local option | **Free** |
| **BAAI/bge-small-en-v1.5** (embeddings) | Yes | Dense retrieval vectors | No | — *is* the local option | **Free** |
| **ms-marco-MiniLM-L-6-v2** (reranker) | Yes | Relevance reranking + the refusal gate | No | degrades gracefully to cosine gating if unavailable | **Free** |
| **ChromaDB** | Yes | Local vector store | No | — embedded, on disk | **Free** |
| **rank-bm25** | Yes | Lexical retrieval | No | — in-process library | **Free** |
| **FFmpeg** | Yes | Audio decoding/normalisation | No | — system binary | **Free** |
| **Hugging Face Hub** | First run only | Downloads the three models (~2 GB) | No | pre-populate `~/.cache/huggingface` for offline use | **Free** |

## The short version

**Exactly one paid service is required: OpenRouter.** Set `OPENROUTER_API_KEY`
and that is the entire external dependency — no Anthropic or OpenAI account is
needed.

Everything else — transcription, embeddings, reranking, the vector store,
lexical search, and the whole offline evaluation — runs locally on CPU at zero
cost. This was deliberate: the API budget should be spent on the parts that
genuinely need a frontier model, not on infrastructure.

## Why OpenRouter

- **One key, every vendor.** Switching `CHAT_MODEL` between Anthropic, OpenAI,
  Google and open-weight models is a one-line `.env` change.
- **It enables a cross-vendor judge.** The evaluation judge deliberately runs
  on a *different* vendor from the chat model, because a judge grading its own
  family's output is prone to self-preference bias. On a single-vendor key that
  would need a second account; here it is one line.
- **It reports real cost.** Each call returns the actual credits spent, so
  `eval/results/*/summary.json` records measured spend rather than a
  hard-coded price table that can silently go stale.

## Models used

Both verified present in OpenRouter's live catalogue (`make models`):

| Role | Model | $/1M in | $/1M out |
|---|---|---|---|
| Chat | `anthropic/claude-sonnet-5` | $2.00 | $10.00 |
| Judge | `openai/gpt-5-mini` | $0.25 | $2.00 |

Neither is hard-coded anywhere in the source — both are `.env` values, and
`make models` lists valid alternatives. `make doctor` verifies the configured
ids against the live catalogue before you spend anything on a failed run.

## What works with no key at all

```bash
make ingest           # ASR + chunking + indexing
make search Q="..."   # hybrid retrieval
make eval-retrieval   # full offline evaluation of retrieval and refusal
make test             # the whole test suite
make web              # UI + /api/search and /api/episodes
```

Only `make chat` and `make eval` require a key, and both fail with an
actionable message rather than a stack trace when one is absent.

## Network access

Required once, on first run, to download the ASR, embedding and reranking
models from Hugging Face. After that the pipeline runs fully offline apart from
LLM calls.
