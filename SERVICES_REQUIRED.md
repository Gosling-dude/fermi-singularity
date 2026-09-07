# Services Required

What this system actually depends on, as built.

| Service | Required? | Purpose | API key | Local alternative | Cost |
|---|---|---|---|---|---|
| **Anthropic** *(or OpenAI)* | **Yes** — for chat & judge | Answer generation, follow-up query rewriting, evaluation judge | `ANTHROPIC_API_KEY` | none — this is the only paid dependency | ~$0.005–0.01/turn; ~$0.10–0.20 per full eval run |
| **OpenAI** | Optional | Alternative chat/judge provider; optional Whisper ASR and embeddings | `OPENAI_API_KEY` | Anthropic for chat; local for ASR/embeddings | ~$0.005/turn (`gpt-4o`) |
| **faster-whisper** (ASR) | Yes | Transcribes the raw audio — *this is our own ASR pass* | No | — *is* the local option | **Free** |
| **BAAI/bge-small-en-v1.5** (embeddings) | Yes | Dense retrieval vectors | No | — *is* the local option | **Free** |
| **ms-marco-MiniLM-L-6-v2** (reranker) | Yes | Relevance reranking + the refusal gate | No | degrades gracefully to cosine gating if unavailable | **Free** |
| **ChromaDB** | Yes | Local vector store | No | — embedded, on disk | **Free** |
| **rank-bm25** | Yes | Lexical retrieval | No | — in-process library | **Free** |
| **FFmpeg** | Yes | Audio decoding/normalisation | No | — system binary | **Free** |
| **Hugging Face Hub** | First run only | Downloads the three models (~2 GB) | No | pre-populate `~/.cache/huggingface` for offline use | **Free** |

## The short version

**Exactly one paid service is required: an LLM provider for chat and the
evaluation judge.** Set `ANTHROPIC_API_KEY` *or* `OPENAI_API_KEY` (matching
`CHAT_PROVIDER`) and that is the entire external dependency.

Everything else — transcription, embeddings, reranking, the vector store,
lexical search, and the whole offline evaluation — runs locally on CPU at zero
cost. This was deliberate: the API budget should be spent on the parts that
genuinely need a frontier model, not on infrastructure.

## What works with no key at all

```bash
make ingest           # ASR + chunking + indexing
make search Q="..."   # hybrid retrieval
make eval-retrieval   # full offline evaluation of retrieval and refusal
make test             # all 113 tests
make web              # UI + /api/search and /api/episodes
```

Only `make chat` and `make eval` require a key, and both fail with an
actionable message rather than a stack trace when one is absent.

## Network access

Required once, on first run, to download the ASR, embedding and reranking
models from Hugging Face. After that the pipeline runs fully offline apart from
LLM calls.
