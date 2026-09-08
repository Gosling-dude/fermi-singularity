# eval/results/

Two generations of results live here, and they are **not comparable**:

## `fermi_*` — the real Fermi episodes (final, submitted results)

Produced from the three official Fermi "Great Papers" MP3s in `audio/`,
transcribed by this system's own faster-whisper pass. These are the numbers
quoted in `EVAL.md` and `README.md`.

| Run | Mode | What it is |
|---|---|---|
| `fermi_baseline` | retrieval-only | dense-only retrieval, no reranking (`RETRIEVAL_MODE=dense ENABLE_RERANK=false`) |
| `fermi_improved` | retrieval-only | hybrid retrieval + calibrated relevance gate (the shipped configuration) |
| `fermi_full` | full | shipped configuration with generation and the LLM judge |

`make eval-compare` renders `fermi_baseline` against `fermi_improved`.

## `synthetic_*` — historical, synthetic development corpus

Produced earlier against three locally synthesised fixture episodes
(`make fixtures`), used to build and debug the pipeline before the real audio
arrived. They are kept as a record of the development process — in particular
`synthetic_rerank_only`, a change that made things worse and was reverted.

**They are retained for provenance only. They describe a different corpus and
must not be read as results for the Fermi episodes.**
